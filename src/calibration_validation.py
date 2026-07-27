"""Validate non-leaking isotonic PD calibration on a later OOT population."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, auc, brier_score_loss, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fannie_mae_pipeline import PROJECT_ROOT
from oot_risk_modelling import EXCLUDED_COLUMNS, TARGET_COLUMN, configure_plot_font, ks_statistic, stratified_training_sample


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SNAPSHOT_PATH = PROCESSED_DIR / "fannie_pd_snapshot.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "calibration_validation_report.md"


def feature_groups(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return feature names excluding identifiers, dates and the realized outcome."""
    candidates = [column for column in frame.columns if column not in EXCLUDED_COLUMNS]
    numeric = [column for column in candidates if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in candidates if column not in numeric]
    return numeric, categorical


def make_xgboost_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    """Use the same XGBoost specification as the current strategy candidate."""
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    transformers: list[tuple[str, Pipeline, list[str]]] = [("numeric", numeric_pipe, numeric)]
    if categorical:
        categorical_pipe = Pipeline(
            [("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=25))]
        )
        transformers.append(("categorical", categorical_pipe, categorical))
    return Pipeline(
        [
            ("preprocessor", ColumnTransformer(transformers)),
            (
                "model",
                xgb.XGBClassifier(
                    objective="binary:logistic",
                    n_estimators=250,
                    learning_rate=0.05,
                    max_depth=4,
                    min_child_weight=100,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=4,
                    tree_method="hist",
                    eval_metric="logloss",
                ),
            ),
        ]
    )


def expected_calibration_error(y_true: pd.Series, score: np.ndarray, bins: int = 10) -> float:
    """Calculate equal-frequency ECE: average absolute PD-to-observed-rate gap."""
    data = pd.DataFrame({"bad": np.asarray(y_true), "score": score})
    data["bin"] = pd.qcut(data["score"].rank(method="first"), q=bins, duplicates="drop")
    grouped = data.groupby("bin", observed=False).agg(predicted_pd=("score", "mean"), observed_bad_rate=("bad", "mean"), count=("bad", "size"))
    return float((grouped["count"] * (grouped["predicted_pd"] - grouped["observed_bad_rate"]).abs()).sum() / len(data))


def evaluate(y_true: pd.Series, score: np.ndarray) -> dict[str, float]:
    """Evaluate discrimination and probability calibration without selecting on OOT labels."""
    fpr, tpr, _ = roc_curve(y_true, score)
    return {
        "auc": float(auc(fpr, tpr)),
        "ks": ks_statistic(y_true, score),
        "pr_auc": float(average_precision_score(y_true, score)),
        "brier_score": float(brier_score_loss(y_true, score)),
        "ece": expected_calibration_error(y_true, score),
    }


def calibration_table(y_true: pd.Series, score: np.ndarray, score_type: str) -> pd.DataFrame:
    """Build a score-decile table for the report and Streamlit chart."""
    data = pd.DataFrame({"bad": np.asarray(y_true), "score": score})
    data["decile"] = pd.qcut(data["score"].rank(method="first"), q=10, labels=[f"D{i}" for i in range(1, 11)])
    table = data.groupby("decile", observed=False).agg(sample_count=("bad", "size"), predicted_pd=("score", "mean"), observed_bad_rate=("bad", "mean")).reset_index()
    table.insert(0, "score_type", score_type)
    return table


def save_figure(y_true: pd.Series, raw_score: np.ndarray, calibrated_score: np.ndarray) -> None:
    """Save a bilingual OOT calibration comparison visual."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    configure_plot_font()
    fig, axis = plt.subplots(figsize=(8.5, 6))
    for score, label, color in [
        (raw_score, "Raw XGBoost / 原始 XGBoost", "#bc6c25"),
        (calibrated_score, "Isotonic calibrated / Isotonic 校准后", "#2a6f97"),
    ]:
        observed, predicted = calibration_curve(y_true, score, n_bins=10, strategy="quantile")
        axis.plot(predicted, observed, marker="o", linewidth=2, label=label, color=color)
    axis.plot([0, 1], [0, 1], linestyle="--", color="#6c757d", label="Perfect calibration / 完美校准")
    axis.set_title("OOT probability calibration / 时间外概率校准")
    axis.set_xlabel("Average predicted PD / 平均预测 PD")
    axis.set_ylabel("Observed bad rate / 实际坏样本率")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "xgboost_calibration_comparison.png", dpi=160)
    plt.close(fig)


def write_report(
    model_train: pd.DataFrame,
    calibration: pd.DataFrame,
    oot: pd.DataFrame,
    calibration_quarter: str,
    metrics: pd.DataFrame,
) -> None:
    """Create a compact bilingual calibration-validation report."""
    raw, calibrated = metrics.set_index("score_version").loc[["Raw XGBoost / 原始分数", "Isotonic calibrated / 校准后分数"]].to_dict("index").values()
    report = f"""# Probability Calibration Validation / 概率校准验证报告

## Non-leaking time design / 不泄漏的时间设计

The base XGBoost model is fitted only on observations before **{calibration_quarter}**. The {calibration_quarter} cohort is reserved only to learn an isotonic score-to-PD mapping. The final evaluation remains 2024Q2-2024Q3, and its labels were never used for fitting or calibration. 基础 XGBoost 只在 **{calibration_quarter}** 之前的观察样本上训练；{calibration_quarter} 批次仅用于学习从分数到 PD 的 Isotonic 映射；最终评估仍是 2024Q2-2024Q3，其标签从未参与训练或校准。

| Population / 样本 | Count / 数量 | Bad rate / 坏样本率 |
| --- | ---: | ---: |
| Model training / 模型训练 | {len(model_train):,} | {model_train[TARGET_COLUMN].mean():.2%} |
| Calibration cohort / 校准批次 ({calibration_quarter}) | {len(calibration):,} | {calibration[TARGET_COLUMN].mean():.2%} |
| Final OOT / 最终 OOT (2024Q2-Q3) | {len(oot):,} | {oot[TARGET_COLUMN].mean():.2%} |

## Final OOT evidence / 最终 OOT 证据

| Score version / 分数版本 | AUC | KS | PR-AUC | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw XGBoost / 原始分数 | {raw['auc']:.3f} | {raw['ks']:.3f} | {raw['pr_auc']:.3f} | {raw['brier_score']:.5f} | {raw['ece']:.4f} |
| Isotonic calibrated / 校准后分数 | {calibrated['auc']:.3f} | {calibrated['ks']:.3f} | {calibrated['pr_auc']:.3f} | {calibrated['brier_score']:.5f} | {calibrated['ece']:.4f} |

**Decision / 决策：** Calibration is judged primarily by lower Brier score and ECE; AUC and KS are expected to remain approximately unchanged because calibration is a monotonic mapping. This study validates probability quality only. It does **not** silently replace the stored strategy scores or thresholds. 校准主要以更低的 Brier 和 ECE 判断；因为校准是单调映射，AUC 和 KS 理应大致不变。本研究只验证概率质量，**不会**悄悄替换当前策略使用的分数或阈值。

![OOT probability calibration](figures/xgboost_calibration_comparison.png)

## Metric definitions / 指标定义

- **Brier score:** mean squared error between predicted PD and realized bad flag; lower is better. Brier 分数：预测 PD 与实际坏样本标记的均方误差，越低越好。
- **ECE:** weighted average absolute difference between predicted PD and observed bad rate by score decile; lower is better. ECE：按分数十分位计算的预测 PD 与实际坏样本率绝对差的加权平均，越低越好。
- This is validation evidence, not a claim that the mapping is production-calibrated forever; it must be rechecked on later cohorts. 这是验证证据，不代表该映射永久适用于生产环境；后续批次仍需复核。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    """Train earlier, calibrate on the next quarter, and validate only on later OOT quarters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_PATH)
    parser.add_argument("--oot-start", default="2024-04-01")
    parser.add_argument("--max-train-rows", type=int, default=750_000)
    args = parser.parse_args()

    data = pd.read_csv(args.snapshot, low_memory=False)
    data["observation_date"] = pd.to_datetime(data["observation_date"], errors="coerce")
    data = data.dropna(subset=["observation_date", TARGET_COLUMN]).sort_values("observation_date")
    oot_start = pd.Timestamp(args.oot_start)
    pre_oot = data.loc[data["observation_date"] < oot_start].copy()
    oot = data.loc[data["observation_date"] >= oot_start].copy()
    calibration_quarter = str(pre_oot["observation_date"].dt.to_period("Q").max())
    calibration = pre_oot.loc[pre_oot["observation_date"].dt.to_period("Q").astype(str) == calibration_quarter].copy()
    model_train = pre_oot.loc[pre_oot["observation_date"].dt.to_period("Q").astype(str) < calibration_quarter].copy()
    if model_train.empty or calibration.empty or oot.empty:
        raise ValueError("Expected model-training, calibration and final OOT populations.")
    if any(frame[TARGET_COLUMN].nunique() < 2 for frame in [model_train, calibration, oot]):
        raise ValueError("Each population requires both good and bad outcomes.")

    fit_train = stratified_training_sample(model_train, args.max_train_rows)
    numeric, categorical = feature_groups(fit_train)
    features = numeric + categorical
    model = make_xgboost_pipeline(numeric, categorical)
    model.fit(fit_train[features], fit_train[TARGET_COLUMN].astype(int))

    calibration_raw = model.predict_proba(calibration[features])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(calibration_raw, calibration[TARGET_COLUMN].astype(int))
    oot_raw = model.predict_proba(oot[features])[:, 1]
    oot_calibrated = calibrator.predict(oot_raw)

    metrics = pd.DataFrame(
        [
            {"score_version": "Raw XGBoost / 原始分数", **evaluate(oot[TARGET_COLUMN].astype(int), oot_raw)},
            {"score_version": "Isotonic calibrated / 校准后分数", **evaluate(oot[TARGET_COLUMN].astype(int), oot_calibrated)},
        ]
    )
    tables = pd.concat(
        [
            calibration_table(oot[TARGET_COLUMN], oot_raw, "Raw XGBoost / 原始分数"),
            calibration_table(oot[TARGET_COLUMN], oot_calibrated, "Isotonic calibrated / 校准后分数"),
        ],
        ignore_index=True,
    )
    score_output = oot[["loan_id", "origination_date", "observation_date", TARGET_COLUMN]].copy()
    score_output["xgboost_raw_pd"] = oot_raw
    score_output["xgboost_isotonic_pd"] = oot_calibrated

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(PROCESSED_DIR / "calibration_validation_performance.csv", index=False)
    tables.to_csv(PROCESSED_DIR / "calibration_validation_curve.csv", index=False)
    score_output.to_csv(PROCESSED_DIR / "oot_calibration_study_scores.csv", index=False)
    save_figure(oot[TARGET_COLUMN].astype(int), oot_raw, oot_calibrated)
    write_report(fit_train, calibration, oot, calibration_quarter, metrics)
    print(f"Calibration held out {calibration_quarter}; final OOT rows: {len(oot):,}.")
    print(f"Calibration report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()

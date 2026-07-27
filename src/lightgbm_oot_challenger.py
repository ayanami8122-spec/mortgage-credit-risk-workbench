"""Train LightGBM as an isolated, reproducible OOT challenger model."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, auc, brier_score_loss, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "lightgbm_challenger_report.md"
SNAPSHOT_PATH = PROCESSED_DIR / "fannie_pd_snapshot.csv"
TARGET = "bad_12m_90dpd"
EXCLUDED = {"loan_id", "origination_date", "observation_date", TARGET}
CHINESE_FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")


def configure_plot_font() -> None:
    """Configure CJK-capable plotting so generated charts remain bilingual."""
    if CHINESE_FONT_PATH.exists():
        font_manager.fontManager.addfont(str(CHINESE_FONT_PATH))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=CHINESE_FONT_PATH).get_name()
    else:
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def split_out_of_time(snapshot: pd.DataFrame, test_start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the same chronological OOT split used by the existing models."""
    data = snapshot.copy()
    data["observation_date"] = pd.to_datetime(data["observation_date"], errors="coerce")
    data = data.dropna(subset=["observation_date", TARGET]).sort_values("observation_date")
    train = data.loc[data["observation_date"] < test_start].copy()
    test = data.loc[data["observation_date"] >= test_start].copy()
    if train.empty or test.empty or train[TARGET].nunique() < 2 or test[TARGET].nunique() < 2:
        raise ValueError("OOT split needs non-empty train/test populations with both target classes.")
    return train, test


def stratified_training_sample(train: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Use the same target-rate-preserving training cap as the XGBoost experiment."""
    if len(train) <= max_rows:
        return train
    sampled = []
    for _, group in train.groupby(TARGET, sort=False):
        count = round(max_rows * len(group) / len(train))
        sampled.append(group.sample(n=count, random_state=42))
    return pd.concat(sampled, ignore_index=True).sort_values("observation_date")


def feature_groups(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Derive numerical and categorical predictor lists without IDs, dates or label."""
    features = [column for column in frame.columns if column not in EXCLUDED]
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in features if column not in numeric]
    return numeric, categorical


def preprocessing(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Fit all imputation and encoding on the training population only."""
    transformers: list[tuple[str, Pipeline, list[str]]] = [
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric)
    ]
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=25)),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers)


def evaluate(y_true: pd.Series, score: np.ndarray) -> dict[str, float]:
    """Return the same OOT evaluation metrics used for the existing models."""
    fpr, tpr, _ = roc_curve(y_true, score)
    return {
        "auc": float(auc(fpr, tpr)),
        "ks": float(np.max(tpr - fpr)),
        "pr_auc": float(average_precision_score(y_true, score)),
        "brier_score": float(brier_score_loss(y_true, score)),
    }


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Calculate score PSI with bins fixed on the fitted training population."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    expected_share = pd.Series(pd.cut(expected, bins=edges, include_lowest=True)).value_counts(sort=False) / len(expected)
    actual_share = pd.Series(pd.cut(actual, bins=edges, include_lowest=True)).value_counts(sort=False) / len(actual)
    expected_share = expected_share.clip(lower=0.0001)
    actual_share = actual_share.clip(lower=0.0001)
    return float(((actual_share - expected_share) * np.log(actual_share / expected_share)).sum())


def save_figures(y_test: pd.Series, score: np.ndarray, importance: pd.DataFrame) -> None:
    """Write bilingual ROC/calibration and gain-importance charts."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    configure_plot_font()

    fpr, tpr, _ = roc_curve(y_test, score)
    calibration = pd.DataFrame({"score": score, "bad": y_test}).groupby(
        pd.qcut(pd.Series(score).rank(method="first"), 10), observed=False
    ).agg({"score": "mean", "bad": "mean"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(fpr, tpr, color="#3a86ff", linewidth=2, label=f"LightGBM (AUC={auc(fpr, tpr):.3f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#6c757d")
    axes[0].set_title("LightGBM OOT ROC curve / LightGBM 时间外 ROC 曲线")
    axes[0].set_xlabel("False positive rate / 假阳性率")
    axes[0].set_ylabel("True positive rate / 真阳性率")
    axes[0].legend()
    axes[1].plot(calibration["score"], calibration["bad"], marker="o", color="#3a86ff", label="LightGBM / LightGBM")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="#6c757d")
    axes[1].set_title("Calibration by score decile / 分数十分位校准")
    axes[1].set_xlabel("Average predicted PD / 平均预测 PD")
    axes[1].set_ylabel("Observed bad rate / 观察坏样本率")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "lightgbm_oot_roc_and_calibration.png", dpi=160)
    plt.close(fig)

    top = importance.head(15).sort_values("gain")
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.barh(top["feature"], top["gain"], color="#3a86ff")
    axis.set_title("LightGBM feature importance by gain / LightGBM 增益特征重要性")
    axis.set_xlabel("Gain / 增益")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "lightgbm_feature_importance.png", dpi=160)
    plt.close(fig)


def write_report(metrics: dict[str, float], train: pd.DataFrame, test: pd.DataFrame, psi: float, importance: pd.DataFrame, n_estimators: int) -> None:
    """Write a concise bilingual model-governance record for the challenger."""
    existing = pd.read_csv(PROCESSED_DIR / "oot_model_performance.csv")
    comparison = pd.concat(
        [
            existing,
            pd.DataFrame([{"model": "LightGBM / LightGBM", **metrics}]),
        ],
        ignore_index=True,
    )
    rows = "\n".join(
        f"| {row.model} | {row.auc:.3f} | {row.ks:.3f} | {row.pr_auc:.3f} | {row.brier_score:.3f} |"
        for row in comparison.itertuples(index=False)
    )
    importance_rows = "\n".join(
        f"| {row.feature} | {row.gain:.3f} |" for row in importance.head(10).itertuples(index=False)
    )
    report = f"""# LightGBM OOT Challenger Report / LightGBM 时间外挑战者报告

## Protocol / 比较协议

LightGBM uses the same point-in-time snapshot, target definition, `2024-04-01` OOT cutoff and stratified training cap as the existing model experiment. It is a challenger, not an automatically selected production model. LightGBM 使用与现有实验相同的时点快照、标签定义、`2024-04-01` OOT 切分点和分层训练上限。它是挑战者，不会因单个指标更高而自动成为生产模型。

| Population / 样本 | Count / 数量 | Bad rate / 坏样本率 |
| --- | ---: | ---: |
| LightGBM train / LightGBM 训练集 | {len(train):,} | {train[TARGET].mean():.2%} |
| OOT test / 时间外测试集 | {len(test):,} | {test[TARGET].mean():.2%} |

Parameters: `n_estimators={n_estimators}`, `learning_rate=0.05`, `num_leaves=31`, `min_child_samples=100`. 参数如上；其目的为可复现比较，不表示完成了全面调参。

## OOT comparison / 时间外比较

| Model / 模型 | AUC | KS | PR-AUC | Brier score / Brier 分数 |
| --- | ---: | ---: | ---: | ---: |
{rows}

![LightGBM OOT ROC and calibration](figures/lightgbm_oot_roc_and_calibration.png)

## Stability / 稳定性

LightGBM train-to-OOT score PSI is **{psi:.3f}**. This is a score-distribution signal and must be considered with calibration, OOT metrics and feature drift. LightGBM 训练集到 OOT 测试集的分数 PSI 为 **{psi:.3f}**。它只是分数分布信号，必须结合校准、OOT 指标和特征漂移判断。

## Feature importance / 特征重要性

| Feature / 特征 | Gain / 增益 |
| --- | ---: |
{importance_rows}

![LightGBM feature importance](figures/lightgbm_feature_importance.png)

## Decision / 决策

Keep LightGBM as a challenger unless it provides a meaningful OOT performance or runtime/stability advantage and its calibration remains acceptable. 若 LightGBM 没有展现有意义的 OOT 性能或运行效率/稳定性优势，或校准不可接受，则保留其挑战者地位，不替换当前 XGBoost 候选模型。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    """Fit LightGBM on the prepared snapshot and write challenger artefacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_PATH)
    parser.add_argument("--test-start", default="2024-04-01")
    parser.add_argument("--max-train-rows", type=int, default=750_000)
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()

    snapshot = pd.read_csv(args.snapshot, low_memory=False)
    train, test = split_out_of_time(snapshot, args.test_start)
    fit_train = stratified_training_sample(train, args.max_train_rows)
    numeric, categorical = feature_groups(fit_train)
    features = numeric + categorical
    transformer = preprocessing(numeric, categorical)
    x_train = transformer.fit_transform(fit_train[features])
    x_test = transformer.transform(test[features])
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=args.n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
        verbosity=-1,
        importance_type="gain",
    )
    model.fit(x_train, fit_train[TARGET])
    score = model.predict_proba(x_test)[:, 1]
    train_score = model.predict_proba(x_train)[:, 1]
    metrics = evaluate(test[TARGET], score)
    psi = population_stability_index(train_score, score)

    transformed_names = transformer.get_feature_names_out()
    importance = pd.DataFrame({"feature": transformed_names, "gain": model.feature_importances_})
    importance["feature"] = importance["feature"].str.replace("numeric__", "", regex=False).str.replace("categorical__", "", regex=False)
    importance = importance.sort_values("gain", ascending=False, ignore_index=True)
    score_columns = ["loan_id", "origination_date", "observation_date", TARGET, "original_upb"]
    scored = test[score_columns].copy()
    scored["lightgbm_pd"] = score

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"model": "LightGBM / LightGBM", **metrics}]).to_csv(PROCESSED_DIR / "lightgbm_oot_model_performance.csv", index=False)
    scored.to_csv(PROCESSED_DIR / "lightgbm_oot_test_scores.csv", index=False)
    importance.to_csv(PROCESSED_DIR / "lightgbm_feature_importance.csv", index=False)
    pd.DataFrame([{"metric": "lightgbm_score_psi", "value": psi}]).to_csv(PROCESSED_DIR / "lightgbm_score_stability.csv", index=False)
    existing = pd.read_csv(PROCESSED_DIR / "oot_model_performance.csv")
    pd.concat([existing, pd.DataFrame([{"model": "LightGBM / LightGBM", **metrics}])], ignore_index=True).to_csv(
        PROCESSED_DIR / "champion_challenger_performance.csv", index=False
    )
    save_figures(test[TARGET], score, importance)
    write_report(metrics, fit_train, test, psi, importance, args.n_estimators)
    print(f"Trained LightGBM challenger: {len(fit_train):,} train loans, {len(test):,} OOT test loans.")
    print(f"OOT metrics: {metrics}; score PSI: {psi:.3f}")
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()

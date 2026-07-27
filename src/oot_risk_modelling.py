"""Train and evaluate logistic-regression and XGBoost PD models out of time."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import font_manager
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, auc, brier_score_loss, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fannie_mae_pipeline import PROJECT_ROOT


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SNAPSHOT_PATH = PROCESSED_DIR / "fannie_pd_snapshot.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "oot_model_report.md"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TARGET_COLUMN = "bad_12m_90dpd"
EXCLUDED_COLUMNS = {"loan_id", "origination_date", "observation_date", TARGET_COLUMN}
CHINESE_FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")


def configure_plot_font() -> None:
    """Set a CJK-capable font after Matplotlib's style configuration."""
    if CHINESE_FONT_PATH.exists():
        font_manager.fontManager.addfont(str(CHINESE_FONT_PATH))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=CHINESE_FONT_PATH).get_name()
    else:
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def split_out_of_time(snapshot: pd.DataFrame, test_start: str | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split cohorts chronologically, never randomly, using observation date."""
    data = snapshot.copy()
    data["observation_date"] = pd.to_datetime(data["observation_date"], errors="coerce")
    data = data.dropna(subset=["observation_date", TARGET_COLUMN]).sort_values("observation_date")
    if test_start:
        cutoff = pd.Timestamp(test_start)
    else:
        cohorts = pd.Series(data["observation_date"].dt.to_period("Q").unique()).sort_values()
        if len(cohorts) < 4:
            raise ValueError("At least four observation quarters are required for an automatic OOT split.")
        cutoff = cohorts.iloc[max(1, int(len(cohorts) * 0.8))].start_time
    train = data.loc[data["observation_date"] < cutoff].copy()
    test = data.loc[data["observation_date"] >= cutoff].copy()
    if train.empty or test.empty or train[TARGET_COLUMN].nunique() < 2 or test[TARGET_COLUMN].nunique() < 2:
        raise ValueError("OOT split needs non-empty train/test sets with both target classes. Adjust --test-start or source period.")
    return train, test, cutoff


def stratified_training_sample(train: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Cap training cost while preserving the rare-event target rate exactly by stratum."""
    if len(train) <= max_rows:
        return train
    sampled = []
    for _, group in train.groupby(TARGET_COLUMN, sort=False):
        count = round(max_rows * len(group) / len(train))
        sampled.append(group.sample(n=count, random_state=42))
    return pd.concat(sampled, ignore_index=True).sort_values("observation_date")


def feature_groups(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Separate numerical and categorical features after excluding IDs, dates and labels."""
    candidates = [column for column in frame.columns if column not in EXCLUDED_COLUMNS]
    numeric = [column for column in candidates if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in candidates if column not in numeric]
    if not numeric:
        raise ValueError("No numerical model features are available.")
    return numeric, categorical


def preprocessing(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Use fitting-only imputation and encoding to avoid OOT leakage."""
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    transformers: list[tuple[str, Pipeline, list[str]]] = [("numeric", numeric_pipe, numeric)]
    if categorical:
        categorical_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=25)),
            ]
        )
        transformers.append(("categorical", categorical_pipe, categorical))
    return ColumnTransformer(transformers)


def ks_statistic(y_true: pd.Series, score: np.ndarray) -> float:
    """Calculate maximum cumulative good/bad separation."""
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(np.max(tpr - fpr))


def evaluate(y_true: pd.Series, score: np.ndarray) -> dict[str, float]:
    """Calculate ranking, imbalance-aware and probability-quality metrics."""
    fpr, tpr, _ = roc_curve(y_true, score)
    return {
        "auc": float(auc(fpr, tpr)),
        "ks": ks_statistic(y_true, score),
        "pr_auc": float(average_precision_score(y_true, score)),
        "brier_score": float(brier_score_loss(y_true, score)),
    }


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Compute score PSI using bins fixed on the training population."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    expected_bins = pd.Series(pd.cut(expected, bins=edges, include_lowest=True))
    actual_bins = pd.Series(pd.cut(actual, bins=edges, include_lowest=True))
    expected_share = expected_bins.value_counts(sort=False) / len(expected)
    actual_share = actual_bins.value_counts(sort=False) / len(actual)
    expected_share = expected_share.clip(lower=0.0001)
    actual_share = actual_share.clip(lower=0.0001)
    return float(((actual_share - expected_share) * np.log(actual_share / expected_share)).sum())


def train_models(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict[str, Pipeline], pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Fit a scorecard-style logistic baseline and an XGBoost challenger."""
    numeric, categorical = feature_groups(train)
    x_train, y_train = train[numeric + categorical], train[TARGET_COLUMN].astype(int)
    x_test, y_test = test[numeric + categorical], test[TARGET_COLUMN].astype(int)

    logistic = Pipeline(
        [
            ("preprocessor", preprocessing(numeric, categorical)),
            ("model", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )
    xgboost = Pipeline(
        [
            ("preprocessor", preprocessing(numeric, categorical)),
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
                    importance_type="gain",
                ),
            ),
        ]
    )
    logistic.fit(x_train, y_train)
    xgboost.fit(x_train, y_train)

    score_columns = ["loan_id", "origination_date", "observation_date", TARGET_COLUMN]
    if "original_upb" in test.columns:
        score_columns.append("original_upb")
    scored = test[score_columns].copy()
    metrics: list[dict[str, float | str]] = []
    for name, model in {"Logistic regression / 逻辑回归": logistic, "XGBoost / XGBoost": xgboost}.items():
        score = model.predict_proba(x_test)[:, 1]
        key = "logistic_pd" if name.startswith("Logistic") else "xgboost_pd"
        scored[key] = score
        metrics.append({"model": name, **evaluate(y_test, score)})
    return {"logistic": logistic, "xgboost": xgboost}, pd.DataFrame(metrics), scored, numeric, categorical


def cohort_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    """Show XGBoost performance by later observation cohort when sample size permits."""
    scored = scored.copy()
    scored["cohort"] = pd.to_datetime(scored["observation_date"]).dt.to_period("Q").astype(str)
    rows: list[dict[str, object]] = []
    for cohort, group in scored.groupby("cohort", sort=True):
        if group[TARGET_COLUMN].nunique() < 2:
            continue
        rows.append({"cohort": cohort, "sample_count": len(group), "bad_rate": group[TARGET_COLUMN].mean(), **evaluate(group[TARGET_COLUMN], group["xgboost_pd"])})
    return pd.DataFrame(rows)


def xgboost_importance(model: Pipeline) -> pd.DataFrame:
    """Extract transformed-feature gain importance for the challenger model."""
    names = model.named_steps["preprocessor"].get_feature_names_out()
    gain = model.named_steps["model"].feature_importances_
    output = pd.DataFrame({"feature": names, "gain": gain})
    output["feature"] = output["feature"].str.replace("numeric__", "", regex=False).str.replace("categorical__", "", regex=False)
    return output.sort_values("gain", ascending=False, ignore_index=True)


def save_figures(scored: pd.DataFrame, importance: pd.DataFrame) -> None:
    """Save bilingual OOT performance, calibration and importance charts."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    configure_plot_font()
    y_test = scored[TARGET_COLUMN]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for column, label, color in [
        ("logistic_pd", "Logistic regression / 逻辑回归", "#2a6f97"),
        ("xgboost_pd", "XGBoost / XGBoost", "#bc6c25"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, scored[column])
        axes[0].plot(fpr, tpr, label=f"{label} (AUC={auc(fpr, tpr):.3f})", color=color, linewidth=2)
        calibration = pd.DataFrame({"score": scored[column], "bad": y_test}).groupby(pd.qcut(scored[column].rank(method="first"), 10)).agg({"score": "mean", "bad": "mean"})
        axes[1].plot(calibration["score"], calibration["bad"], marker="o", label=label, color=color)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#6c757d")
    axes[0].set_title("OOT ROC curve / 时间外 ROC 曲线")
    axes[0].set_xlabel("False positive rate / 假阳性率")
    axes[0].set_ylabel("True positive rate / 真阳性率")
    axes[0].legend(fontsize=8)
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="#6c757d")
    axes[1].set_title("Calibration by score decile / 分数十分位校准")
    axes[1].set_xlabel("Average predicted PD / 平均预测 PD")
    axes[1].set_ylabel("Observed bad rate / 观察坏样本率")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "oot_roc_and_calibration.png", dpi=160)
    plt.close(fig)

    top = importance.head(15).sort_values("gain")
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.barh(top["feature"], top["gain"], color="#bc6c25")
    axis.set_title("XGBoost feature importance by gain / XGBoost 增益特征重要性")
    axis.set_xlabel("Gain / 增益")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "xgboost_feature_importance.png", dpi=160)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str], percent_columns: set[str]) -> str:
    """Render report tables with consistent financial-model formatting."""
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame[columns].itertuples(index=False, name=None):
        values = []
        for column, value in zip(columns, row):
            if column in percent_columns:
                values.append(f"{value:.2%}")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{value:.3f}")
            elif isinstance(value, (int, np.integer)):
                values.append(f"{value:,}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cutoff: pd.Timestamp,
    metrics: pd.DataFrame,
    cohort: pd.DataFrame,
    importance: pd.DataFrame,
    score_psi: float,
) -> None:
    """Create an auditable bilingual main-model report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    display_metrics = metrics.rename(
        columns={"model": "Model / 模型", "auc": "AUC", "ks": "KS", "pr_auc": "PR-AUC", "brier_score": "Brier score / Brier 分数"}
    )
    display_importance = importance.head(10).rename(columns={"feature": "Feature / 特征", "gain": "Gain / 增益"})
    report = f"""# Out-of-Time PD Model Report / 时间外 PD 模型报告

## Split design / 切分设计

Training uses observation dates before `{cutoff:%Y-%m-%d}`; testing uses that date and later. This is a chronological out-of-time split, not a random split. 训练集使用 `{cutoff:%Y-%m-%d}` 之前的观察日期，测试集使用该日及之后的观察日期。这是按时间的 OOT 切分，不是随机切分。

| Population / 样本 | Loan count / 贷款数 | Bad rate / 坏样本率 |
| --- | ---: | ---: |
| Train / 训练集 | {len(train):,} | {train[TARGET_COLUMN].mean():.2%} |
| OOT test / 时间外测试集 | {len(test):,} | {test[TARGET_COLUMN].mean():.2%} |

## Champion-challenger performance / 冠军挑战者表现

| Model / 模型 | AUC | KS | PR-AUC | Brier score / Brier 分数 |
| --- | ---: | ---: | ---: | ---: |
{markdown_table(display_metrics, ['Model / 模型', 'AUC', 'KS', 'PR-AUC', 'Brier score / Brier 分数'], set()).split(chr(10), 2)[2]}

**Interpretation / 解读：** Logistic regression remains the transparent baseline. XGBoost is the challenger intended to capture nonlinear interactions. The selected production candidate must be chosen using both performance and stability, not AUC alone. 逻辑回归仍是透明的基准模型；XGBoost 是用于捕捉非线性交互的挑战模型。最终候选模型必须同时依据性能与稳定性选择，而非仅看 AUC。

![OOT ROC and calibration](figures/oot_roc_and_calibration.png)

## Score stability / 分数稳定性

XGBoost score PSI between train and OOT test is **{score_psi:.3f}**. As a working convention, PSI below 0.10 is low drift, 0.10-0.25 merits review, and above 0.25 is material drift. XGBoost 在训练集与时间外测试集之间的分数 PSI 为 **{score_psi:.3f}**。作为工作约定，PSI 小于 0.10 表示低漂移，0.10-0.25 需要复核，高于 0.25 表示明显漂移。

## OOT cohort performance / 时间外批次表现

{markdown_table(cohort.rename(columns={'cohort': 'Cohort / 批次', 'sample_count': 'Sample count / 样本量', 'bad_rate': 'Bad rate / 坏样本率', 'auc': 'AUC', 'ks': 'KS', 'pr_auc': 'PR-AUC', 'brier_score': 'Brier score / Brier 分数'}), ['Cohort / 批次', 'Sample count / 样本量', 'Bad rate / 坏样本率', 'AUC', 'KS'], {'Bad rate / 坏样本率'}) if not cohort.empty else 'No eligible OOT cohort has both target classes / 没有同时包含好坏样本的可评估时间外批次。'}

## XGBoost feature importance / XGBoost 特征重要性

{markdown_table(display_importance, ['Feature / 特征', 'Gain / 增益'], set())}

![XGBoost feature importance](figures/xgboost_feature_importance.png)

## Next decision layer / 下一步决策层

The model now ranks risk out of time. The next layer should define policy bands for pass, manual review and reject, then compare approval rate, observed bad rate, expected loss and capacity. 模型现在能够进行时间外风险排序。下一层应设定通过、人工审核和拒绝区间，并比较通过率、观察坏样本率、预期损失和审核产能。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    """Run OOT champion-challenger modelling on prepared Fannie PD snapshots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_PATH)
    parser.add_argument("--test-start", help="First observation date in OOT test, e.g. 2020-01-01.")
    parser.add_argument("--max-train-rows", type=int, default=750_000, help="Stratified maximum model-training rows.")
    args = parser.parse_args()
    if not args.snapshot.exists():
        raise FileNotFoundError(
            f"Prepared snapshot not found: {args.snapshot}. Run src/fannie_mae_pipeline.py after downloading source files."
        )

    snapshot = pd.read_csv(args.snapshot, low_memory=False)
    train, test, cutoff = split_out_of_time(snapshot, args.test_start)
    fit_train = stratified_training_sample(train, args.max_train_rows)
    models, metrics, scored, _, _ = train_models(fit_train, test)
    importance = xgboost_importance(models["xgboost"])
    cohort = cohort_metrics(scored)
    x_train = fit_train[[column for column in fit_train.columns if column not in EXCLUDED_COLUMNS]]
    x_test = test[[column for column in test.columns if column not in EXCLUDED_COLUMNS]]
    train_score = models["xgboost"].predict_proba(x_train)[:, 1]
    score_psi = population_stability_index(train_score, scored["xgboost_pd"].to_numpy())

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(PROCESSED_DIR / "oot_model_performance.csv", index=False)
    scored.to_csv(PROCESSED_DIR / "oot_test_scores.csv", index=False)
    cohort.to_csv(PROCESSED_DIR / "oot_cohort_performance.csv", index=False)
    importance.to_csv(PROCESSED_DIR / "xgboost_feature_importance.csv", index=False)
    pd.DataFrame([{"metric": "xgboost_score_psi", "value": score_psi}]).to_csv(
        PROCESSED_DIR / "oot_score_stability.csv", index=False
    )
    save_figures(scored, importance)
    write_report(fit_train, test, cutoff, metrics, cohort, importance, score_psi)
    print(f"Trained OOT models: {len(fit_train):,} stratified train loans, {len(test):,} test loans.")
    print(f"OOT model report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()

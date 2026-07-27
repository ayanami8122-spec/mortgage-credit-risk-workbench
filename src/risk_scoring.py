"""Build an interpretable credit-risk scoring baseline from the profiled UCI data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_quality import (
    BILL_COLUMNS,
    PAYMENT_COLUMNS,
    PAYMENT_STATUS_COLUMNS,
    PROJECT_ROOT,
    TARGET_COLUMN,
    load_data,
)


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROJECT_ROOT / "reports" / "risk_scoring_report.md"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
CHINESE_FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")
RANDOM_STATE = 42
TEST_SIZE = 0.20

FEATURE_LABELS = {
    "LIMIT_BAL": "Credit limit / 授信额度",
    "recent_delinquency_status": "Recent repayment status / 最近一期还款状态",
    "max_delinquency_status": "Worst repayment status / 历史最大逾期状态",
    "delinquent_month_count": "Delinquent month count / 历史逾期月数",
    "avg_bill_amount": "Average bill amount / 平均账单金额",
    "avg_payment_amount": "Average payment amount / 平均还款金额",
    "balance_to_limit_ratio": "Average balance-to-limit ratio / 平均额度使用率",
    "payment_to_bill_ratio": "Payment-to-bill ratio / 还款账单比",
    "bill_amount_volatility": "Bill amount volatility / 账单金额波动",
}


def configure_plot_font() -> None:
    """Set a local CJK font after Matplotlib style configuration."""
    if CHINESE_FONT_PATH.exists():
        font_manager.fontManager.addfont(str(CHINESE_FONT_PATH))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=CHINESE_FONT_PATH).get_name()
    else:
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def prepare_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build interpretable behavioural and financial features available before outcome."""
    repayment = frame[PAYMENT_STATUS_COLUMNS]
    bills = frame[BILL_COLUMNS]
    payments = frame[PAYMENT_COLUMNS]

    features = pd.DataFrame(index=frame.index)
    features["LIMIT_BAL"] = frame["LIMIT_BAL"]
    features["recent_delinquency_status"] = frame["PAY_0"]
    features["max_delinquency_status"] = repayment.max(axis=1)
    features["delinquent_month_count"] = (repayment >= 1).sum(axis=1)
    features["avg_bill_amount"] = bills.mean(axis=1)
    features["avg_payment_amount"] = payments.mean(axis=1)
    features["balance_to_limit_ratio"] = (features["avg_bill_amount"] / frame["LIMIT_BAL"]).clip(-2, 10)

    positive_bills = bills.clip(lower=0).sum(axis=1)
    features["payment_to_bill_ratio"] = (payments.sum(axis=1) / positive_bills.replace(0, np.nan)).fillna(0).clip(0, 5)
    features["bill_amount_volatility"] = bills.std(axis=1)

    if features.isna().any().any():
        raise ValueError("Prepared model features contain missing values.")
    return features, frame[TARGET_COLUMN].astype(int)


def ks_statistic(y_true: pd.Series | np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Return maximum KS and its corresponding predicted-probability threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    index = int(np.argmax(tpr - fpr))
    return float(tpr[index] - fpr[index]), float(thresholds[index])


def metric_row(model_name: str, y_true: pd.Series, scores: np.ndarray) -> dict[str, float | str]:
    """Calculate AUC and KS for a model's holdout predictions."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    ks, threshold = ks_statistic(y_true, scores)
    return {"model": model_name, "auc": auc(fpr, tpr), "ks": ks, "ks_threshold": threshold}


def model_and_score(
    features: pd.DataFrame, target: pd.Series,
) -> tuple[dict[str, Pipeline | RandomForestClassifier], pd.DataFrame, pd.Series, pd.DataFrame]:
    """Use the same stratified holdout to compare logistic regression and a tree model."""
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, stratify=target, random_state=RANDOM_STATE
    )
    logistic = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE)),
        ]
    )
    forest = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=30,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    logistic.fit(x_train, y_train)
    forest.fit(x_train, y_train)

    logistic_scores = logistic.predict_proba(x_test)[:, 1]
    forest_scores = forest.predict_proba(x_test)[:, 1]
    metrics = pd.DataFrame(
        [
            metric_row("Logistic regression / 逻辑回归", y_test, logistic_scores),
            metric_row("Random forest / 随机森林", y_test, forest_scores),
        ]
    )
    scored_test = x_test.copy()
    scored_test["actual_default"] = y_test
    scored_test["logistic_default_probability"] = logistic_scores
    scored_test["forest_default_probability"] = forest_scores
    return {"logistic": logistic, "forest": forest}, metrics, y_test, scored_test


def risk_deciles(scored_test: pd.DataFrame) -> pd.DataFrame:
    """Show observed default concentration from lowest to highest logistic-model risk."""
    scored = scored_test.copy()
    scored["risk_decile"] = pd.qcut(
        scored["logistic_default_probability"].rank(method="first"), q=10, labels=False
    ) + 1
    summary = (
        scored.groupby("risk_decile", observed=True)
        .agg(
            sample_count=("actual_default", "size"),
            observed_default_rate=("actual_default", "mean"),
            average_predicted_probability=("logistic_default_probability", "mean"),
        )
        .reset_index()
    )
    summary["risk_decile_label"] = [
        f"D{decile} / {'Lowest' if decile == 1 else 'Highest' if decile == 10 else 'Intermediate'} risk / "
        f"{'最低风险' if decile == 1 else '最高风险' if decile == 10 else '中间风险'}"
        for decile in summary["risk_decile"]
    ]
    return summary


def feature_coefficients(logistic: Pipeline) -> pd.DataFrame:
    """Translate standardized logistic coefficients into an auditable explanation table."""
    coefficients = pd.DataFrame(
        {
            "feature": list(FEATURE_LABELS),
            "coefficient": logistic.named_steps["model"].coef_[0],
        }
    )
    coefficients["feature_label"] = coefficients["feature"].map(FEATURE_LABELS)
    coefficients["odds_ratio_per_standard_deviation"] = np.exp(coefficients["coefficient"])
    coefficients["risk_direction"] = np.where(
        coefficients["coefficient"] >= 0,
        "Higher risk / 风险上升",
        "Lower risk / 风险下降",
    )
    return coefficients.sort_values("coefficient", ascending=False, ignore_index=True)


def save_figures(
    y_test: pd.Series,
    scored_test: pd.DataFrame,
    coefficients: pd.DataFrame,
    deciles: pd.DataFrame,
) -> None:
    """Create bilingual model-performance and business-interpretation charts."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    configure_plot_font()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for column, label, color in [
        ("logistic_default_probability", "Logistic regression / 逻辑回归", "#2a6f97"),
        ("forest_default_probability", "Random forest / 随机森林", "#bc6c25"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, scored_test[column])
        axes[0].plot(fpr, tpr, label=f"{label} (AUC={auc(fpr, tpr):.3f})", color=color, linewidth=2)
        axes[1].plot(np.linspace(0, 1, len(tpr)), tpr - fpr, label=label, color=color, linewidth=2)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#6c757d")
    axes[0].set_title("ROC curve / ROC 曲线")
    axes[0].set_xlabel("False positive rate / 假阳性率")
    axes[0].set_ylabel("True positive rate / 真阳性率")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0, linestyle="--", color="#6c757d")
    axes[1].set_title("KS curve / KS 曲线")
    axes[1].set_xlabel("Score population percentile / 分数人群百分位")
    axes[1].set_ylabel("TPR - FPR / 累积好坏客户差")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_roc_and_ks.png", dpi=160)
    plt.close(fig)

    top_features = coefficients.head(8).sort_values("coefficient")
    fig, axis = plt.subplots(figsize=(9, 5))
    colors = np.where(top_features["coefficient"] >= 0, "#b23a48", "#457b9d")
    axis.barh(top_features["feature_label"], top_features["coefficient"], color=colors)
    axis.axvline(0, color="#495057", linewidth=0.8)
    axis.set_title("Logistic-regression feature effects / 逻辑回归特征影响")
    axis.set_xlabel("Standardized coefficient / 标准化系数")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "logistic_feature_effects.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(deciles["risk_decile"], deciles["observed_default_rate"], color="#b23a48")
    axis.plot(
        deciles["risk_decile"], deciles["average_predicted_probability"], color="#2a6f97",
        marker="o", linewidth=2, label="Predicted probability / 预测违约概率",
    )
    axis.set_xticks(deciles["risk_decile"])
    axis.set_title("Observed risk by logistic-score decile / 逻辑回归风险十分位表现")
    axis.set_xlabel("D1 lowest risk to D10 highest risk / D1 最低风险至 D10 最高风险")
    axis.set_ylabel("Default rate or probability / 违约率或概率")
    axis.yaxis.set_major_formatter("{x:.0%}")
    axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "risk_decile_performance.png", dpi=160)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str], percent_columns: set[str]) -> str:
    """Render selected model outputs as a Markdown table with concise formatting."""
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
    metrics: pd.DataFrame,
    scored_test: pd.DataFrame,
    coefficients: pd.DataFrame,
    deciles: pd.DataFrame,
) -> None:
    """Produce a bilingual report that separates performance, explanation and limits."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    logistic_metrics = metrics.loc[metrics["model"].str.startswith("Logistic")].iloc[0]
    prediction = (scored_test["logistic_default_probability"] >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(scored_test["actual_default"], prediction).ravel()
    confusion = pd.DataFrame(
        {
            "Metric / 指标": ["True negatives / 正确识别未违约", "False positives / 误判为违约", "False negatives / 漏判违约", "True positives / 正确识别违约"],
            "Count / 数量": [tn, fp, fn, tp],
        }
    )
    display_metrics = metrics.rename(
        columns={"model": "Model / 模型", "auc": "AUC", "ks": "KS", "ks_threshold": "KS threshold / KS 阈值"}
    )
    display_metrics["AUC"] = display_metrics["AUC"].map(lambda value: f"{value:.3f}")
    display_metrics["KS"] = display_metrics["KS"].map(lambda value: f"{value:.2%}")
    display_metrics["KS threshold / KS 阈值"] = display_metrics["KS threshold / KS 阈值"].map(
        lambda value: f"{value:.2%}"
    )
    display_coefficients = coefficients.head(6).rename(
        columns={
            "feature_label": "Feature / 特征",
            "coefficient": "Standardized coefficient / 标准化系数",
            "odds_ratio_per_standard_deviation": "Odds ratio / 优势比",
            "risk_direction": "Direction / 方向",
        }
    )
    display_deciles = deciles.rename(
        columns={
            "risk_decile_label": "Risk decile / 风险十分位",
            "sample_count": "Sample count / 样本量",
            "observed_default_rate": "Observed default rate / 观察违约率",
            "average_predicted_probability": "Average predicted probability / 平均预测概率",
        }
    )
    report = f"""# Risk Scoring Baseline Report / 风险评分基准模型报告

## Purpose and boundary / 目的与边界

This stage converts the findings from data diagnosis into an interpretable probability-of-default baseline. It uses a stratified random 80/20 train-test split, with {len(scored_test):,} customers in the holdout set. 这一阶段将数据诊断结论转化为可解释的违约概率基准模型，采用分层随机 80/20 训练测试切分；测试集包含 {len(scored_test):,} 名客户。

The score is for model evaluation only. It is not an approval policy, a production scorecard, or a time-based validation result. 该分数仅用于模型评估，不等同于审批策略、生产评分卡或基于时间的验证结果。

## Model features / 建模特征

The baseline deliberately uses repayment and account-behaviour signals plus credit limit. It excludes `ID`, target data, and demographic fields such as sex, education and marriage status from the model; those fields remain in the descriptive analysis. 基准模型有意使用还款、账户行为和授信额度信号；排除了 `ID`、目标变量以及性别、教育、婚姻等人口属性，这些属性仅保留在描述性分析中。

## Holdout performance / 测试集表现

{markdown_table(display_metrics, ['Model / 模型', 'AUC', 'KS', 'KS threshold / KS 阈值'], set())}

**Interpretation / 解读：** The logistic-regression baseline achieves AUC {logistic_metrics['auc']:.3f} and KS {logistic_metrics['ks']:.3f} on unseen customers. AUC measures ranking ability; KS measures the largest cumulative separation between default and non-default customers. 逻辑回归在未参与训练的客户上取得 AUC {logistic_metrics['auc']:.3f}、KS {logistic_metrics['ks']:.3f}；AUC 衡量排序能力，KS 衡量违约与未违约客群的最大累积区分度。

![ROC and KS curves](figures/model_roc_and_ks.png)

## What a 0.5 cutoff means / 0.5 阈值的含义

The following is a technical reference only. A 0.5 cutoff is not recommended as an approval rule because approval decisions must balance pass rate, default rate and expected loss. 下表仅作技术参照。0.5 不是建议的准入阈值，因为审批决策需要同时权衡通过率、违约率和预期损失。

{markdown_table(confusion, ['Metric / 指标', 'Count / 数量'], set())}

## Main risk drivers / 主要风险驱动因素

Coefficients are standardized: for numeric variables, they reflect a one-standard-deviation increase. A positive coefficient means that a higher value is associated with a higher predicted default risk while holding other features constant. 系数经过标准化：数值变量的系数反映增加一个标准差的影响。正系数表示在其他变量不变时，该变量升高与更高的预测违约风险相关。

{markdown_table(display_coefficients, ['Feature / 特征', 'Standardized coefficient / 标准化系数', 'Odds ratio / 优势比', 'Direction / 方向'], set())}

![Logistic feature effects](figures/logistic_feature_effects.png)

## Score-decile validation / 风险十分位验证

D1 is the lowest predicted-risk group and D10 is the highest. A useful score should show increasing observed default rates as risk decile rises. D1 为预测风险最低组，D10 为最高组；一个有用的评分应随风险分位提升而呈现更高的观察违约率。

{markdown_table(display_deciles, ['Risk decile / 风险十分位', 'Sample count / 样本量', 'Observed default rate / 观察违约率', 'Average predicted probability / 平均预测概率'], {'Observed default rate / 观察违约率', 'Average predicted probability / 平均预测概率'})}

![Risk decile performance](figures/risk_decile_performance.png)

## Findings and next stage / 结论与下一阶段

1. **Behavioural signals dominate / 行为信号主导：** Recent and persistent delinquency are expected to be the strongest risk indicators, consistent with the data-diagnosis stage. 最近逾期与逾期持续性预计是最强风险指标，与数据诊断结论一致。
2. **Tree model is a comparison, not the chosen scorecard / 树模型用于对照：** The random forest checks whether a nonlinear benchmark materially improves ranking; the logistic model remains the primary explanation model. 随机森林用于检验非线性模型是否显著提升排序能力；逻辑回归仍是主要解释模型。
3. **Next work is strategy simulation / 下一步是策略模拟：** Use the logistic predicted probability to define pass, manual-review and reject bands, then compare pass rate, observed default rate and expected-risk exposure. 使用逻辑回归预测概率设定通过、人工审核和拒绝区间，并比较通过率、观察违约率和风险暴露。
4. **Required production checks / 生产前必要检查：** A real scorecard requires out-of-time validation, calibration, feature stability, fairness/compliance review, and policy approval. 真实评分卡还需要跨时间验证、校准、特征稳定性、公平合规审查与策略审批。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    """Run feature preparation, model comparison, interpretation and reporting."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_data()
    features, target = prepare_features(frame)
    models, metrics, y_test, scored_test = model_and_score(features, target)
    coefficients = feature_coefficients(models["logistic"])
    deciles = risk_deciles(scored_test)

    metrics.to_csv(PROCESSED_DIR / "model_performance.csv", index=False)
    coefficients.to_csv(PROCESSED_DIR / "logistic_feature_coefficients.csv", index=False)
    deciles.to_csv(PROCESSED_DIR / "risk_decile_performance.csv", index=False)
    scored_test.to_csv(PROCESSED_DIR / "test_set_risk_scores.csv", index=False)
    save_figures(y_test, scored_test, coefficients, deciles)
    write_report(metrics, scored_test, coefficients, deciles)

    print(f"Prepared {features.shape[1]} model features from {len(features):,} customers.")
    print(f"Risk scoring report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()

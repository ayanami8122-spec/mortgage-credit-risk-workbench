"""Monitor feature-level PSI and data-quality rules across risk-model cohorts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SNAPSHOT_PATH = PROCESSED_DIR / "fannie_pd_snapshot.csv"
SCORES_PATH = PROCESSED_DIR / "oot_test_scores.csv"
FEATURE_OUTPUT = PROCESSED_DIR / "feature_stability_monitoring.csv"
QUALITY_OUTPUT = PROCESSED_DIR / "data_quality_monitoring.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "portfolio_health_report.md"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
CHINESE_FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")

CONTINUOUS_FEATURES = [
    "borrower_credit_score",
    "original_ltv",
    "original_dti",
    "original_upb",
    "hist_upb_to_original_upb",
]
CATEGORICAL_FEATURES = [
    "hist_max_dpd",
    "hist_30dpd_months",
    "loan_purpose",
    "property_type",
    "occupancy_status",
    "property_state",
]
QUALITY_RULES = {
    "borrower_credit_score": (300, 850),
    "original_ltv": (0, 100),
    "original_dti": (0, 65),
    "original_upb": (0, np.inf),
    "hist_upb_to_original_upb": (0, 1.20),
}


def configure_plot_font() -> None:
    """Configure a CJK-capable chart font for bilingual monitoring output."""
    if CHINESE_FONT_PATH.exists():
        font_manager.fontManager.addfont(str(CHINESE_FONT_PATH))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=CHINESE_FONT_PATH).get_name()
    else:
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def psi_status(value: float) -> str:
    """Classify PSI with documented project monitoring thresholds."""
    if value >= 0.25:
        return "Alert / 预警"
    if value >= 0.10:
        return "Review / 复核"
    return "Stable / 稳定"


def distribution_psi(expected: pd.Series, actual: pd.Series, feature_type: str, bins: int = 10) -> float:
    """Calculate PSI for a continuous or categorical feature with fixed baseline bins."""
    expected = expected.copy()
    actual = actual.copy()
    if feature_type == "continuous":
        clean_expected = pd.to_numeric(expected, errors="coerce").dropna()
        if clean_expected.nunique() < 2:
            return 0.0
        edges = np.unique(np.quantile(clean_expected, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        expected_bins = pd.cut(pd.to_numeric(expected, errors="coerce"), bins=edges, include_lowest=True).astype("string").fillna("Missing / 缺失")
        actual_bins = pd.cut(pd.to_numeric(actual, errors="coerce"), bins=edges, include_lowest=True).astype("string").fillna("Missing / 缺失")
    else:
        expected_bins = expected.astype("string").fillna("Missing / 缺失")
        actual_bins = actual.astype("string").fillna("Missing / 缺失")
        allowed = set(expected_bins.unique())
        actual_bins = actual_bins.where(actual_bins.isin(allowed), "Other / 其他")
        if "Other / 其他" not in allowed:
            expected_bins = expected_bins.where(expected_bins.isin(allowed), "Other / 其他")

    categories = expected_bins.value_counts().index.union(actual_bins.value_counts().index)
    expected_share = expected_bins.value_counts().reindex(categories, fill_value=0) / len(expected_bins)
    actual_share = actual_bins.value_counts().reindex(categories, fill_value=0) / len(actual_bins)
    expected_share = expected_share.clip(lower=0.0001)
    actual_share = actual_share.clip(lower=0.0001)
    return float(((actual_share - expected_share) * np.log(actual_share / expected_share)).sum())


def build_feature_stability(snapshot: pd.DataFrame, oot_scores: pd.DataFrame) -> pd.DataFrame:
    """Compare train-to-OOT and first-to-later-OOT feature distributions."""
    data = snapshot.copy()
    data["observation_date"] = pd.to_datetime(data["observation_date"])
    oot_scores = oot_scores[["loan_id", "observation_date"]].copy()
    oot_scores["observation_date"] = pd.to_datetime(oot_scores["observation_date"])
    oot_loan_ids = set(oot_scores["loan_id"])
    oot = data.loc[data["loan_id"].isin(oot_loan_ids)].copy()
    train = data.loc[~data["loan_id"].isin(oot_loan_ids)].copy()
    oot["cohort"] = oot["observation_date"].dt.to_period("Q").astype(str)
    cohorts = sorted(oot["cohort"].unique())
    if len(cohorts) < 2:
        raise ValueError("Feature monitoring requires at least two OOT cohorts.")
    baseline = oot.loc[oot["cohort"] == cohorts[0]]
    comparisons = [
        ("Train vs OOT / 训练集对 OOT", train, oot),
        (f"{cohorts[0]} vs {cohorts[-1]} / OOT 批次对比", baseline, oot.loc[oot["cohort"] == cohorts[-1]]),
    ]
    rows: list[dict[str, object]] = []
    for comparison, expected, actual in comparisons:
        for feature in CONTINUOUS_FEATURES:
            rows.append(
                {
                    "comparison": comparison,
                    "feature": feature,
                    "feature_type": "Continuous / 连续变量",
                    "psi": distribution_psi(expected[feature], actual[feature], "continuous"),
                    "baseline_missing_rate": expected[feature].isna().mean(),
                    "comparison_missing_rate": actual[feature].isna().mean(),
                }
            )
        for feature in CATEGORICAL_FEATURES:
            rows.append(
                {
                    "comparison": comparison,
                    "feature": feature,
                    "feature_type": "Categorical / 分类变量",
                    "psi": distribution_psi(expected[feature], actual[feature], "categorical"),
                    "baseline_missing_rate": expected[feature].isna().mean(),
                    "comparison_missing_rate": actual[feature].isna().mean(),
                }
            )
    output = pd.DataFrame(rows)
    output["psi_status"] = output["psi"].map(psi_status)
    output["missing_rate_change"] = output["comparison_missing_rate"] - output["baseline_missing_rate"]
    return output.sort_values(["comparison", "psi"], ascending=[True, False], ignore_index=True)


def build_data_quality(snapshot: pd.DataFrame, oot_scores: pd.DataFrame) -> pd.DataFrame:
    """Check missing and domain-invalid rates in train and OOT populations."""
    data = snapshot.copy()
    oot_loan_ids = set(oot_scores["loan_id"])
    populations = {
        "Train / 训练集": data.loc[~data["loan_id"].isin(oot_loan_ids)],
        "OOT / 时间外测试集": data.loc[data["loan_id"].isin(oot_loan_ids)],
    }
    rows: list[dict[str, object]] = []
    for population, frame in populations.items():
        for feature, (minimum, maximum) in QUALITY_RULES.items():
            values = pd.to_numeric(frame[feature], errors="coerce")
            invalid = values.notna() & ((values < minimum) | (values > maximum))
            rows.append(
                {
                    "population": population,
                    "feature": feature,
                    "sample_count": len(frame),
                    "missing_rate": values.isna().mean(),
                    "invalid_rate": invalid.mean(),
                    "quality_status": "Alert / 预警" if invalid.mean() > 0.01 or values.isna().mean() > 0.02 else "Stable / 稳定",
                }
            )
    return pd.DataFrame(rows)


def save_figures(feature_stability: pd.DataFrame, quality: pd.DataFrame) -> None:
    """Save bilingual feature-PSI and data-quality visuals."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    configure_plot_font()
    train_oot = feature_stability.loc[feature_stability["comparison"] == "Train vs OOT / 训练集对 OOT"].head(10).sort_values("psi")
    fig, axis = plt.subplots(figsize=(10, 6))
    colors = ["#b23a48" if value >= 0.25 else "#f4a261" if value >= 0.10 else "#2a6f97" for value in train_oot["psi"]]
    axis.barh(train_oot["feature"], train_oot["psi"], color=colors)
    axis.axvline(0.10, linestyle="--", color="#f4a261", label="Review 0.10 / 复核线")
    axis.axvline(0.25, linestyle="--", color="#b23a48", label="Alert 0.25 / 预警线")
    axis.set_title("Top feature PSI: train vs OOT / 特征 PSI：训练集对 OOT")
    axis.set_xlabel("PSI")
    axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_psi_monitoring.png", dpi=160)
    plt.close(fig)

    pivot = quality.pivot(index="feature", columns="population", values=["missing_rate", "invalid_rate"]).fillna(0)
    fig, axis = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(pivot))
    width = 0.36
    train_issue = pivot[("missing_rate", "Train / 训练集")] + pivot[("invalid_rate", "Train / 训练集")]
    oot_issue = pivot[("missing_rate", "OOT / 时间外测试集")] + pivot[("invalid_rate", "OOT / 时间外测试集")]
    axis.bar(positions - width / 2, train_issue, width=width, label="Train issues / 训练集问题率", color="#6c757d")
    axis.bar(positions + width / 2, oot_issue, width=width, label="OOT issues / OOT 问题率", color="#3a86ff")
    axis.set_xticks(positions, pivot.index, rotation=25, ha="right")
    axis.yaxis.set_major_formatter("{x:.2%}")
    axis.set_title("Data-quality rule results / 数据质量规则结果")
    axis.set_ylabel("Missing + invalid rate / 缺失与越界率")
    axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "data_quality_monitoring.png", dpi=160)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render compact high-PSI rows for the bilingual report."""
    columns = ["Feature / 特征", "PSI", "Status / 状态", "Missing-rate change / 缺失率变化"]
    lines = ["| " + " | ".join(columns) + " |", "| --- | ---: | --- | ---: |"]
    for row in frame.itertuples(index=False):
        lines.append(f"| {row.feature} | {row.psi:.3f} | {row.psi_status} | {row.missing_rate_change:.2%} |")
    return "\n".join(lines)


def write_report(feature_stability: pd.DataFrame, quality: pd.DataFrame) -> None:
    """Write an auditable bilingual portfolio-health report."""
    top_train_oot = feature_stability.loc[feature_stability["comparison"] == "Train vs OOT / 训练集对 OOT"].head(10)
    alerts = quality.loc[quality["quality_status"] == "Alert / 预警"]
    quality_text = "No configured data-quality rule breached its alert threshold. 所配置的数据质量规则均未触发预警阈值。" if alerts.empty else alerts.to_csv(index=False)
    report = f"""# Portfolio Health Monitoring Report / 组合健康度监控报告

## Purpose / 目的

Score PSI alone says that the model output distribution moved. This report identifies which input features moved and separately checks missing and domain-invalid values. 分数 PSI 只能说明模型输出分布发生变化；本报告进一步识别哪些输入特征发生变化，并单独检查缺失与越界值。

## Highest feature PSI: train vs OOT / 训练集对 OOT 的高 PSI 特征

{markdown_table(top_train_oot)}

![Feature PSI monitoring](figures/feature_psi_monitoring.png)

## Data-quality rules / 数据质量规则

Configured domains: FICO 300-850, original LTV 0-100, original DTI 0-65, original UPB > 0, and early UPB/original UPB ratio 0-1.20. 配置的字段范围：FICO 300-850、原始 LTV 0-100、原始 DTI 0-65、原始余额大于 0、早期余额/原始余额比值 0-1.20。

{quality_text}

![Data-quality monitoring](figures/data_quality_monitoring.png)

## Reading the result / 如何解读

- PSI >= 0.25: investigate material movement before relying on unchanged score cutoffs. PSI 大于等于 0.25：应在沿用原分数阈值前调查明显变化。
- PSI 0.10-0.25: review portfolio mix, feature definitions and policy changes. PSI 在 0.10-0.25：复核客群、字段口径与策略变化。
- Feature PSI does not prove causal deterioration; it narrows the investigation to fields that changed. 特征 PSI 不证明因果恶化，但能将调查范围缩小到发生变化的字段。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    """Generate feature stability and data quality monitoring outputs."""
    snapshot = pd.read_csv(SNAPSHOT_PATH, low_memory=False)
    scores = pd.read_csv(SCORES_PATH)
    feature_stability = build_feature_stability(snapshot, scores)
    quality = build_data_quality(snapshot, scores)
    feature_stability.to_csv(FEATURE_OUTPUT, index=False)
    quality.to_csv(QUALITY_OUTPUT, index=False)
    save_figures(feature_stability, quality)
    write_report(feature_stability, quality)
    print(f"Monitored {len(feature_stability):,} feature-comparison rows and {len(quality):,} data-quality rows.")
    print(f"Portfolio health report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()

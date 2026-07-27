"""Monitor score drift and bad-rate movement across out-of-time cohorts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from fannie_mae_pipeline import PROJECT_ROOT
from oot_risk_modelling import population_stability_index


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SCORES_PATH = PROCESSED_DIR / "oot_test_scores.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "monitoring_report.md"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TARGET_COLUMN = "bad_12m_90dpd"
CHINESE_FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")


def configure_plot_font() -> None:
    """Set CJK text support after applying the plot style."""
    if CHINESE_FONT_PATH.exists():
        font_manager.fontManager.addfont(str(CHINESE_FONT_PATH))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=CHINESE_FONT_PATH).get_name()
    else:
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def build_monitoring_table(scores: pd.DataFrame) -> tuple[pd.DataFrame, str, float]:
    """Compare every later OOT cohort against the first available OOT baseline cohort."""
    data = scores.copy()
    data["observation_date"] = pd.to_datetime(data["observation_date"])
    data["cohort"] = data["observation_date"].dt.to_period("Q").astype(str)
    cohorts = sorted(data["cohort"].unique())
    if len(cohorts) < 2:
        raise ValueError("Monitoring requires at least two OOT cohorts.")
    baseline_cohort = cohorts[0]
    baseline_scores = data.loc[data["cohort"] == baseline_cohort, "xgboost_pd"].to_numpy()
    rows = []
    for cohort, group in data.groupby("cohort", sort=True):
        psi = population_stability_index(baseline_scores, group["xgboost_pd"].to_numpy())
        rows.append(
            {
                "cohort": cohort,
                "sample_count": len(group),
                "bad_rate": group[TARGET_COLUMN].mean(),
                "average_score": group["xgboost_pd"].mean(),
                "score_psi_vs_baseline": psi,
                "psi_status": "Alert / 预警" if psi >= 0.25 else "Review / 复核" if psi >= 0.10 else "Stable / 稳定",
            }
        )
    table = pd.DataFrame(rows)
    return table, baseline_cohort, float(table["score_psi_vs_baseline"].iloc[-1])


def save_figure(table: pd.DataFrame, baseline: str) -> None:
    """Visualize risk level and score-distribution drift by cohort."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    configure_plot_font()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].plot(table["cohort"], table["bad_rate"], marker="o", color="#b23a48", label="Bad rate / 坏样本率")
    axes[0].plot(table["cohort"], table["average_score"], marker="o", color="#2a6f97", label="Average PD / 平均 PD")
    axes[0].set_title("Risk level by OOT cohort / 时间外批次风险水平")
    axes[0].set_ylabel("Rate / 比率")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].yaxis.set_major_formatter("{x:.0%}")
    axes[0].legend()
    axes[1].bar(table["cohort"], table["score_psi_vs_baseline"], color="#bc6c25")
    axes[1].axhline(0.10, linestyle="--", color="#f4a261", label="Review 0.10 / 复核线")
    axes[1].axhline(0.25, linestyle="--", color="#b23a48", label="Alert 0.25 / 预警线")
    axes[1].set_title(f"Score PSI vs {baseline} / 相对 {baseline} 的分数 PSI")
    axes[1].set_ylabel("PSI")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "oot_monitoring_trends.png", dpi=160)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render monitoring results for the report."""
    columns = ["Cohort / 批次", "Sample count / 样本量", "Bad rate / 坏样本率", "Average PD / 平均 PD", "PSI vs baseline / 相对基准 PSI", "Status / 状态"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.cohort} | {row.sample_count:,} | {row.bad_rate:.2%} | {row.average_score:.2%} | {row.score_psi_vs_baseline:.3f} | {row.psi_status} |"
        )
    return "\n".join(lines)


def write_report(table: pd.DataFrame, baseline: str) -> None:
    """Create a bilingual score-monitoring report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# OOT Score Monitoring Report / 时间外评分监控报告

## Baseline / 基准批次

The first available OOT cohort, **{baseline}**, is the monitoring baseline. Later cohorts are compared against it using XGBoost PD distribution PSI. 第一个可用的时间外批次 **{baseline}** 作为监控基准，后续批次通过 XGBoost PD 分布 PSI 与其比较。

## Cohort monitoring / 批次监控

{markdown_table(table)}

![OOT monitoring trends](figures/oot_monitoring_trends.png)

## Alert interpretation / 预警解读

- **Stable / 稳定:** PSI < 0.10. Score distribution is close to baseline. PSI 小于 0.10，分数分布接近基准。
- **Review / 复核:** 0.10 <= PSI < 0.25. Check portfolio mix, feature distribution, policy changes and data quality. PSI 在 0.10 至 0.25 之间，应核查客群结构、特征分布、策略变更和数据质量。
- **Alert / 预警:** PSI >= 0.25. Material distribution shift; investigate before relying on the current score performance. PSI 大于等于 0.25，存在明显分布漂移，应在继续依赖模型表现前调查原因。

PSI is a distribution-stability signal, not evidence that model discrimination has failed. It must be read with bad rate, AUC/KS and data-quality checks. PSI 是分布稳定性信号，不等同于模型区分能力失效；必须结合坏样本率、AUC/KS 与数据质量核查共同解读。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    """Generate cohort-level OOT monitoring outputs."""
    if not SCORES_PATH.exists():
        raise FileNotFoundError("OOT score file not found. Run src/oot_risk_modelling.py first.")
    table, baseline, _ = build_monitoring_table(pd.read_csv(SCORES_PATH))
    table.to_csv(PROCESSED_DIR / "oot_monitoring_summary.csv", index=False)
    save_figure(table, baseline)
    write_report(table, baseline)
    print(f"Monitored {len(table)} OOT cohorts against baseline {baseline}.")
    print(f"Monitoring report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()

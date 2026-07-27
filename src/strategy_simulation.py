"""Simulate pass, review and reject policies on out-of-time XGBoost scores."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

from fannie_mae_pipeline import PROJECT_ROOT


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SCORES_PATH = PROCESSED_DIR / "oot_test_scores.csv"
SCENARIOS_PATH = PROJECT_ROOT / "config" / "strategy_scenarios.csv"
OUTPUT_PATH = PROCESSED_DIR / "strategy_simulation_summary.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "strategy_simulation_report.md"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TARGET_COLUMN = "bad_12m_90dpd"
CHINESE_FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")


def configure_plot_font() -> None:
    """Set a CJK-capable font so every report chart remains bilingual."""
    if CHINESE_FONT_PATH.exists():
        font_manager.fontManager.addfont(str(CHINESE_FONT_PATH))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=CHINESE_FONT_PATH).get_name()
    else:
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def apply_strategy(scores: pd.DataFrame, pass_max_pd: float, review_max_pd: float, lgd: float) -> pd.DataFrame:
    """Assign operational actions and expected loss without using future outcomes."""
    if not 0 < pass_max_pd < review_max_pd < 1:
        raise ValueError("Thresholds must satisfy 0 < pass_max_pd < review_max_pd < 1.")
    result = scores.copy()
    result["action"] = "Reject / 拒绝"
    result.loc[result["xgboost_pd"] <= review_max_pd, "action"] = "Manual review / 人工审核"
    result.loc[result["xgboost_pd"] <= pass_max_pd, "action"] = "Pass / 通过"
    exposure = result.get("original_upb", pd.Series(1.0, index=result.index)).fillna(0)
    result["expected_loss"] = result["xgboost_pd"] * exposure * lgd
    return result


def summarize_strategy(scores: pd.DataFrame, scenario: pd.Series) -> pd.DataFrame:
    """Summarize operational volume, realised bad rate and expected loss by action."""
    assigned = apply_strategy(
        scores,
        float(scenario["pass_max_pd"]),
        float(scenario["review_max_pd"]),
        float(scenario["lgd_assumption"]),
    )
    summary = (
        assigned.groupby("action", observed=True)
        .agg(
            loan_count=(TARGET_COLUMN, "size"),
            action_rate=(TARGET_COLUMN, "size"),
            observed_bad_rate=(TARGET_COLUMN, "mean"),
            average_predicted_pd=("xgboost_pd", "mean"),
            exposure_amount=("original_upb", "sum") if "original_upb" in assigned else ("xgboost_pd", "size"),
            expected_loss=("expected_loss", "sum"),
        )
        .reset_index()
    )
    summary["action_rate"] = summary["action_rate"] / len(assigned)
    summary.insert(0, "scenario", scenario["scenario"])
    summary["pass_max_pd"] = float(scenario["pass_max_pd"])
    summary["review_max_pd"] = float(scenario["review_max_pd"])
    return summary


def scenario_overview(summary: pd.DataFrame) -> pd.DataFrame:
    """Create decision-ready scenario metrics rather than one uncontextualized cutoff."""
    rows = []
    for scenario, group in summary.groupby("scenario", sort=False):
        pass_row = group.loc[group["action"] == "Pass / 通过"]
        review_row = group.loc[group["action"] == "Manual review / 人工审核"]
        accepted = group.loc[group["action"].isin(["Pass / 通过", "Manual review / 人工审核"])]
        rejected = group.loc[group["action"] == "Reject / 拒绝"]
        rows.append(
            {
                "scenario": scenario,
                "pass_rate": pass_row["action_rate"].sum(),
                "review_rate": review_row["action_rate"].sum(),
                "approval_or_review_rate": accepted["action_rate"].sum(),
                "pass_bad_rate": pass_row["observed_bad_rate"].iloc[0] if not pass_row.empty else float("nan"),
                # Manual-review loans are provisionally treated as booked. This makes
                # the scenario explicit; production monitoring should replace it with
                # the observed post-review decision.
                "provisional_booked_expected_loss": accepted["expected_loss"].sum(),
                "expected_loss_avoided_by_reject": rejected["expected_loss"].sum(),
                "pre_decision_portfolio_expected_loss": group["expected_loss"].sum(),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str], percentage_columns: set[str]) -> str:
    """Render compact bilingual decision tables."""
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame[columns].itertuples(index=False, name=None):
        values = []
        for column, value in zip(columns, row):
            if column in percentage_columns:
                values.append(f"{value:.2%}")
            elif isinstance(value, float):
                values.append(f"{value:,.2f}")
            elif isinstance(value, int):
                values.append(f"{value:,}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def save_figure(overview: pd.DataFrame) -> None:
    """Visualize the risk-retention trade-off across policy scenarios."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    configure_plot_font()
    labels = overview["scenario"].str.split(" / ").str[0]
    positions = range(len(overview))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    reject_rate = 1 - overview["approval_or_review_rate"]
    width = 0.36
    axes[0].bar([position - width / 2 for position in positions], overview["review_rate"], width=width, color="#f4a261", label="Manual review / 人工审核")
    axes[0].bar([position + width / 2 for position in positions], reject_rate, width=width, color="#b23a48", label="Reject / 拒绝")
    axes[0].set_xticks(list(positions), labels)
    axes[0].yaxis.set_major_formatter("{x:.2%}")
    axes[0].set_title("Review and reject volume / 审核与拒绝量")
    axes[0].set_ylabel("Share of OOT loans / 时间外贷款占比")
    axes[0].legend(fontsize=8)

    booked = overview["provisional_booked_expected_loss"] / 1_000_000
    avoided = overview["expected_loss_avoided_by_reject"] / 1_000_000
    axes[1].bar(positions, booked, color="#b23a48", label="Booked EL / 放款后预期损失")
    axes[1].bar(positions, avoided, bottom=booked, color="#6c757d", label="EL avoided / 拒绝规避损失")
    axes[1].set_xticks(list(positions), labels)
    axes[1].set_title("Expected-loss trade-off / 预期损失取舍")
    axes[1].set_ylabel("Millions of currency units / 百万货币单位")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "strategy_tradeoff.png", dpi=160)
    plt.close(fig)


def write_report(detail: pd.DataFrame, overview: pd.DataFrame) -> None:
    """Write a bilingual decision-policy report with explicit assumptions."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    display_overview = overview.rename(
        columns={
            "scenario": "Scenario / 方案", "pass_rate": "Pass rate / 通过率", "review_rate": "Review rate / 审核率",
            "approval_or_review_rate": "Non-reject rate / 非拒绝率", "pass_bad_rate": "Pass bad rate / 通过客群坏样本率",
            "provisional_booked_expected_loss": "Booked EL / 放款后预期损失",
            "expected_loss_avoided_by_reject": "EL avoided / 拒绝规避损失",
        }
    )
    report = f"""# Strategy Simulation Report / 策略模拟报告

## Purpose / 目的

This layer translates out-of-time XGBoost PD estimates into pass, manual-review and reject actions. It evaluates several scenarios instead of asserting that one arbitrary threshold is optimal. 本层将时间外 XGBoost PD 估计转化为通过、人工审核和拒绝动作；它比较多个方案，而非宣称任意单一阈值最优。

## Scenario comparison / 方案对比

{markdown_table(display_overview, ['Scenario / 方案', 'Pass rate / 通过率', 'Review rate / 审核率', 'Non-reject rate / 非拒绝率', 'Pass bad rate / 通过客群坏样本率', 'Booked EL / 放款后预期损失', 'EL avoided / 拒绝规避损失'], {'Pass rate / 通过率', 'Review rate / 审核率', 'Non-reject rate / 非拒绝率', 'Pass bad rate / 通过客群坏样本率'})}

![Strategy trade-off](figures/strategy_tradeoff.png)

## Reading the output / 如何阅读结果

- A lower pass threshold usually reduces pass rate and pass-population bad rate. 更低的通过阈值通常会降低通过率，同时降低通过客群坏样本率。
- The manual-review band consumes operational capacity and needs a separate review policy. 人工审核区间会占用运营产能，还需要独立的人工审核规则。
- Expected loss is `PD × exposure × LGD`. LGD is an assumption in `config/strategy_scenarios.csv`, not an observed recovery result. 预期损失为 `PD × 暴露金额 × LGD`；LGD 来自配置假设，并非实际回收率结果。
- "Booked EL" assumes manual-review loans are ultimately approved; "EL avoided" is the expected loss attached to rejected loans. In a live system, both should be reconciled to the actual post-review decision. “放款后预期损失”暂按人工审核贷款最终放款计算；“拒绝规避损失”是拒绝贷款对应的预期损失。线上系统应以实际复核结果回填这两个指标。

## Action detail / 动作明细

{markdown_table(detail.rename(columns={'scenario': 'Scenario / 方案', 'action': 'Action / 动作', 'loan_count': 'Loan count / 贷款数', 'action_rate': 'Action rate / 动作占比', 'observed_bad_rate': 'Observed bad rate / 观察坏样本率', 'average_predicted_pd': 'Average PD / 平均 PD', 'expected_loss': 'Expected loss / 预期损失'}), ['Scenario / 方案', 'Action / 动作', 'Loan count / 贷款数', 'Action rate / 动作占比', 'Observed bad rate / 观察坏样本率', 'Average PD / 平均 PD', 'Expected loss / 预期损失'], {'Action rate / 动作占比', 'Observed bad rate / 观察坏样本率', 'Average PD / 平均 PD'})}
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    """Run the configured policy scenarios over OOT prediction results."""
    if not SCORES_PATH.exists():
        raise FileNotFoundError("OOT score file not found. Run src/oot_risk_modelling.py first.")
    scores = pd.read_csv(SCORES_PATH)
    scenarios = pd.read_csv(SCENARIOS_PATH)
    summaries = [summarize_strategy(scores, scenario) for _, scenario in scenarios.iterrows()]
    detail = pd.concat(summaries, ignore_index=True)
    overview = scenario_overview(detail)
    detail.to_csv(OUTPUT_PATH, index=False)
    overview.to_csv(PROCESSED_DIR / "strategy_scenario_overview.csv", index=False)
    save_figure(overview)
    write_report(detail, overview)
    print(f"Simulated {len(scenarios)} policy scenarios on {len(scores):,} OOT loans.")
    print(f"Strategy report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()

"""Profile UCI credit-default data for quality issues and risk segmentation."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "default_of_credit_card_clients.xls"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROJECT_ROOT / "reports" / "data_quality_summary.md"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
    "default%20of%20credit%20card%20clients.xls"
)
TARGET_COLUMN = "default_payment_next_month"
PAYMENT_STATUS_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLUMNS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENT_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]
MONEY_COLUMNS = ["LIMIT_BAL", *BILL_COLUMNS, *PAYMENT_COLUMNS]

CHINESE_FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")


def configure_plot_font() -> None:
    """Apply a local CJK-capable font after any Matplotlib style reset."""
    if CHINESE_FONT_PATH.exists():
        font_manager.fontManager.addfont(str(CHINESE_FONT_PATH))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=CHINESE_FONT_PATH).get_name()
    else:
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def download_data(force: bool = False) -> None:
    """Fetch the original UCI workbook once."""
    if RAW_DATA_PATH.exists() and not force:
        return
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading source data to {RAW_DATA_PATH}")
    urlretrieve(DATA_URL, RAW_DATA_PATH)


def load_data() -> pd.DataFrame:
    """Load the source workbook and normalize its target-field name."""
    frame = pd.read_excel(RAW_DATA_PATH, header=1)
    return frame.rename(columns={"default payment next month": TARGET_COLUMN})


def field_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Build a field-level completeness and cardinality profile."""
    return pd.DataFrame(
        {
            "column": frame.columns,
            "dtype": frame.dtypes.astype(str).values,
            "missing_count": frame.isna().sum().values,
            "missing_rate": frame.isna().mean().values,
            "unique_count": frame.nunique(dropna=True).values,
        }
    )


def code_quality(frame: pd.DataFrame) -> pd.DataFrame:
    """Check documented categorical codes and retain observed distributions."""
    expected_codes = {
        "SEX": {1, 2},
        "EDUCATION": {1, 2, 3, 4},
        "MARRIAGE": {1, 2, 3},
    }
    rows: list[dict[str, object]] = []
    for column, valid_codes in expected_codes.items():
        observed = sorted(frame[column].dropna().unique())
        unexpected_count = int((~frame[column].isin(valid_codes)).sum())
        rows.append(
            {
                "field": column,
                "expected_codes": ", ".join(map(str, sorted(valid_codes))),
                "observed_codes": ", ".join(map(str, observed)),
                "unexpected_count": unexpected_count,
                "unexpected_rate": unexpected_count / len(frame),
                "note": "Treat unexpected codes as 'other/unknown' in later features.",
            }
        )
    for column in PAYMENT_STATUS_COLUMNS:
        invalid_count = int(((frame[column] < -2) | (frame[column] > 8)).sum())
        rows.append(
            {
                "field": column,
                "expected_codes": "-2 to 8",
                "observed_codes": f"{frame[column].min()} to {frame[column].max()}",
                "unexpected_count": invalid_count,
                "unexpected_rate": invalid_count / len(frame),
                "note": "-2 to 0 represent no-delay states; 1 to 8 represent delay months.",
            }
        )
    return pd.DataFrame(rows)


def money_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Profile monetary fields and flag values outside the central 1%-99% range."""
    rows: list[dict[str, object]] = []
    for column in MONEY_COLUMNS:
        series = frame[column]
        p01, p99 = series.quantile([0.01, 0.99])
        rows.append(
            {
                "field": column,
                "minimum": series.min(),
                "p01": p01,
                "median": series.median(),
                "p99": p99,
                "maximum": series.max(),
                "zero_count": int((series == 0).sum()),
                "negative_count": int((series < 0).sum()),
                "above_p99_count": int((series > p99).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_segments(frame: pd.DataFrame) -> pd.DataFrame:
    """Create analyst-friendly segment labels without changing the raw fields."""
    segmented = frame.copy()
    segmented["gender_segment"] = segmented["SEX"].map({1: "Male", 2: "Female"}).fillna("Unknown")
    segmented["education_segment"] = segmented["EDUCATION"].map(
        {1: "Graduate school", 2: "University", 3: "High school", 4: "Other"}
    ).fillna("Other / unknown")
    segmented["marriage_segment"] = segmented["MARRIAGE"].map(
        {1: "Married", 2: "Single", 3: "Other"}
    ).fillna("Other / unknown")
    segmented["age_segment"] = pd.cut(
        segmented["AGE"], bins=[0, 25, 35, 45, 55, float("inf")],
        labels=["<=25", "26-35", "36-45", "46-55", "56+"],
    )
    segmented["limit_segment"] = pd.qcut(
        segmented["LIMIT_BAL"], q=4, duplicates="drop",
        labels=["Q1 low", "Q2", "Q3", "Q4 high"],
    )
    repayment = segmented[PAYMENT_STATUS_COLUMNS]
    segmented["max_delinquency_status"] = repayment.max(axis=1)
    segmented["delinquent_month_count"] = (repayment >= 1).sum(axis=1)
    segmented["recent_delinquency_status"] = segmented["PAY_0"]
    segmented["max_delinquency_segment"] = pd.cut(
        segmented["max_delinquency_status"],
        bins=[float("-inf"), 0, 1, 2, float("inf")],
        labels=["No recorded delinquency", "1 month", "2 months", "3+ months"],
    )
    segmented["delinquent_month_segment"] = pd.cut(
        segmented["delinquent_month_count"],
        bins=[-1, 0, 1, 3, float("inf")],
        labels=["0 months", "1 month", "2-3 months", "4+ months"],
    )
    return segmented


def segment_default_rates(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate volume and default rate for core customer segments."""
    rows: list[pd.DataFrame] = []
    for segment in [
        "gender_segment", "education_segment", "marriage_segment", "age_segment",
        "limit_segment", "max_delinquency_segment", "delinquent_month_segment",
        "recent_delinquency_status",
    ]:
        summary = (
            frame.groupby(segment, observed=True)[TARGET_COLUMN]
            .agg(sample_count="size", default_rate="mean")
            .reset_index()
            .rename(columns={segment: "segment_value"})
        )
        summary.insert(0, "segment", segment.removesuffix("_segment"))
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def save_figures(segment_rates: pd.DataFrame, money_stats: pd.DataFrame) -> None:
    """Save concise visual outputs used in the report and later dashboard."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    configure_plot_font()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, segment in zip(axes, ["age", "limit"]):
        subset = segment_rates[segment_rates["segment"] == segment]
        axis.bar(subset["segment_value"].astype(str), subset["default_rate"], color="#b23a48")
        chinese_title = {"age": "年龄分层违约率", "limit": "额度分层违约率"}[segment]
        axis.set_title(f"Default rate by {segment} segment\n{chinese_title}")
        axis.set_ylabel("Default rate / 违约率")
        axis.tick_params(axis="x", rotation=25)
        axis.yaxis.set_major_formatter("{x:.0%}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "default_rate_by_segment.png", dpi=160)
    plt.close(fig)

    subset = money_stats[money_stats["field"].isin(["LIMIT_BAL", "BILL_AMT1", "PAY_AMT1"])]
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(subset["field"], subset["negative_count"], color="#457b9d", label="Negative values")
    axis.bar(subset["field"], subset["zero_count"], bottom=subset["negative_count"], color="#f4a261", label="Zero values")
    axis.set_title("Zero and negative values in selected monetary fields\n选定金额字段的零值与负值")
    axis.set_ylabel("Record count / 记录数")
    axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "monetary_value_flags.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for axis, segment, title in zip(
        axes,
        ["education", "max_delinquency"],
        [
            "Default rate by education\n教育程度分层违约率",
            "Default rate by worst repayment status\n最大逾期状态分层违约率",
        ],
    ):
        subset = segment_rates[segment_rates["segment"] == segment]
        axis.bar(subset["segment_value"].astype(str), subset["default_rate"], color="#6a994e")
        axis.set_title(title)
        axis.set_ylabel("Default rate / 违约率")
        axis.tick_params(axis="x", rotation=25)
        axis.yaxis.set_major_formatter("{x:.0%}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "default_rate_education_and_delinquency.png", dpi=160)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str], percent_columns: set[str] | None = None) -> str:
    """Render a small dataframe as a Markdown table without extra dependencies."""
    percent_columns = percent_columns or set()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---" for _ in columns]) + " |"
    lines = [header, divider]
    for row in frame[columns].itertuples(index=False, name=None):
        values = []
        for column, value in zip(columns, row):
            if column in percent_columns:
                values.append(f"{value:.2%}")
            elif isinstance(value, float):
                values.append(f"{value:,.0f}")
            else:
                values.append(f"{value:,}" if isinstance(value, int) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def display_segment_table(segment_rates: pd.DataFrame, segment: str) -> str:
    """Render one risk-segmentation question as a small, readable bilingual table."""
    labels = {
        "age": {"<=25": "<=25 / 25 岁及以下", "26-35": "26-35 / 26-35 岁", "36-45": "36-45 / 36-45 岁", "46-55": "46-55 / 46-55 岁", "56+": "56+ / 56 岁及以上"},
        "limit": {"Q1 low": "Q1 low / 最低额度四分位", "Q2": "Q2 / 第二额度四分位", "Q3": "Q3 / 第三额度四分位", "Q4 high": "Q4 high / 最高额度四分位"},
        "education": {"Graduate school": "Graduate school / 研究生", "University": "University / 大学", "High school": "High school / 高中", "Other": "Other / 其他", "Other / unknown": "Other / unknown / 其他或未知"},
        "max_delinquency": {"No recorded delinquency": "No recorded delinquency / 无记录逾期", "1 month": "1 month / 最大逾期 1 个月", "2 months": "2 months / 最大逾期 2 个月", "3+ months": "3+ months / 最大逾期 3 个月及以上"},
        "delinquent_month": {"0 months": "0 months / 逾期 0 个月", "1 month": "1 month / 逾期 1 个月", "2-3 months": "2-3 months / 逾期 2-3 个月", "4+ months": "4+ months / 逾期 4 个月及以上"},
        "recent_delinquency_status": {
            -2: "-2 / No consumption / 无消费", -1: "-1 / Paid duly / 按时还款",
            0: "0 / Revolving credit / 循环信用", 1: "1 / 1-month delay / 逾期 1 个月",
            2: "2 / 2-month delay / 逾期 2 个月", 3: "3 / 3-month delay / 逾期 3 个月",
            4: "4 / 4-month delay / 逾期 4 个月", 5: "5 / 5-month delay / 逾期 5 个月",
            6: "6 / 6-month delay / 逾期 6 个月", 7: "7 / 7-month delay / 逾期 7 个月",
            8: "8 / 8-month delay / 逾期 8 个月",
        },
    }
    table = segment_rates.loc[segment_rates["segment"] == segment, ["segment_value", "sample_count", "default_rate"]].copy()
    table["segment_value"] = table["segment_value"].map(labels[segment]).fillna(table["segment_value"].astype(str))
    table = table.rename(
        columns={
            "segment_value": "Group / 分层",
            "sample_count": "Sample count / 样本量",
            "default_rate": "Default rate / 违约率",
        }
    )
    return markdown_table(
        table,
        ["Group / 分层", "Sample count / 样本量", "Default rate / 违约率"],
        {"Default rate / 违约率"},
    )


def write_report(frame: pd.DataFrame) -> None:
    """Write tables, charts and an analyst-oriented quality report."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    profile = field_profile(frame)
    codes = code_quality(frame)
    money = money_profile(frame)
    segments = segment_default_rates(build_segments(frame))

    profile.to_csv(PROCESSED_DIR / "field_profile.csv", index=False)
    codes.to_csv(PROCESSED_DIR / "categorical_code_checks.csv", index=False)
    money.to_csv(PROCESSED_DIR / "monetary_field_profile.csv", index=False)
    segments.to_csv(PROCESSED_DIR / "segment_default_rates.csv", index=False)
    save_figures(segments, money)

    target = frame[TARGET_COLUMN].value_counts(normalize=True).sort_index()
    target_table = pd.DataFrame({"default_flag": target.index, "sample_rate": target.values})
    negative_bill_fields = money.loc[money["field"].str.startswith("BILL_"), "negative_count"].sum()

    report = f"""# Data Quality and Risk Segmentation Report / 数据质量与风险分层报告

## Scope / 分析范围

- Source / 数据来源: [UCI Default of Credit Card Clients]({DATA_URL})
- Sample / 样本规模: {len(frame):,} customers / 客户; {frame.shape[1]} fields / 字段
- Target / 目标变量: `{TARGET_COLUMN}` (1 = default / 违约, 0 = non-default / 未违约)

## Completeness and uniqueness / 完整性与唯一性

| Metric / 指标 | Result / 结果 |
| --- | ---: |
| Fully duplicated records / 完全重复记录 | {frame.duplicated().sum():,} |
| Duplicated customer IDs / 重复客户 ID | {frame['ID'].duplicated().sum():,} |
| Fields with missing values / 存在缺失值的字段 | {(profile['missing_count'] > 0).sum():,} |

{markdown_table(target_table, ['default_flag', 'sample_rate'], {'sample_rate'})}

**Interpretation / 解读：** The target is moderately imbalanced, with enough positive samples for a baseline scorecard. Future model evaluation should use stratified splits and report AUC, KS, recall, and approval-risk trade-offs rather than accuracy alone. 目标变量存在适度类别不平衡，但违约样本量足以支持基准评分模型；后续应采用分层切分，并报告 AUC、KS、召回率及通过率与风险间的权衡，而非只看准确率。

## Categorical code checks / 分类编码核查

{markdown_table(codes, ['field', 'expected_codes', 'observed_codes', 'unexpected_count', 'unexpected_rate'], {'unexpected_rate'})}

**Interpretation / 解读：** `EDUCATION` values 0, 5 and 6, and `MARRIAGE` value 0 are outside the primary documented categories. They should be retained and grouped as "other/unknown" rather than deleted, since their presence may carry risk information. `EDUCATION` 的 0、5、6 和 `MARRIAGE` 的 0 不属于主要文档定义的类别；应保留并归为“其他/未知”，而非删除，因为这些取值本身可能携带风险信息。

## Monetary field checks / 金额字段核查

The full result is saved to `data/processed/monetary_field_profile.csv`. Billing fields contain {negative_bill_fields:,} negative values in total. A negative bill can be a valid credit balance, so it is a review flag rather than an automatic error. Extreme values are summarized with the 1st and 99th percentiles; model treatment will be decided after checking their risk relationship.

完整结果见 `data/processed/monetary_field_profile.csv`。账单字段合计有 {negative_bill_fields:,} 个负值；负账单可能表示账户贷方余额，因此属于需核查的业务现象，而非自动判错。极端值以 1% 和 99% 分位数汇总，后续将在确认其与风险的关系后决定模型处理方式。

## Customer risk segmentation / 客群风险分层

### Age / 年龄

这张表回答：不同年龄段的客户，下一月违约率是否不同？年龄仅用于描述性分层，不能单独作为授信决策依据。

{display_segment_table(segments, 'age')}

### Credit limit / 授信额度

这张表回答：额度不同的客户，风险水平是否不同？Q1 是样本中额度最低的四分之一，Q4 是最高的四分之一。

{display_segment_table(segments, 'limit')}

### Education / 教育程度

这张表回答：教育程度字段在不同取值下的观察违约率。样本量很小的“其他/未知”组只作描述，不应据此得出稳定结论。

{display_segment_table(segments, 'education')}

**Interpretation / 解读：** Among the displayed customer variables, credit limit shows the strongest separation: the lowest-limit quartile has a materially higher observed default rate than the highest-limit quartile. This is descriptive evidence, not proof that limit causes default. 在展示的客户变量中，授信额度的风险区分最明显：最低额度四分位的观察违约率显著高于最高四分位；这只是描述性证据，并不证明额度导致违约。

## Historical delinquency analysis / 历史逾期分析

`PAY_0` is the most recent repayment status; `PAY_2` to `PAY_6` are older months. For this analysis, a value >=1 is classified as a delinquency month. The project derives the worst status and the count of delinquent months across all six observations.

`PAY_0` 为最近一期还款状态，`PAY_2` 至 `PAY_6` 为更早月份。本文将取值 >=1 定义为发生逾期，并基于六期历史构造最大逾期状态与逾期月数。

### Worst repayment status / 历史最大逾期状态

这张表回答：过去六期中，客户出现过的最严重逾期状态，和下月违约有什么关系？这是“逾期严重程度”的指标。

{display_segment_table(segments, 'max_delinquency')}

### Number of delinquent months / 历史逾期月数

这张表回答：过去六期中，客户发生逾期的月份越多，下月违约率是否越高？这是“逾期持续性”的指标。

{display_segment_table(segments, 'delinquent_month')}

### Most recent repayment status / 最近一期还款状态

这张表回答：最近一个月的还款状态，和下月违约有什么关系？尾部取值（4-8）的样本量很小，比例会波动，因此观察重点应放在样本量较大的 0、1、2 三组。

{display_segment_table(segments, 'recent_delinquency_status')}

**Interpretation / 解读：** Repayment history has a direct and substantially stronger relationship with next-month default than the demographic segments above. It should be a central feature family in the later scorecard, subject to train-validation and stability checks. 与上述客群属性相比，还款历史与下月违约存在更直接且更强的关系；它应成为后续评分模型的核心特征族，但仍需经过训练验证和稳定性检验。

Charts / 图表: `reports/figures/default_rate_by_segment.png`, `reports/figures/default_rate_education_and_delinquency.png`, and `reports/figures/monetary_value_flags.png`.

## Findings for the next stage / 下一阶段要点

1. **Data completeness / 数据完整性：** There are no missing, duplicate-record, or duplicated-ID issues in this source. 本数据不存在缺失、重复记录或重复客户 ID 问题。
2. **Category recoding / 分类重编码：** Recoding is required before modelling to handle undocumented education and marriage codes consistently. 建模前需统一处理未文档化的教育和婚姻编码。
3. **Leading risk signal / 核心风险信号：** Historical repayment status shows the clearest risk separation and is the leading candidate feature family for the next stage. 历史还款状态的风险区分最清晰，是下一阶段首要的候选特征族。
4. **Monitoring limitation / 监控限制：** This dataset has no application date or observation date. The later monitoring module must clearly label any baseline-versus-observation split as a simulated monitoring exercise. 数据集没有申请日期或观察日期，后续任何基准期与观察期的划分都必须明确标注为模拟监控。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redownload", action="store_true", help="Download a fresh source file.")
    args = parser.parse_args()
    download_data(force=args.redownload)
    frame = load_data()
    write_report(frame)
    print(f"Loaded {len(frame):,} records and {frame.shape[1]} fields.")
    print(f"Data quality report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()

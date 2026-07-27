"""Normalize Fannie Mae loan files and build point-in-time PD modelling snapshots."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROJECT_ROOT / "reports" / "fannie_data_preparation_report.md"

ORIGINATION_ALIASES = {
    "loan_id": ["loan_id", "loan_identifier", "loan_sequence_number"],
    "origination_date": ["origination_date", "orig_date"],
    "original_upb": ["original_upb", "original_unpaid_principal_balance", "original_loan_amount"],
    "original_ltv": ["original_ltv", "original_loan_to_value"],
    "original_dti": ["original_dti", "original_debt_to_income_ratio"],
    "borrower_credit_score": ["borrower_credit_score", "credit_score", "fico"],
    "original_interest_rate": ["original_interest_rate", "interest_rate"],
    "original_loan_term": ["original_loan_term", "loan_term"],
    "origination_channel": ["origination_channel", "channel"],
    "loan_purpose": ["loan_purpose", "purpose"],
    "occupancy_status": ["occupancy_status", "occupancy"],
    "property_state": ["property_state", "state"],
    "property_type": ["property_type"],
    "product_type": ["product_type"],
}
PERFORMANCE_ALIASES = {
    "loan_id": ["loan_id", "loan_identifier", "loan_sequence_number"],
    "reporting_month": ["monthly_reporting_period", "reporting_month", "report_date"],
    "current_actual_upb": ["current_actual_upb", "current_upb", "current_unpaid_principal_balance"],
    "current_delinquency_status": ["current_loan_delinquency_status", "current_delinquency_status", "delinquency_status"],
    "loan_age": ["loan_age", "months_on_book"],
    "remaining_months_to_legal_maturity": ["remaining_months_to_legal_maturity"],
    "zero_balance_code": ["zero_balance_code"],
    "current_interest_rate": ["current_interest_rate"],
    "modification_flag": ["modification_flag"],
}
ORIGINATION_REQUIRED = ["loan_id", "origination_date", "original_upb", "original_ltv", "borrower_credit_score"]
PERFORMANCE_REQUIRED = ["loan_id", "reporting_month", "current_actual_upb", "current_delinquency_status"]

# Official sample files are pipe-delimited and headerless. The first fields are
# stable across the legacy and current layouts; trailing fields may expand over time.
FANNIE_ACQUISITION_POSITIONAL = [
    "loan_id", "origination_channel", "seller_name", "original_interest_rate", "original_upb",
    "original_loan_term", "origination_date", "first_payment_date", "original_ltv", "original_cltv",
    "number_of_borrowers", "original_dti", "borrower_credit_score", "first_time_home_buyer_indicator",
    "loan_purpose", "property_type", "number_of_units", "occupancy_status", "property_state",
    "zip_code_short", "mortgage_insurance_percentage", "product_type", "co_borrower_credit_score",
    "mortgage_insurance_type", "relocation_mortgage_indicator",
]
FANNIE_PERFORMANCE_POSITIONAL = [
    "reference_pool_id", "loan_id", "reporting_month", "origination_channel", "seller_name",
    "servicer_name", "master_servicer", "original_interest_rate", "current_interest_rate", "original_upb",
    "upb_at_issuance", "current_actual_upb", "original_loan_term", "origination_date", "first_payment_date",
    "loan_age", "remaining_months_to_legal_maturity", "remaining_months_to_maturity", "maturity_date",
    "original_ltv", "original_cltv", "number_of_borrowers", "original_dti", "borrower_credit_score",
    "co_borrower_credit_score", "first_time_home_buyer_indicator", "loan_purpose", "property_type",
    "number_of_units", "occupancy_status", "property_state", "msa_msda", "zip_code_short",
    "mortgage_insurance_percentage", "amortization_type", "prepayment_penalty_indicator",
    "interest_only_indicator", "interest_only_first_payment_date", "months_to_amortization",
    "current_delinquency_status", "loan_payment_history", "modification_flag", "mortgage_insurance_cancellation_indicator",
    "zero_balance_code", "zero_balance_effective_date",
]


def clean_column_name(value: str) -> str:
    """Normalize external source headers into stable snake-case lookup keys."""
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def read_delimited(path: Path, positional_layout: list[str]) -> pd.DataFrame:
    """Read a source extract, accepting either headers or official headerless layout."""
    if not path.exists():
        raise FileNotFoundError(f"Source file does not exist: {path}")
    frame = pd.read_csv(path, sep=None, engine="python", dtype=str, header=None)
    if len(frame.columns) < 2:
        raise ValueError(f"{path.name} has fewer than two columns. Confirm its delimiter.")
    first_row = [clean_column_name(str(value)) for value in frame.iloc[0].fillna("")]
    known_headers = {alias for aliases in [*ORIGINATION_ALIASES.values(), *PERFORMANCE_ALIASES.values()] for alias in aliases}
    if len(set(first_row).intersection(known_headers)) >= 2:
        frame.columns = first_row
        frame = frame.iloc[1:].reset_index(drop=True)
    else:
        column_names = positional_layout + [f"extra_field_{index}" for index in range(len(positional_layout), len(frame.columns))]
        frame.columns = column_names[: len(frame.columns)]
    return frame


def resolve_aliases(frame: pd.DataFrame, aliases: dict[str, list[str]], required: list[str]) -> pd.DataFrame:
    """Map source-specific header names to the platform's canonical data contract."""
    rename_map: dict[str, str] = {}
    for canonical, options in aliases.items():
        source_column = next((option for option in options if option in frame.columns), None)
        if source_column:
            rename_map[source_column] = canonical
    normalized = frame.rename(columns=rename_map)
    missing = [column for column in required if column not in normalized.columns]
    if missing:
        available = ", ".join(frame.columns[:20])
        raise ValueError(
            "Input does not satisfy the data contract. Missing canonical fields: "
            f"{missing}. First available headers: {available}"
        )
    return normalized


def normalize_origination(path: Path) -> pd.DataFrame:
    """Create typed origination records from a source extract."""
    frame = resolve_aliases(read_delimited(path, FANNIE_ACQUISITION_POSITIONAL), ORIGINATION_ALIASES, ORIGINATION_REQUIRED)
    frame = frame.drop_duplicates(subset="loan_id", keep="last").copy()
    frame["origination_date"] = parse_month_date(frame["origination_date"])
    for column in ["original_upb", "original_ltv", "original_dti", "borrower_credit_score", "original_interest_rate", "original_loan_term"]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    categorical_columns = set(ORIGINATION_ALIASES).intersection(frame.columns) - {
        "loan_id", "origination_date", "original_upb", "original_ltv", "original_dti",
        "borrower_credit_score", "original_interest_rate", "original_loan_term",
    }
    for column in categorical_columns:
        frame[column] = frame[column].fillna("Unknown").astype(str)
    return frame


def normalize_performance(path: Path) -> pd.DataFrame:
    """Create typed monthly loan-performance records from a source extract."""
    frame = resolve_aliases(read_delimited(path, FANNIE_PERFORMANCE_POSITIONAL), PERFORMANCE_ALIASES, PERFORMANCE_REQUIRED)
    frame = frame.copy()
    frame["reporting_month"] = parse_month_date(frame["reporting_month"])
    for column in [
        "current_actual_upb", "current_delinquency_status", "loan_age",
        "remaining_months_to_legal_maturity", "zero_balance_code", "current_interest_rate",
    ]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "modification_flag" not in frame:
        frame["modification_flag"] = "Unknown"
    frame["modification_flag"] = frame["modification_flag"].fillna("Unknown").astype(str)
    return frame.drop_duplicates(subset=["loan_id", "reporting_month"], keep="last")


def parse_month_date(values: pd.Series) -> pd.Series:
    """Parse Fannie MMYYYY dates first, then fall back to standard date strings."""
    text = values.astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
    parsed = pd.to_datetime(text, format="%m%Y", errors="coerce")
    fallback = pd.to_datetime(text, errors="coerce")
    return parsed.fillna(fallback)


def months_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
    """Calculate whole month differences while retaining data-frame vectorization."""
    return (later.dt.year - earlier.dt.year) * 12 + (later.dt.month - earlier.dt.month)


def build_pd_snapshot(
    origination: pd.DataFrame,
    performance: pd.DataFrame,
    observation_months: int = 6,
    outcome_months: int = 12,
) -> pd.DataFrame:
    """Build one point-in-time record per loan with a future 90+ DPD outcome.

    Features only use months 1..observation_months. The target uses the following
    outcome_months, preventing future repayment information from leaking into inputs.
    """
    joined = performance.merge(origination, on="loan_id", how="inner", suffixes=("", "_orig"))
    joined["month_on_book"] = months_between(joined["reporting_month"], joined["origination_date"])
    joined = joined.loc[joined["month_on_book"] >= 1].copy()
    joined = joined.dropna(subset=["origination_date", "reporting_month", "current_delinquency_status"])

    availability = joined.groupby("loan_id", as_index=False)["month_on_book"].max()
    eligible_ids = availability.loc[
        availability["month_on_book"] >= observation_months + outcome_months, "loan_id"
    ]
    joined = joined.loc[joined["loan_id"].isin(eligible_ids)].copy()
    history = joined.loc[joined["month_on_book"].between(1, observation_months)].copy()
    future = joined.loc[
        joined["month_on_book"].between(observation_months + 1, observation_months + outcome_months)
    ].copy()
    if history.empty or future.empty:
        raise ValueError("No eligible history/future windows were found. Download longer performance history.")

    history = history.sort_values(["loan_id", "month_on_book"])
    aggregates = history.groupby("loan_id").agg(
        hist_max_dpd=("current_delinquency_status", "max"),
        hist_30dpd_months=("current_delinquency_status", lambda values: int((values >= 1).sum())),
        hist_avg_upb=("current_actual_upb", "mean"),
        hist_upb_volatility=("current_actual_upb", "std"),
        hist_modified_months=("modification_flag", lambda values: int(values.eq("Y").sum())),
    )
    latest = history.groupby("loan_id", as_index=False).tail(1).set_index("loan_id")
    aggregates["hist_latest_upb"] = latest["current_actual_upb"]
    aggregates["hist_latest_dpd"] = latest["current_delinquency_status"]
    aggregates["hist_upb_to_original_upb"] = aggregates["hist_latest_upb"] / latest["original_upb"]
    aggregates = aggregates.replace([np.inf, -np.inf], np.nan)

    target = future.groupby("loan_id")["current_delinquency_status"].max().ge(3).astype(int)
    static_columns = [column for column in ORIGINATION_ALIASES if column in origination.columns and column != "loan_id"]
    static = origination.set_index("loan_id")[static_columns]
    snapshot = static.join(aggregates, how="inner").join(target.rename("bad_12m_90dpd"), how="inner")
    snapshot["observation_date"] = snapshot["origination_date"] + pd.DateOffset(months=observation_months)
    snapshot = snapshot.reset_index()
    if snapshot["bad_12m_90dpd"].nunique() < 2:
        raise ValueError("The constructed outcome has one class only; broaden the source period or validate DPD coding.")
    return snapshot


def quality_summary(origination: pd.DataFrame, performance: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    """Provide auditable row-count and completeness checks for each pipeline stage."""
    rows = [
        ("Origination loans / 起源贷款", len(origination), origination["loan_id"].nunique(), origination.isna().mean().mean()),
        ("Monthly performance rows / 月度表现记录", len(performance), performance["loan_id"].nunique(), performance.isna().mean().mean()),
        ("Eligible PD snapshots / 可建模 PD 快照", len(snapshot), snapshot["loan_id"].nunique(), snapshot.isna().mean().mean()),
    ]
    return pd.DataFrame(rows, columns=["stage", "row_count", "unique_loans", "average_field_missing_rate"])


def markdown_table(frame: pd.DataFrame, columns: list[str], percent_columns: set[str]) -> str:
    """Render a compact Markdown table."""
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame[columns].itertuples(index=False, name=None):
        formatted = []
        for column, value in zip(columns, row):
            if column in percent_columns:
                formatted.append(f"{value:.2%}")
            elif isinstance(value, (int, np.integer)):
                formatted.append(f"{value:,}")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, snapshot: pd.DataFrame, observation_months: int, outcome_months: int) -> None:
    """Write a bilingual data-preparation report for auditability."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# Fannie Mae Data Preparation Report / Fannie Mae 数据准备报告

## Point-in-time design / 时点设计

Features use loan-performance records from months 1-{observation_months} after origination. The label is whether a loan reaches 90+ days past due during months {observation_months + 1}-{observation_months + outcome_months}. 特征仅使用起源后第 1-{observation_months} 个月的月度表现记录；标签定义为第 {observation_months + 1}-{observation_months + outcome_months} 个月内是否发生 90 天及以上逾期。

This separation is deliberate: future performance must not enter a PD feature. 此划分用于避免将未来还款表现泄露到 PD 特征中。

## Pipeline quality checks / 流水线质量核查

{markdown_table(summary, ['stage', 'row_count', 'unique_loans', 'average_field_missing_rate'], {'average_field_missing_rate'})}

## Constructed modelling sample / 构造的建模样本

| Metric / 指标 | Value / 数值 |
| --- | ---: |
| Eligible loans / 可用贷款数 | {len(snapshot):,} |
| 90+ DPD bad rate / 90 天以上逾期坏样本率 | {snapshot['bad_12m_90dpd'].mean():.2%} |
| Earliest origination / 最早起源日期 | {snapshot['origination_date'].min():%Y-%m-%d} |
| Latest origination / 最晚起源日期 | {snapshot['origination_date'].max():%Y-%m-%d} |

## Outputs / 输出文件

- `data/processed/fannie_pd_snapshot.csv`: one loan per row with static origination features, six-month behavioural features, observation date, and the future 12-month bad flag. 每行一笔贷款，包含起源特征、六个月行为特征、观察日期与未来十二个月坏样本标识。
- `data/processed/fannie_pipeline_quality.csv`: pipeline row counts and missingness checks. 流水线行数与缺失率检查。

## Next control / 下一项控制

Use `observation_date` to train on earlier cohorts and evaluate on later cohorts. Do not random-split this main dataset. 使用 `observation_date` 在较早批次训练、较晚批次测试；主数据集不得随机切分。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    """CLI entry point for main-case source ingestion and label construction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origination", type=Path, nargs="+", required=True, help="One or more headered origination extracts.")
    parser.add_argument("--performance", type=Path, nargs="+", required=True, help="One or more headered monthly performance extracts.")
    parser.add_argument("--observation-months", type=int, default=6)
    parser.add_argument("--outcome-months", type=int, default=12)
    args = parser.parse_args()
    if args.observation_months < 1 or args.outcome_months < 1:
        raise ValueError("Observation and outcome windows must both be at least one month.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    origination = pd.concat([normalize_origination(path) for path in args.origination], ignore_index=True)
    origination = origination.drop_duplicates(subset="loan_id", keep="last")
    performance = pd.concat([normalize_performance(path) for path in args.performance], ignore_index=True)
    performance = performance.drop_duplicates(subset=["loan_id", "reporting_month"], keep="last")
    snapshot = build_pd_snapshot(origination, performance, args.observation_months, args.outcome_months)
    summary = quality_summary(origination, performance, snapshot)

    origination.to_csv(PROCESSED_DIR / "fannie_origination_normalized.csv", index=False)
    performance.to_csv(PROCESSED_DIR / "fannie_performance_normalized.csv", index=False)
    snapshot.to_csv(PROCESSED_DIR / "fannie_pd_snapshot.csv", index=False)
    summary.to_csv(PROCESSED_DIR / "fannie_pipeline_quality.csv", index=False)
    write_report(summary, snapshot, args.observation_months, args.outcome_months)
    print(f"Built {len(snapshot):,} point-in-time PD snapshots.")
    print(f"Data preparation report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()

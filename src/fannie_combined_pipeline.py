"""Build point-in-time PD snapshots from Fannie Mae combined monthly files with DuckDB."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from fannie_mae_pipeline import PROJECT_ROOT


RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fannie_mae"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATABASE_PATH = PROCESSED_DIR / "fannie_mae.duckdb"
SNAPSHOT_PATH = PROCESSED_DIR / "fannie_pd_snapshot.csv"
QUALITY_PATH = PROCESSED_DIR / "fannie_pipeline_quality.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "fannie_data_preparation_report.md"


def sql_file_list(files: list[Path]) -> str:
    """Quote local source paths for DuckDB's multi-file CSV reader."""
    quoted = ["'" + str(path.resolve()).replace("\\", "/").replace("'", "''") + "'" for path in files]
    return "[" + ", ".join(quoted) + "]"


def build_snapshot(connection: duckdb.DuckDBPyConnection, files: list[Path]) -> None:
    """Materialize a six-month feature window and the next-twelve-month 90+ DPD label.

    The official combined format is pipe-delimited and headerless. Column positions
    below are verified against the official sample and CRT field-layout glossary.
    """
    file_list = sql_file_list(files)
    connection.execute(
        f"""
        CREATE OR REPLACE TABLE fannie_pd_snapshot AS
        WITH source_rows AS (
            SELECT
                column001 AS loan_id,
                CAST(try_strptime(column002, '%m%Y') AS DATE) AS reporting_month,
                TRY_CAST(column007 AS DOUBLE) AS original_interest_rate,
                TRY_CAST(column009 AS DOUBLE) AS original_upb,
                TRY_CAST(column011 AS DOUBLE) AS current_actual_upb,
                TRY_CAST(column012 AS INTEGER) AS original_loan_term,
                CAST(try_strptime(column013, '%m%Y') AS DATE) AS origination_date,
                TRY_CAST(column019 AS DOUBLE) AS original_ltv,
                TRY_CAST(column022 AS DOUBLE) AS original_dti,
                TRY_CAST(column023 AS DOUBLE) AS borrower_credit_score,
                column026 AS loan_purpose,
                column027 AS property_type,
                column029 AS occupancy_status,
                column030 AS property_state,
                TRY_CAST(column039 AS INTEGER) AS current_delinquency_status,
                column041 AS modification_flag,
                TRY_CAST(column043 AS INTEGER) AS zero_balance_code,
                filename AS source_file
            FROM read_csv({file_list}, delim='|', header=false, all_varchar=true, filename=true)
        ),
        relevant_rows AS (
            SELECT *, date_diff('month', origination_date, reporting_month) AS month_on_book
            FROM source_rows
            WHERE loan_id IS NOT NULL
              AND origination_date IS NOT NULL
              AND reporting_month IS NOT NULL
              AND current_delinquency_status IS NOT NULL
              AND date_diff('month', origination_date, reporting_month) BETWEEN 1 AND 18
        ),
        eligible_loans AS (
            SELECT loan_id
            FROM relevant_rows
            GROUP BY loan_id
            HAVING COUNT(*) FILTER (WHERE month_on_book BETWEEN 1 AND 6) >= 6
               AND COUNT(*) FILTER (WHERE month_on_book BETWEEN 7 AND 18) >= 12
        ),
        snapshot AS (
            SELECT
                r.loan_id,
                MAX(r.origination_date) AS origination_date,
                MAX(r.origination_date) + INTERVAL 6 MONTH AS observation_date,
                MAX(r.original_upb) AS original_upb,
                MAX(r.original_interest_rate) AS original_interest_rate,
                MAX(r.original_loan_term) AS original_loan_term,
                MAX(r.original_ltv) AS original_ltv,
                MAX(r.original_dti) AS original_dti,
                MAX(r.borrower_credit_score) AS borrower_credit_score,
                MAX(r.loan_purpose) AS loan_purpose,
                MAX(r.property_type) AS property_type,
                MAX(r.occupancy_status) AS occupancy_status,
                MAX(r.property_state) AS property_state,
                MAX(r.current_delinquency_status) FILTER (WHERE r.month_on_book BETWEEN 1 AND 6) AS hist_max_dpd,
                COUNT(*) FILTER (WHERE r.month_on_book BETWEEN 1 AND 6 AND r.current_delinquency_status >= 1) AS hist_30dpd_months,
                AVG(r.current_actual_upb) FILTER (WHERE r.month_on_book BETWEEN 1 AND 6) AS hist_avg_upb,
                STDDEV_SAMP(r.current_actual_upb) FILTER (WHERE r.month_on_book BETWEEN 1 AND 6) AS hist_upb_volatility,
                ARG_MAX(r.current_actual_upb, r.reporting_month) FILTER (WHERE r.month_on_book BETWEEN 1 AND 6) AS hist_latest_upb,
                ARG_MAX(r.current_delinquency_status, r.reporting_month) FILTER (WHERE r.month_on_book BETWEEN 1 AND 6) AS hist_latest_dpd,
                COUNT(*) FILTER (WHERE r.month_on_book BETWEEN 1 AND 6 AND r.modification_flag = 'Y') AS hist_modified_months,
                CASE WHEN MAX(r.current_delinquency_status) FILTER (WHERE r.month_on_book BETWEEN 7 AND 18) >= 3 THEN 1 ELSE 0 END AS bad_12m_90dpd
            FROM relevant_rows r
            INNER JOIN eligible_loans e USING (loan_id)
            GROUP BY r.loan_id
        )
        SELECT *, hist_latest_upb / NULLIF(original_upb, 0) AS hist_upb_to_original_upb
        FROM snapshot
        """
    )


def write_outputs(connection: duckdb.DuckDBPyConnection, files: list[Path]) -> None:
    """Export the model snapshot and a compact, bilingual preparation report."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_file = str(SNAPSHOT_PATH.resolve()).replace("\\", "/").replace("'", "''")
    connection.execute(f"COPY fannie_pd_snapshot TO '{snapshot_file}' (HEADER, DELIMITER ',')")
    summary = connection.execute(
        """
        SELECT
            COUNT(*) AS loan_count,
            AVG(bad_12m_90dpd) AS bad_rate,
            MIN(origination_date) AS earliest_origination,
            MAX(origination_date) AS latest_origination,
            AVG(CASE WHEN borrower_credit_score IS NULL THEN 1.0 ELSE 0.0 END) AS fico_missing_rate,
            AVG(CASE WHEN original_ltv IS NULL THEN 1.0 ELSE 0.0 END) AS ltv_missing_rate
        FROM fannie_pd_snapshot
        """
    ).fetchone()
    quality = (
        "metric,value\n"
        f"loan_count,{summary[0]}\n"
        f"bad_rate,{summary[1]}\n"
        f"earliest_origination,{summary[2]}\n"
        f"latest_origination,{summary[3]}\n"
        f"fico_missing_rate,{summary[4]}\n"
        f"ltv_missing_rate,{summary[5]}\n"
    )
    QUALITY_PATH.write_text(quality, encoding="utf-8")
    file_names = "<br>".join(path.name for path in files)
    report = f"""# Fannie Mae Data Preparation Report / Fannie Mae 数据准备报告

## Source files / 数据源文件

{file_names}

These are official combined monthly files: each row contains both origination attributes and a monthly performance record. 这些是官方联合月度文件：每行同时包含起源属性与一条月度表现记录。

## Point-in-time definition / 时点定义

- **Feature window / 特征窗：** months 1-6 after origination. 起源后第 1-6 个月。
- **Outcome window / 结果窗：** months 7-18 after origination. 起源后第 7-18 个月。
- **Bad definition / 坏样本定义：** any 90+ DPD event (`current_delinquency_status >= 3`) in the outcome window. 结果窗内出现任一 90 天及以上逾期。

## Output quality / 输出质量

| Metric / 指标 | Value / 数值 |
| --- | ---: |
| Eligible PD snapshots / 可用 PD 快照 | {summary[0]:,} |
| 90+ DPD bad rate / 90 天以上逾期坏样本率 | {summary[1]:.2%} |
| Earliest origination / 最早起源日期 | {summary[2]} |
| Latest origination / 最晚起源日期 | {summary[3]} |
| FICO missing rate / FICO 缺失率 | {summary[4]:.2%} |
| LTV missing rate / LTV 缺失率 | {summary[5]:.2%} |

## Data integrity / 数据完整性

The pipeline reads only required columns from the headerless, pipe-delimited source and retains only months 1-18. It therefore does not load the full monthly history into memory and does not use future repayment performance as model input. 流水线只读取无表头竖线分隔源文件所需的字段，并仅保留第 1-18 个月；因此不会将完整月度历史载入内存，也不会将未来还款表现用作模型输入。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    """Create a primary-project PD snapshot from selected Fannie cohorts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=Path, nargs="+", help="Combined monthly CSV files; defaults to 2021Q1-2024Q1.")
    args = parser.parse_args()
    files = args.files or [RAW_DIR / f"{year}Q1.csv" for year in range(2021, 2025)]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing source files: {missing}")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(DATABASE_PATH))
    temp_directory = str(PROCESSED_DIR.resolve()).replace("\\", "/")
    connection.execute(f"SET temp_directory='{temp_directory}'")
    try:
        build_snapshot(connection, files)
        write_outputs(connection, files)
        count = connection.execute("SELECT COUNT(*) FROM fannie_pd_snapshot").fetchone()[0]
        print(f"Built {count:,} point-in-time PD snapshots from {len(files)} source files.")
        print(f"Snapshot written to {SNAPSHOT_PATH}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()

# Main Data Source and Actual File Structure / 主数据源与实际文件结构

## Chosen source / 选定数据源

The main project uses the official **Fannie Mae Single-Family Loan Performance Data**. The local project currently uses four downloaded combined files: `2021Q1.csv`, `2022Q1.csv`, `2023Q1.csv`, and `2024Q1.csv`.

主项目使用官方 **Fannie Mae Single-Family Loan Performance Data**。当前本地项目使用四份已下载的联合文件：`2021Q1.csv`、`2022Q1.csv`、`2023Q1.csv` 与 `2024Q1.csv`。

## What a Q1 file means / Q1 文件不等于三个月数据

`2021Q1.csv` does **not** mean the file contains only three months of observations. In the combined data format, a loan cohort is represented by repeated monthly performance records. One loan identifier can appear once per reporting month, together with origination attributes that remain mostly unchanged. The `Q1` file name identifies the cohort/release grouping, not a maximum of three monthly records per loan.

`2021Q1.csv` **不**表示文件中每笔贷款只有三个月数据。在联合数据格式中，同一笔贷款会随月度表现重复出现；每个报告月都有一行记录，同时带有基本不变的起源属性。文件名中的 `Q1` 是批次/发布分组，不是每笔贷款最多三个月的限制。

Therefore, the four files are not “16 months of data”. They contain many loans, and each eligible loan has at least 18 monthly records after its origination date. The pipeline deliberately retains only months 1-18 for the current PD use case.

因此，四份文件不是“16 个月的数据”。它们包含很多笔贷款，每笔符合条件的贷款在起源后至少有 18 条月度记录。当前 PD 流水线刻意只保留第 1-18 个月。

## Raw-file grain / 原始文件粒度

The raw files are headerless, pipe-delimited files with 113 fields. The practical grain is:

> One row = one loan identifier + one monthly reporting period.

> 一行 = 一笔贷款标识 + 一个报告月份。

Examples of fields used by the project:

| Field / 字段 | Meaning / 含义 | Role / 用途 |
| --- | --- | --- |
| `loan_id` | Loan identifier / 贷款标识 | Groups repeated monthly records / 串联同一笔贷款的多个月记录 |
| `reporting_month` | Reporting month / 报告月份 | Determines month-on-book / 计算贷款已存续月数 |
| `origination_date` | Origination month / 起源月份 | Starting point of the 18-month timeline / 18 个月时间轴起点 |
| `borrower_credit_score` | FICO credit score / FICO 信用分 | Borrower credit feature / 借款人信用特征 |
| `original_ltv`, `original_dti` | Original LTV and DTI / 原始 LTV、DTI | Affordability and collateral features / 偿付与抵押特征 |
| `current_actual_upb` | Current unpaid balance / 当前未偿本金 | Monthly balance behaviour / 月度余额行为 |
| `current_delinquency_status` | Delinquency status in months / 逾期状态（月） | Delinquency history and 90+ DPD label / 逾期特征与 90+ DPD 标签 |

## One illustrative loan timeline / 一笔贷款的示意时间线

For a loan originated in March 2024, the project creates the following structure:

| Loan month / 贷款月数 | Calendar month / 日历月份 | Project use / 项目用途 |
| --- | --- | --- |
| 1-6 | Apr-Sep 2024 | Features: FICO, LTV, DTI, early balance movement, early delinquency, modification history / 特征 |
| 7-18 | Oct 2024-Sep 2025 | Outcome only: whether any month reaches 90+ DPD / 只用于标签 |

The current processed data actually contain loans originated from `2020-12-01` to `2024-03-01`. The latest snapshot has an observation date of `2024-09-01`, which is six months after a March 2024 origination. It exists only because the downloaded records include the following twelve monthly performance observations required for its label.

当前处理后的数据实际包含 `2020-12-01` 至 `2024-03-01` 起源的贷款。最新快照的观察日为 `2024-09-01`，即 2024 年 3 月起源后第六个月；它能被保留，正是因为下载数据中还有构造标签所需的后续 12 个月表现记录。

## Eligibility rule / 样本保留规则

The pipeline keeps a loan only when it has:

1. At least six records in months 1-6; and
2. At least twelve records in months 7-18.

流水线只保留同时满足以下条件的贷款：

1. 第 1-6 个月至少有 6 条记录；且
2. 第 7-18 个月至少有 12 条记录。

This is why no loan in the modelling dataset has an incomplete 18-month construction window.

因此，进入建模数据集的贷款不会存在 18 个月构造窗口不完整的问题。

## Current output / 当前输出

The raw monthly panel is converted into one point-in-time modelling snapshot per loan. Current output: `2,015,703` eligible loan snapshots, with a 90+ DPD bad-sample rate of `0.61%`.

原始月度面板最终被压缩为“每笔贷款一条、在第 6 个月观察”的建模快照。当前输出为 `2,015,703` 条可用贷款快照，90+ DPD 坏样本率为 `0.61%`。

## Data-use boundary / 数据使用边界

Fannie Mae data are governed by its terms. This project retains source files locally and does not redistribute them. The project is a historical risk-modelling simulation, not a live lending system.

Fannie Mae 数据受官方条款约束。项目仅在本地保留源文件，不再分发；本项目是历史风险建模模拟，不是线上放贷系统。

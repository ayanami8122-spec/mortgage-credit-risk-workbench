"""Streamlit workbench for the Fannie Mae credit-risk project outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
TARGET = "bad_12m_90dpd"


st.set_page_config(
    page_title="Credit Risk Workbench",
    page_icon="CR",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_csv(filename: str) -> pd.DataFrame:
    """Load a generated artefact without touching raw source files."""
    path = PROCESSED / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing generated output: {path}")
    return pd.read_csv(path, low_memory=False)


def fmt_pct(value: float) -> str:
    return f"{float(value):.2%}"


def fmt_money(value: float) -> str:
    return f"{float(value) / 1_000_000:,.1f}M"


def strategy_summary(scores: pd.DataFrame, pass_max_pd: float, review_max_pd: float, lgd: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate an interactive OOT strategy scenario using existing model scores."""
    assigned = scores[["xgboost_pd", "original_upb", TARGET]].copy()
    assigned["Action / 动作"] = "Reject / 拒绝"
    assigned.loc[assigned["xgboost_pd"] <= review_max_pd, "Action / 动作"] = "Manual review / 人工审核"
    assigned.loc[assigned["xgboost_pd"] <= pass_max_pd, "Action / 动作"] = "Pass / 通过"
    assigned["Expected loss / 预期损失"] = assigned["xgboost_pd"] * assigned["original_upb"].fillna(0) * lgd

    detail = (
        assigned.groupby("Action / 动作", observed=True)
        .agg(
            loan_count=(TARGET, "size"),
            observed_bad_rate=(TARGET, "mean"),
            average_pd=("xgboost_pd", "mean"),
            exposure=("original_upb", "sum"),
            expected_loss=("Expected loss / 预期损失", "sum"),
        )
        .reindex(["Pass / 通过", "Manual review / 人工审核", "Reject / 拒绝"], fill_value=0)
        .reset_index()
    )
    detail["Action rate / 动作占比"] = detail["loan_count"] / len(assigned)
    accepted = detail[detail["Action / 动作"].isin(["Pass / 通过", "Manual review / 人工审核"])]
    rejected = detail[detail["Action / 动作"] == "Reject / 拒绝"]
    overview = pd.DataFrame(
        [
            {
                "Non-reject rate / 非拒绝率": accepted["Action rate / 动作占比"].sum(),
                "Pass bad rate / 通过客群坏样本率": detail.loc[detail["Action / 动作"] == "Pass / 通过", "observed_bad_rate"].iloc[0],
                "Booked EL / 放款后预期损失": accepted["expected_loss"].sum(),
                "EL avoided / 拒绝规避损失": rejected["expected_loss"].sum(),
            }
        ]
    )
    return detail, overview


def render_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def render_overview() -> None:
    quality = load_csv("fannie_pipeline_quality.csv").set_index("metric")["value"]
    performance = load_csv("oot_model_performance.csv")
    monitoring = load_csv("oot_monitoring_summary.csv")
    psi = load_csv("oot_score_stability.csv").iloc[0]["value"]
    xgb = performance.loc[performance["model"].str.startswith("XGBoost")].iloc[0]

    render_header("Credit Risk Workbench / 信贷风险工作台", "Fannie Mae mortgage PD, decision strategy and OOT monitoring")
    st.caption("官方 Fannie Mae 数据 | 2021Q1-2024Q1 | 特征窗：第 1-6 月 | 标签窗：第 7-18 月 90+ DPD")
    metrics = st.columns(4)
    metrics[0].metric("Eligible snapshots / 可用快照", f"{int(float(quality['loan_count'])):,}")
    metrics[1].metric("90+ DPD rate / 坏样本率", fmt_pct(quality["bad_rate"]))
    metrics[2].metric("XGBoost OOT AUC", f"{xgb['auc']:.3f}")
    metrics[3].metric("Train-to-OOT PSI", f"{psi:.3f}", "Review / 需复核" if psi >= 0.10 else "Stable / 稳定")

    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("OOT cohort risk / 时间外批次风险")
        chart = monitoring.set_index("cohort")[["bad_rate", "average_score"]].rename(
            columns={"bad_rate": "Bad rate / 坏样本率", "average_score": "Average PD / 平均 PD"}
        )
        st.line_chart(chart, height=290)
    with right:
        st.subheader("Current readout / 当前结论")
        st.info(
            "XGBoost outperforms the transparent logistic baseline on the held-out period. "
            "The train-to-OOT PSI is material, so model stability needs review before any production claim.\n\n"
            "XGBoost 在时间外样本上优于逻辑回归，但训练到 OOT 的 PSI 较高，上线前仍需复核稳定性。"
        )
        monitoring_display = monitoring.rename(
                columns={
                    "cohort": "Cohort / 批次",
                    "sample_count": "Sample count / 样本量",
                    "bad_rate": "Bad rate / 坏样本率",
                    "average_score": "Average PD / 平均 PD",
                    "score_psi_vs_baseline": "PSI vs baseline / 相对基准 PSI",
                    "psi_status": "Status / 状态",
                }
            )
        for column in ["Bad rate / 坏样本率", "Average PD / 平均 PD"]:
            monitoring_display[column] = monitoring_display[column].map(fmt_pct)
        monitoring_display["PSI vs baseline / 相对基准 PSI"] = monitoring_display["PSI vs baseline / 相对基准 PSI"].map("{:.3f}".format)
        st.dataframe(
            monitoring_display,
            hide_index=True,
            width="stretch",
        )


def render_models() -> None:
    performance_file = "champion_challenger_performance.csv" if (PROCESSED / "champion_challenger_performance.csv").exists() else "oot_model_performance.csv"
    performance = load_csv(performance_file)
    cohort = load_csv("oot_cohort_performance.csv")
    importance = load_csv("xgboost_feature_importance.csv").head(15)
    render_header("Model Validation / 模型验证", "Chronological out-of-time validation, not random holdout")

    st.subheader("Champion-challenger results / 冠军挑战者结果")
    st.dataframe(
        performance.rename(
            columns={"model": "Model / 模型", "auc": "AUC", "ks": "KS", "pr_auc": "PR-AUC", "brier_score": "Brier score / Brier 分数"}
        ),
        hide_index=True,
        width="stretch",
        column_config={column: st.column_config.NumberColumn(format="%.3f") for column in ["AUC", "KS", "PR-AUC", "Brier score / Brier 分数"]},
    )
    first, second = st.columns(2)
    with first:
        st.image(FIGURES / "oot_roc_and_calibration.png", caption="OOT ROC and calibration / 时间外 ROC 与校准", width="stretch")
    with second:
        st.image(FIGURES / "xgboost_feature_importance.png", caption="XGBoost feature importance / XGBoost 特征重要性", width="stretch")

    st.subheader("Later OOT cohorts / 后续时间外批次")
    cohort_display = cohort.rename(
            columns={"cohort": "Cohort / 批次", "sample_count": "Sample count / 样本量", "bad_rate": "Bad rate / 坏样本率", "auc": "AUC", "ks": "KS", "pr_auc": "PR-AUC", "brier_score": "Brier score / Brier 分数"}
        )
    cohort_display["Bad rate / 坏样本率"] = cohort_display["Bad rate / 坏样本率"].map(fmt_pct)
    st.dataframe(
        cohort_display,
        hide_index=True,
        width="stretch",
        column_config={"AUC": st.column_config.NumberColumn(format="%.3f"), "KS": st.column_config.NumberColumn(format="%.3f")},
    )
    with st.expander("Feature importance table / 特征重要性明细"):
        st.dataframe(importance, hide_index=True, width="stretch")

    if (FIGURES / "lightgbm_oot_roc_and_calibration.png").exists():
        st.subheader("LightGBM challenger / LightGBM 挑战者")
        st.caption("LightGBM uses the same OOT split as the other models. Strategy simulation continues to use the slightly stronger XGBoost candidate. LightGBM 使用相同 OOT 切分；策略模拟仍使用排序表现略高的 XGBoost 候选。")
        first, second = st.columns(2)
        with first:
            st.image(FIGURES / "lightgbm_oot_roc_and_calibration.png", caption="LightGBM OOT ROC and calibration / LightGBM 时间外 ROC 与校准", width="stretch")
        with second:
            st.image(FIGURES / "lightgbm_feature_importance.png", caption="LightGBM feature importance / LightGBM 特征重要性", width="stretch")

    if (PROCESSED / "calibration_validation_performance.csv").exists():
        calibration = load_csv("calibration_validation_performance.csv")
        st.subheader("Probability calibration study / 概率校准研究")
        st.caption(
            "The 2023Q3 cohort was reserved for calibration and final evidence remains 2024Q2-Q3. "
            "Isotonic calibration did not improve final OOT Brier or ECE, so the current strategy keeps raw XGBoost PD. "
            "2023Q3 被保留为校准批次，最终证据仍来自 2024Q2-Q3。Isotonic 校准没有改善最终 OOT 的 Brier 或 ECE，因此当前策略保留原始 XGBoost PD。"
        )
        calibration_display = calibration.rename(
            columns={"score_version": "Score version / 分数版本", "auc": "AUC", "ks": "KS", "pr_auc": "PR-AUC", "brier_score": "Brier", "ece": "ECE"}
        )
        st.dataframe(
            calibration_display,
            hide_index=True,
            width="stretch",
            column_config={column: st.column_config.NumberColumn(format="%.5f") for column in ["AUC", "KS", "PR-AUC", "Brier", "ECE"]},
        )
        st.image(FIGURES / "xgboost_calibration_comparison.png", caption="Final OOT probability calibration / 最终 OOT 概率校准", width="stretch")


def render_strategy() -> None:
    scores = load_csv("oot_test_scores.csv")
    render_header("Strategy Simulator / 准入策略模拟", "Interactive scenario analysis over the fixed out-of-time score population")
    st.caption("策略模拟使用当前排序表现略高的 XGBoost 候选分数；人工审核贷款暂按最终放款计入 Booked EL；LGD 为情景假设，不是实际回收结果。")

    controls = st.columns([1, 1, 1, 1.3])
    pass_max = controls[0].slider("Pass max PD / 通过阈值", min_value=0.01, max_value=0.50, value=0.12, step=0.01)
    review_max = controls[1].slider("Review max PD / 审核上限", min_value=0.02, max_value=0.70, value=0.30, step=0.01)
    lgd = controls[2].slider("LGD assumption / LGD 假设", min_value=0.10, max_value=0.80, value=0.40, step=0.05)
    controls[3].caption("These controls do not retrain the model. They reallocate an unchanged OOT score population.\n\n这些参数不会重新训练模型，只会重分配固定的时间外评分人群。")

    if pass_max >= review_max:
        st.error("Pass threshold must be below review upper limit. 通过阈值必须低于审核上限。")
        return
    detail, overview = strategy_summary(scores, pass_max, review_max, lgd)
    values = overview.iloc[0]
    metrics = st.columns(4)
    metrics[0].metric("Non-reject / 非拒绝率", fmt_pct(values["Non-reject rate / 非拒绝率"]))
    metrics[1].metric("Pass bad rate / 通过坏样本率", fmt_pct(values["Pass bad rate / 通过客群坏样本率"]))
    metrics[2].metric("Booked EL / 放款后预期损失", fmt_money(values["Booked EL / 放款后预期损失"]))
    metrics[3].metric("EL avoided / 拒绝规避损失", fmt_money(values["EL avoided / 拒绝规避损失"]))

    st.subheader("Action allocation / 动作分配")
    display = detail.rename(
        columns={
            "loan_count": "Loan count / 贷款数",
            "observed_bad_rate": "Observed bad rate / 观察坏样本率",
            "average_pd": "Average PD / 平均 PD",
            "exposure": "Exposure / 暴露金额",
            "expected_loss": "Expected loss / 预期损失",
        }
    )
    for column in ["Action rate / 动作占比", "Observed bad rate / 观察坏样本率", "Average PD / 平均 PD"]:
        display[column] = display[column].map(fmt_pct)
    for column in ["Exposure / 暴露金额", "Expected loss / 预期损失"]:
        display[column] = display[column].map("${:,.0f}".format)
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
    )
    chart = detail.set_index("Action / 动作")[["Action rate / 动作占比", "observed_bad_rate"]].rename(
        columns={"observed_bad_rate": "Observed bad rate / 观察坏样本率"}
    )
    st.bar_chart(chart, height=260)
    st.image(FIGURES / "strategy_tradeoff.png", caption="Configured strategy comparison / 已配置策略方案对比", width="stretch")


def render_monitoring() -> None:
    monitoring = load_csv("oot_monitoring_summary.csv")
    render_header("Portfolio Monitoring / 组合监控", "Risk movement and score distribution stability by OOT cohort")
    baseline = monitoring.iloc[0]["cohort"]
    latest = monitoring.iloc[-1]
    metrics = st.columns(4)
    metrics[0].metric("Baseline / 基准批次", baseline)
    metrics[1].metric("Latest cohort / 最新批次", latest["cohort"])
    metrics[2].metric("Latest bad rate / 最新坏样本率", fmt_pct(latest["bad_rate"]))
    metrics[3].metric("Latest PSI / 最新 PSI", f"{latest['score_psi_vs_baseline']:.3f}", latest["psi_status"])
    st.image(FIGURES / "oot_monitoring_trends.png", caption="OOT risk and PSI monitoring / 时间外风险与 PSI 监控", width="stretch")
    st.subheader("Monitoring table / 监控明细")
    monitoring_display = monitoring.copy()
    monitoring_display["bad_rate"] = monitoring_display["bad_rate"].map(fmt_pct)
    monitoring_display["average_score"] = monitoring_display["average_score"].map(fmt_pct)
    monitoring_display["score_psi_vs_baseline"] = monitoring_display["score_psi_vs_baseline"].map("{:.3f}".format)
    st.dataframe(
        monitoring_display.rename(
            columns={
                "cohort": "Cohort / 批次",
                "sample_count": "Sample count / 样本量",
                "bad_rate": "Bad rate / 坏样本率",
                "average_score": "Average PD / 平均 PD",
                "score_psi_vs_baseline": "PSI vs baseline / 相对基准 PSI",
                "psi_status": "Status / 状态",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.warning(
        "PSI is a distribution-stability signal, not proof that model discrimination failed. "
        "Read it together with bad rate, AUC/KS, data quality and policy changes.\n\n"
        "PSI 是分布稳定性信号，不等同于模型区分能力失效；应结合坏样本率、AUC/KS、数据质量与策略变更解读。"
    )
    if (PROCESSED / "feature_stability_monitoring.csv").exists():
        feature_stability = load_csv("feature_stability_monitoring.csv")
        quality = load_csv("data_quality_monitoring.csv")
        st.subheader("Feature and data-quality monitoring / 特征与数据质量监控")
        st.caption("Feature PSI explains which input distributions moved; rule checks separately identify missing and out-of-domain values. 特征 PSI 用于定位发生变化的输入分布；规则检查则单独识别缺失与越界值。")
        train_to_oot = feature_stability.loc[feature_stability["comparison"].str.startswith("Train vs OOT")].copy()
        train_to_oot["psi"] = train_to_oot["psi"].map("{:.3f}".format)
        train_to_oot["baseline_missing_rate"] = train_to_oot["baseline_missing_rate"].map(fmt_pct)
        train_to_oot["comparison_missing_rate"] = train_to_oot["comparison_missing_rate"].map(fmt_pct)
        st.dataframe(
            train_to_oot.rename(
                columns={
                    "feature": "Feature / 特征",
                    "feature_type": "Type / 类型",
                    "psi": "PSI",
                    "psi_status": "Status / 状态",
                    "baseline_missing_rate": "Train missing / 训练集缺失率",
                    "comparison_missing_rate": "OOT missing / OOT 缺失率",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        first, second = st.columns(2)
        with first:
            st.image(FIGURES / "feature_psi_monitoring.png", caption="Feature PSI / 特征 PSI", width="stretch")
        with second:
            st.image(FIGURES / "data_quality_monitoring.png", caption="Data-quality rules / 数据质量规则", width="stretch")
        with st.expander("Data-quality rule detail / 数据质量规则明细"):
            quality_display = quality.copy()
            for column in ["missing_rate", "invalid_rate"]:
                quality_display[column] = quality_display[column].map(fmt_pct)
            st.dataframe(
                quality_display.rename(
                    columns={
                        "population": "Population / 样本",
                        "feature": "Feature / 特征",
                        "sample_count": "Sample count / 样本量",
                        "missing_rate": "Missing rate / 缺失率",
                        "invalid_rate": "Invalid rate / 越界率",
                        "quality_status": "Status / 状态",
                    }
                ),
                hide_index=True,
                width="stretch",
            )


def main() -> None:
    st.markdown(
        """
        <style>
          [data-testid="stSidebar"] { border-right: 1px solid #283548; }
          [data-testid="stMetric"] { background: #151b26; border: 1px solid #283548; padding: 12px; border-radius: 6px; }
          [data-testid="stMetricLabel"] { color: #aebbd0; }
          [data-testid="stMetricValue"] { color: #f7f9fc; }
          h1, h2, h3 { letter-spacing: 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.header("Risk Workbench")
        st.caption("信贷风险评分与策略监控")
        page = st.radio(
            "Workspace / 工作区",
            ["Portfolio overview / 组合概览", "Model validation / 模型验证", "Strategy simulator / 策略模拟", "Monitoring / 监控"],
            label_visibility="collapsed",
        )
        st.divider()
        if st.button("Refresh generated outputs / 刷新产出"):
            st.cache_data.clear()
            st.rerun()
        st.caption("Source: Fannie Mae 2021Q1-2024Q1\n\n数据不会在页面加载时重新建模。")

    try:
        if page.startswith("Portfolio"):
            render_overview()
        elif page.startswith("Model"):
            render_models()
        elif page.startswith("Strategy"):
            render_strategy()
        else:
            render_monitoring()
    except FileNotFoundError as error:
        st.error("Generated outputs are missing. Run the data/model pipeline first.\n\n缺少已生成结果，请先运行数据与模型流水线。")
        st.code(str(error), language="text")


if __name__ == "__main__":
    main()

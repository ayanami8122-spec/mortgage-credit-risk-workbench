"""Focused tests for point-in-time label construction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fannie_mae_pipeline import build_pd_snapshot  # noqa: E402
from monitoring import build_monitoring_table  # noqa: E402
from oot_risk_modelling import population_stability_index, train_models  # noqa: E402
from strategy_simulation import apply_strategy, scenario_overview, summarize_strategy  # noqa: E402


class FanniePipelineTests(unittest.TestCase):
    def test_snapshot_uses_future_window_only_for_bad_flag(self) -> None:
        origination = pd.DataFrame(
            {
                "loan_id": ["A", "B"],
                "origination_date": pd.to_datetime(["2019-01-01", "2019-01-01"]),
                "original_upb": [200_000, 220_000],
                "original_ltv": [80, 75],
                "borrower_credit_score": [720, 690],
            }
        )
        records = []
        for loan_id in ["A", "B"]:
            for month in range(1, 19):
                records.append(
                    {
                        "loan_id": loan_id,
                        "reporting_month": pd.Timestamp("2019-01-01") + pd.DateOffset(months=month),
                        "current_actual_upb": 200_000 - month * 500,
                        "current_delinquency_status": 0 if loan_id == "A" or month <= 6 else 3,
                        "modification_flag": "N",
                    }
                )
        snapshot = build_pd_snapshot(origination, pd.DataFrame(records), observation_months=6, outcome_months=12)
        bad_flags = snapshot.set_index("loan_id")["bad_12m_90dpd"].to_dict()
        self.assertEqual(bad_flags, {"A": 0, "B": 1})
        self.assertEqual(snapshot.set_index("loan_id").loc["B", "hist_max_dpd"], 0)

    def test_psi_is_zero_for_identical_scores(self) -> None:
        scores = pd.Series([0.05, 0.10, 0.20, 0.30, 0.50]).to_numpy()
        self.assertAlmostEqual(population_stability_index(scores, scores), 0.0, places=8)

    def test_strategy_assigns_three_operational_actions(self) -> None:
        scores = pd.DataFrame(
            {"xgboost_pd": [0.05, 0.20, 0.50], "original_upb": [100_000, 100_000, 100_000]}
        )
        actions = apply_strategy(scores, pass_max_pd=0.10, review_max_pd=0.30, lgd=0.40)
        self.assertEqual(actions["action"].tolist(), ["Pass / 通过", "Manual review / 人工审核", "Reject / 拒绝"])
        self.assertAlmostEqual(actions.loc[0, "expected_loss"], 2_000)

    def test_strategy_overview_excludes_rejected_risk_from_booked_loss(self) -> None:
        scores = pd.DataFrame(
            {
                "xgboost_pd": [0.05, 0.20, 0.50],
                "original_upb": [100_000, 100_000, 100_000],
                "bad_12m_90dpd": [0, 0, 1],
            }
        )
        scenario = pd.Series(
            {"scenario": "Test", "pass_max_pd": 0.10, "review_max_pd": 0.30, "lgd_assumption": 0.40}
        )
        overview = scenario_overview(summarize_strategy(scores, scenario)).iloc[0]
        self.assertAlmostEqual(overview["provisional_booked_expected_loss"], 10_000)
        self.assertAlmostEqual(overview["expected_loss_avoided_by_reject"], 20_000)

    def test_monitoring_compares_later_cohorts_with_baseline(self) -> None:
        scores = pd.DataFrame(
            {
                "observation_date": ["2020-01-01", "2020-01-01", "2020-04-01", "2020-04-01"],
                "xgboost_pd": [0.10, 0.20, 0.20, 0.40],
                "bad_12m_90dpd": [0, 1, 0, 1],
            }
        )
        summary, baseline, _ = build_monitoring_table(scores)
        self.assertEqual(baseline, "2020Q1")
        self.assertEqual(len(summary), 2)

    def test_champion_challenger_models_produce_probabilities(self) -> None:
        sample_size = 240
        features = pd.DataFrame(
            {
                "loan_id": [f"L{i}" for i in range(sample_size)],
                "origination_date": pd.date_range("2019-01-01", periods=sample_size, freq="D"),
                "observation_date": pd.date_range("2019-07-01", periods=sample_size, freq="D"),
                "original_upb": [200_000 + 500 * i for i in range(sample_size)],
                "borrower_credit_score": [650 + i % 100 for i in range(sample_size)],
                "hist_max_dpd": [i % 4 for i in range(sample_size)],
                "hist_30dpd_months": [i % 3 for i in range(sample_size)],
                "loan_purpose": ["P" if i % 2 else "R" for i in range(sample_size)],
                "bad_12m_90dpd": [i % 2 for i in range(sample_size)],
            }
        )
        models, metrics, scored, _, _ = train_models(features.iloc[:180], features.iloc[180:])
        self.assertEqual(set(models), {"logistic", "xgboost"})
        self.assertEqual(len(metrics), 2)
        self.assertTrue(scored["xgboost_pd"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()

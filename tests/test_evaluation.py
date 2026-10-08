"""Hand-calculated checks for manual evaluation; no library metric oracle."""

import unittest

import numpy as np
import pandas as pd

from src.afib.evaluation import (
    assert_patient_disjoint, auroc, average_precision, binary_metrics,
    choose_sensitivity_threshold, majority_baseline, quality_subgroup_metrics,
    source_quality_groups, summarise_paired_drops, validate_scenario_repeats,
    validate_experiment_arms, MODEL_NAMES, FEATURE_SETS, TRAINING_ARMS,
)


class EvaluationTests(unittest.TestCase):
    def test_confusion_and_brier_hand_example(self):
        result = binary_metrics([1, 1, 0, 0], [.9, .4, .6, .1])
        for name in ("tp", "tn", "fp", "fn"):
            self.assertEqual(result[name], 1)
        for name in ("accuracy", "sensitivity", "specificity", "precision", "npv",
                     "positive_f1", "negative_f1", "macro_f1", "balanced_accuracy"):
            self.assertEqual(result[name], .5)
        self.assertAlmostEqual(result["brier"], .185)
        self.assertEqual(result["auroc"], .75)
        self.assertAlmostEqual(result["average_precision"], (1 + 2 / 3) / 2)

    def test_tied_scores_are_grouped_and_order_independent(self):
        y, score = np.array([1, 0, 1, 0]), np.array([.8, .8, .2, .1])
        self.assertEqual(auroc(y, score), .625)
        self.assertAlmostEqual(average_precision(y, score), .25 + 1 / 3)
        for order in ([3, 2, 1, 0], [1, 0, 3, 2]):
            self.assertEqual(auroc(y[order], score[order]), auroc(y, score))
            self.assertEqual(average_precision(y[order], score[order]), average_precision(y, score))
        self.assertEqual(auroc([1, 0, 0, 0], [.5] * 4), .5)
        self.assertEqual(average_precision([1, 0, 0, 0], [.5] * 4), .25)

    def test_perfect_reverse_and_single_class(self):
        self.assertEqual(auroc([0, 1], [.1, .9]), 1)
        self.assertEqual(auroc([0, 1], [.9, .1]), 0)
        self.assertTrue(np.isnan(auroc([0, 0], [.1, .2])))
        self.assertTrue(np.isnan(average_precision([0, 0], [.1, .2])))
        self.assertEqual(average_precision([1, 1], [.1, .2]), 1)
        result = binary_metrics([0, 0], [.1, .2])
        self.assertTrue(np.isnan(result["sensitivity"]))
        self.assertTrue(np.isnan(result["precision"]))
        self.assertTrue(np.isnan(result["macro_f1"]))
        self.assertEqual(result["npv"], 1)

    def test_threshold_uses_inner_examples_and_is_frozen(self):
        selection = choose_sensitivity_threshold([1, 1, 0, 0], [.9, .7, .8, .2], 1)
        self.assertEqual(selection["threshold"], .7)
        self.assertEqual(selection["inner_specificity"], .5)
        stressed = binary_metrics([1, 1, 0, 0], [.6, .5, .8, .2], selection["threshold"])
        self.assertEqual(stressed["sensitivity"], 0)
        self.assertEqual(selection["threshold"], .7)
        self.assertEqual(choose_sensitivity_threshold([1, 1], [.6, .7])["status"], "unavailable_one_class")
        with self.assertRaises(ValueError):
            choose_sensitivity_threshold([0, 1], [.1, .9], 1.1)

    def test_threshold_specificity_tie_prefers_higher_sensitivity(self):
        selection = choose_sensitivity_threshold([1, 1, 0, 0], [.9, .8, .7, .2], .5)
        self.assertEqual(selection["threshold"], .8)
        self.assertEqual(selection["inner_sensitivity"], 1)
        self.assertEqual(selection["inner_specificity"], 1)

    def test_majority_chosen_from_training_not_test(self):
        result = majority_baseline([0, 0, 0, 1], [1, 1, 1, 0])
        self.assertEqual(result["majority_label"], 0)
        self.assertEqual(result["accuracy"], .25)
        self.assertEqual(result["sensitivity"], 0)
        self.assertEqual(result["training_prevalence"], .25)
        self.assertEqual(majority_baseline([0, 1], [0, 1])["majority_label"], 0)

    def test_patient_separation(self):
        assert_patient_disjoint([1, 1, 2], [3, 3, 4])
        with self.assertRaises(ValueError):
            assert_patient_disjoint([1, 2], [2, 3])
        with self.assertRaises(ValueError):
            assert_patient_disjoint([1, None], [3])

    def test_subgroups_missing_class_and_unknown(self):
        result = quality_subgroup_metrics([1, 0, 0], [.8, .2, .1], ["good", "good", None]).set_index("subgroup")
        self.assertTrue(result.loc["good", "both_classes_present"])
        self.assertEqual(result.loc["good", "auroc"], 1)
        self.assertFalse(result.loc["unknown", "both_classes_present"])
        self.assertTrue(np.isnan(result.loc["unknown", "average_precision"]))
        self.assertTrue(np.isnan(result.loc["unknown", "balanced_accuracy"]))
        self.assertEqual(result.loc["unknown", "specificity"], 1)

    def test_quality_groups_use_only_source_flags(self):
        metadata = pd.DataFrame({"ecg_id": [1, 2], "baseline_drift": [None, "V1"],
                                 "static_noise": [" ", None], "burst_noise": [None, None],
                                 "electrodes_problems": [None, None], "target_afib": [0, 1]})
        original = source_quality_groups(metadata)
        metadata["target_afib"] = [1, 0]
        pd.testing.assert_series_equal(original, source_quality_groups(metadata))
        self.assertEqual(original.to_list(), ["no_recorded_quality_flag", "recorded_quality_flag"])
        with self.assertRaises(ValueError):
            source_quality_groups(pd.concat([metadata, metadata.iloc[:1]]))

    def test_protocol_rejects_uniformly_missing_condition_or_repeat(self):
        from src.afib.preprocessing import SCENARIOS
        rows = [{"scenario": "observed", "repeat": 0}] + [
            {"scenario": scenario, "repeat": repeat}
            for scenario in SCENARIOS if scenario != "observed" for repeat in range(3)]
        validate_scenario_repeats(rows, SCENARIOS)
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            validate_scenario_repeats([row for row in rows if row["scenario"] != "noise_0.20"], SCENARIOS)
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            validate_scenario_repeats([row for row in rows if row["repeat"] != 2], SCENARIOS)
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            validate_scenario_repeats(rows + [{"scenario": "observed", "repeat": 1}], SCENARIOS)

    def test_experiment_requires_every_model_arm_and_fold(self):
        rows = [{"model": model, "feature_set": features, "training_arm": training, "outer_fold": fold}
                for model in MODEL_NAMES for features in FEATURE_SETS
                for training in TRAINING_ARMS for fold in range(10)]
        validate_experiment_arms(rows)
        missing_model = [row for row in rows if row["model"] != "tabm"]
        missing_arm = [row for row in rows if not (row["feature_set"] == "enhanced" and row["training_arm"] == "augmented")]
        missing_fold = rows[:-1]
        for incomplete in (missing_model, missing_arm, missing_fold):
            with self.assertRaisesRegex(ValueError, "model/arm protocol mismatch"):
                validate_experiment_arms(incomplete)

    def test_average_repeats_then_paired_fold_sd(self):
        records = []
        for fold in range(10):
            records.append(dict(outer_fold=fold, scenario="observed", repeat=0, score=.9))
            for repeat, offset in enumerate([.01, -.01, 0]):
                records.append(dict(outer_fold=fold, scenario="noise", repeat=repeat,
                                    score=.9 - .01 * fold + offset))
        effects, summary = summarise_paired_drops(records, "score")
        self.assertEqual(len(effects), 10)
        self.assertEqual(summary.iloc[0].n_folds, 10)
        self.assertAlmostEqual(summary.iloc[0].mean_deterioration, .045)
        self.assertAlmostEqual(summary.iloc[0].sd_deterioration, np.sqrt(9.166666666666667) * .01)
        _, opposite = summarise_paired_drops(records, "score", higher_is_better=False)
        self.assertAlmostEqual(opposite.iloc[0].mean_deterioration, -.045)

    def test_pairing_rejects_missing_folds_duplicates_and_small_runs(self):
        rows = [dict(outer_fold=f, scenario=s, repeat=0, score=.8) for f in range(10)
                for s in ("observed", "stress")]
        with self.assertRaises(ValueError):
            summarise_paired_drops(rows[:-1], "score")
        with self.assertRaises(ValueError):
            summarise_paired_drops(rows + [rows[0]], "score")
        with self.assertRaises(ValueError):
            summarise_paired_drops(rows[:4], "score")
        rows[-1]["score"] = float("nan")
        _, summary = summarise_paired_drops(rows, "score")
        self.assertEqual(summary.iloc[0].n_defined_folds, 9)
        self.assertTrue(np.isnan(summary.iloc[0].mean_deterioration))

    def test_pairing_rejects_partial_repeat_set_with_all_folds_present(self):
        rows = [dict(outer_fold=f, scenario="observed", repeat=0, score=.9) for f in range(10)]
        rows += [dict(outer_fold=f, scenario="stress", repeat=r, score=.8)
                 for f in range(10) for r in range(3)]
        with self.assertRaisesRegex(ValueError, "different repeat sets"):
            summarise_paired_drops(rows[:-1], "score")

    def test_invalid_probability_or_shape_is_rejected(self):
        for labels, scores in [([], []), ([0], [1, 0]), ([2], [.2]), ([0], [np.nan]), ([0], [1.2])]:
            with self.assertRaises(ValueError):
                binary_metrics(labels, scores)


if __name__ == "__main__":
    unittest.main()

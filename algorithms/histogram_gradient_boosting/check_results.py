"""Validate previously saved HGB results and summarise them without retraining."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "reports/histogram_gradient_boosting/formal"
METRICS = ["macro_f1", "sensitivity", "specificity", "precision", "npv",
           "balanced_accuracy", "auroc", "average_precision", "brier"]
STRESSES = ["missing_1", "missing_3", "missing_6", "noise_0.05",
            "noise_0.10", "noise_0.20", "gain_0.05", "gain_0.10",
            "gain_0.20", "combined_3lead_noise0.10"]
EXPECTED = {("observed", 0)} | {(s, r) for s in STRESSES for r in range(3)}


def main():
    records = []
    for arm in "ABCD":
        for fold in range(10):
            location = ROOT / f"{arm}_fold{fold}"
            for name in ["complete.json", "inner_tuning.csv", "metrics.csv", "predictions.csv"]:
                if not (location / name).is_file():
                    raise FileNotFoundError(f"{location}: missing {name}")
            meta = json.loads((location / "complete.json").read_text(encoding="utf-8"))
            if meta.get("arm") != arm or meta.get("outer_fold") != fold or meta.get("repeats") != 3:
                raise ValueError(f"{location}: completion metadata mismatch")
            m = pd.read_csv(location / "metrics.csv")
            p = pd.read_csv(location / "predictions.csv")
            t = pd.read_csv(location / "inner_tuning.csv")
            for df, name in [(m, "metrics"), (p, "predictions"), (t, "inner_tuning")]:
                if not ((df["arm"] == arm) & (df["outer_fold"] == fold)).all():
                    raise ValueError(f"{location}: {name} has mismatched arm/fold")
            pairs = list(zip(m["scenario"], m["repeat"]))
            if len(pairs) != 31 or set(pairs) != EXPECTED or len(set(pairs)) != 31:
                raise ValueError(f"{location}: incomplete or duplicate scenario/repeat rows")
            if len(t) != 12 or set(t["inner_fold"]) != {0, 1, 2}:
                raise ValueError(f"{location}: unexpected tuning coverage")
            n = int(m.loc[m["scenario"].eq("observed"), "n"].iloc[0])
            if n <= 0 or len(p) != 31 * n:
                raise ValueError(f"{location}: prediction count mismatch")
            count = p.groupby(["scenario", "repeat"]).size()
            if set(count.index) != EXPECTED or not (count == n).all():
                raise ValueError(f"{location}: incomplete prediction scenarios")
            if not p["prob_afib"].between(0, 1).all() or p["prob_afib"].isna().any():
                raise ValueError(f"{location}: invalid probabilities")
            if not p["y_true"].isin([0, 1]).all():
                raise ValueError(f"{location}: invalid labels")
            base = p.loc[(p["scenario"] == "observed") & (p["repeat"] == 0),
                         ["ecg_id", "y_true"]].sort_values("ecg_id").reset_index(drop=True)
            if base["ecg_id"].duplicated().any():
                raise ValueError(f"{location}: duplicate ECG IDs")
            for (_, _), chunk in p.groupby(["scenario", "repeat"]):
                target = chunk[["ecg_id", "y_true"]].sort_values("ecg_id").reset_index(drop=True)
                if not target.equals(base):
                    raise ValueError(f"{location}: scenario labels or ECG IDs differ")
            if not all(metric in m for metric in METRICS):
                raise ValueError(f"{location}: missing evaluation columns")
            records.append(m)
            print(f"Verified {arm}_fold{fold}")

    all_metrics = pd.concat(records, ignore_index=True)
    observed = all_metrics.loc[(all_metrics["scenario"] == "observed") &
                               (all_metrics["repeat"] == 0)]
    summary = observed.groupby("arm")[METRICS].agg(["mean", "std"])
    fold_means = all_metrics.groupby(["arm", "outer_fold", "scenario"])["macro_f1"].mean().unstack("scenario")
    stress_drops = pd.DataFrame({f"drop_{s}": fold_means["observed"] - fold_means[s] for s in STRESSES})
    drops = stress_drops.groupby(level="arm").agg(["mean", "std"])
    wide_observed = observed.pivot(index="outer_fold", columns="arm", values="macro_f1")
    comparisons = {"B_minus_A": ("B", "A"), "C_minus_A": ("C", "A"),
                   "D_minus_B": ("D", "B"), "D_minus_C": ("D", "C")}
    differences = pd.DataFrame({name: wide_observed[a] - wide_observed[b]
                                for name, (a, b) in comparisons.items()})
    paired = differences.agg(["mean", "std"]).T
    output = PROJECT / "reports/histogram_gradient_boosting"
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "final_observed_metrics.csv")
    drops.to_csv(output / "final_stress_drops.csv")
    paired.to_csv(output / "final_paired_method_differences.csv")
    all_metrics.to_csv(output / "final_all_metrics.csv", index=False)
    print("\n=== OBSERVED MACRO-F1 (mean +/- sample SD) ===")
    print(summary["macro_f1"].round(4).to_string())
    print("\nAll 40 folds and 31 scenarios per fold verified; final summaries saved.")


if __name__ == "__main__":
    main()

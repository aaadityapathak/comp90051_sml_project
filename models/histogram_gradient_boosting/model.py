"""Fixed HGB estimator shared by the original and retuned depth searches."""

from sklearn.ensemble import HistGradientBoostingClassifier


def create_model(max_depth=2):
    """Vary tree depth only; keep the remaining recorded settings unchanged."""
    # Early stopping is disabled so every candidate receives all 150 iterations.
    return HistGradientBoostingClassifier(
        max_depth=max_depth,
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=16,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42
    )

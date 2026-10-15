from sklearn.ensemble import HistGradientBoostingClassifier

def create_model(max_depth=2):
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

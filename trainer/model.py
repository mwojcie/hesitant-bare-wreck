from sklearn.ensemble import GradientBoostingClassifier


def get_classifier():
    return GradientBoostingClassifier(random_state=42, max_depth=10)

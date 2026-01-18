from sklearn.base import BaseEstimator, TransformerMixin

class CardinalityReducer(BaseEstimator, TransformerMixin):
    """Reduce categorical cardinality by grouping rare categories as OTHER."""
    def __init__(self, columns, min_count=10):
        self.columns = columns
        self.min_count = min_count
        self.category_maps_ = {}

    def fit(self, X, y=None):
        X_copy = X.copy()
        for col in self.columns:
            if col in X_copy.columns:
                freq = X_copy[col].value_counts()
                keep = freq[freq >= self.min_count].index.tolist()
                self.category_maps_[col] = keep
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            if col in X_copy.columns and col in self.category_maps_:
                keep = self.category_maps_[col]
                X_copy[col] = X_copy[col].where(X_copy[col].isin(keep), "OTHER")
        return X_copy

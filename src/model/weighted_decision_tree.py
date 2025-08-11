import numpy as np
from utils.voting_split_manager import FNWeightedSplitManager

class TreeNode:
    def __init__(self, is_leaf, prediction=None, feature=None, threshold=None, left=None, right=None):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

class WeightedDecisionTree:
    def __init__(self, criteria_list, max_depth=3, min_samples_split=5, positive_label=1, epsilon=1e-5):
        self.criteria_list = criteria_list
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.positive_label = positive_label
        self.epsilon = epsilon
        self.root = None

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        # شرط توقف
        if (depth >= self.max_depth
            or len(np.unique(y)) == 1
            or X.shape[0] < self.min_samples_split):
            # برگ
            values, counts = np.unique(y, return_counts=True)
            prediction = values[np.argmax(counts)]
            return TreeNode(is_leaf=True, prediction=prediction)

        # ساخت splitها: لیست (feature_idx, threshold, y_left, y_right)
        splits = []
        for feature in range(X.shape[1]):
            values = np.unique(X[:, feature])
            for i in range(1, len(values)):
                thresh = (values[i-1] + values[i]) / 2
                mask_left = X[:, feature] <= thresh
                mask_right = X[:, feature] > thresh
                if np.sum(mask_left) == 0 or np.sum(mask_right) == 0:
                    continue
                y_left = y[mask_left]
                y_right = y[mask_right]
                splits.append((feature, thresh, y_left, y_right))

        if not splits:
            # اگر split معتبری نبود، گره برگ
            values, counts = np.unique(y, return_counts=True)
            prediction = values[np.argmax(counts)]
            return TreeNode(is_leaf=True, prediction=prediction)

        # رأی‌گیری وزنی بین همه معیارها (FNWeightedSplitManager)
        manager = FNWeightedSplitManager(
            self.criteria_list,
            positive_label=self.positive_label,
            epsilon=self.epsilon
        )
        best_split = manager.evaluate_all_splits(splits)

        if best_split is None:
            # اگر split انتخاب نشد، گره برگ
            values, counts = np.unique(y, return_counts=True)
            prediction = values[np.argmax(counts)]
            return TreeNode(is_leaf=True, prediction=prediction)

        feature = best_split['feature']
        threshold = best_split['threshold']
        mask_left = X[:, feature] <= threshold
        mask_right = X[:, feature] > threshold
        left_child = self._build_tree(X[mask_left], y[mask_left], depth+1)
        right_child = self._build_tree(X[mask_right], y[mask_right], depth+1)
        return TreeNode(
            is_leaf=False,
            feature=feature,
            threshold=threshold,
            left=left_child,
            right=right_child
        )

    def predict_single(self, x, node):
        if node.is_leaf:
            return node.prediction
        if x[node.feature] <= node.threshold:
            return self.predict_single(x, node.left)
        else:
            return self.predict_single(x, node.right)

    def predict(self, X):
        X = np.array(X)
        return np.array([self.predict_single(x, self.root) for x in X])

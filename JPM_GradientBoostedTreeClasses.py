import numpy as np
# Decision Tree Class

# Used as a week learned within the gradient boosting ensemble (builds iteratively which is why there is a max depth of 1)
# High bias but low variance, 
# Ideal for boosting methods that aim to iteratively reduce error by focusing on hard-to-predict instances.
class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth
        self.tree = None

    # Each node is a decision point for traversing through the decision tree
    # Follows decision tree logic
    class Node:
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
            # feature_Index and threshold are determining factors for where the split happens
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    # Recursive binary splitting
    def _build_tree(self, X, y, depth):
        # Continues to split until the maximum specified depth is reached
        num_samples, num_features = X.shape
        # Stopping condition: if the current depth exceeds the max depth or the dataset cannot be split further
        if depth >= self.max_depth or num_samples <= 1:
            leaf_value = self._calculate_leaf_value(y)
            # Create a leaf node with the calculated value
            return self.Node(value=leaf_value)

        # Finds the best features by iterating through and selecting lowest MSE
        best_feature, best_threshold = self._find_best_split(X, y, num_samples, num_features)
        if best_feature is None:
            # If no split can improve the outcome, create a leaf node
            return self.Node(value=self._calculate_leaf_value(y))
        
        # Split the dataset and recursively build left and right subtrees
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left_subtree = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_subtree = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return self.Node(feature_index=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def _calculate_leaf_value(self, y):
        # The leaf value can be the mean of the target values, calculates that mean value for the leaf node
        return np.mean(y)

    def _find_best_split(self, X, y, num_samples, num_features):
        # Finds the best feature and threshold to split on based on the lowest MSE
        best_feature, best_threshold = None, None
        best_mse = np.inf
        for feature_index in range(num_features):
            thresholds = np.unique(X[:, feature_index])
            # Splits the dataset and calculates the MSE for this split
            for threshold in thresholds:
                left_idxs, right_idxs = self._split(X[:, feature_index], threshold)
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue
                mse = self._calculate_mse(y[left_idxs], y[right_idxs])
                # Update split if current mse is better
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature_index
                    best_threshold = threshold
        return best_feature, best_threshold

    # Splits dataset into left/right based on threshold of given feature
    def _split(self, feature_values, threshold):
        left_idxs = np.where(feature_values <= threshold)[0]
        right_idxs = np.where(feature_values > threshold)[0]
        return left_idxs, right_idxs

    def _calculate_mse(self, left_y, right_y):
        # Calculate the MSE of the left and right splits by weighted averages of variance
        total_left_mse = np.var(left_y) * len(left_y) if len(left_y) > 0 else 0
        total_right_mse = np.var(right_y) * len(right_y) if len(right_y) > 0 else 0
        total_mse = (total_left_mse + total_right_mse) / (len(left_y) + len(right_y))
        return total_mse
    
    def predict(self, X):
        # Predictions array to store predictions for each sample in X
        predictions = np.array([self._traverse_tree(x, self.tree) for x in X])
        return predictions

    def _traverse_tree(self, x, node):
        # Recursive method to traverse the tree for a single sample 'x' until a leaf node is reached
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
        
# Boosted Gradient Class
# Gradient Boost Class

class GradientBoostAll:
    def __init__(self, n_estimators: int = 25, max_depth: int = 1, learning_rate: int =.1):
        self.max_depth = max_depth # Max depth of the trees
        self.n_estimators = n_estimators # Number of trees
        self.learning_rate = learning_rate # Learning rate, step size for parameter update
        self.trees = [] # List of our trees
    
    def fit(self, X_train, y_train):
        # To start all residuals = y_train
        residuals = np.copy(y_train)
        self.f_hat = 0 
        # Now time to make decision trees
        for i in range(self.n_estimators):
            # Build and Fit Tree to data
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_train, residuals)
            # Save our tree
            self.trees.append(tree)
            # Make prediction
            f_hat_b = tree.predict(X_train)
            # Update f_hat
            self.f_hat += (self.learning_rate*f_hat_b) 
            # Update residuals
            residuals = residuals - (self.learning_rate*f_hat_b)
        return self
    
    def predict(self, X_test):
        y_hat = np.zeros((X_test.shape[0], ))
        for tree in self.trees:
            y_hat += self.learning_rate*tree.predict(X_test)
        return y_hat
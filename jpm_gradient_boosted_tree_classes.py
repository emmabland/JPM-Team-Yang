""" Regression Decision Tree and Boosted Gradient Classes for Boosted Gradient Trees ML Model

DecisionTree: Decision Tree class which builds and utilizes regression trees.
GradientBoostAll: Gradient Boost class which boosts the DecisionTree class
"""
import numpy as np
import pandas as pd

# Decision Tree Class
class DecisionTree:
    """A Decision Tree used for regression

    Instance Attributes:
        max_depth (int): Max amount of branches for a tree (Stopping Criteria)
        tree (DecisionTree.Node): Where tree will be saved, compilation of nodes and their branches
    """
    def __init__(self, max_depth: int =1):
        self.max_depth = max_depth
        self.tree = None

    class Node:
        """A node of a decision tree. 
        Each node is a decision point for traversing through the decision tree,
        or a leaf node for stopping transversal.

        Instance Attributes:
            feature_index (int): A number representing the column/feature we are thresholding on
            threshold (float): The value of our feature_index that a Decision Node splits on
            left (DecisionTree.Node): The next node that branches left from our current 
                decision node (<= Threshold)
            right (DecisionTree.Node): The next node that branches right from our current 
                decision node (> Threshold)
            value: Value that the decision tree predicts if a data point ends its tree transversal 
                at this leaf node (only leaf nodes have a value, and they only have a value)
        """
        def __init__(self, feature_index: int = None, threshold: float = None, left: 'DecisionTree.Node' = None, right: 'DecisionTree.Node' = None, value: float =None):
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits tree to the given data

        Args:
            X (np.ndarray): Input (feature) data you are fitting tree to
            y (np.ndarray): Output (observation) data you are fitting tree to
        """
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> 'DecisionTree.Node':
        """Builds our tree based on the given data through performing Recursive Binary Splitting

        Args:
            X (np.ndarray): Input (features) training data
            y (np.ndarray): Output (observations) training data 
            depth (int): Tracks depth of tree as tree is built through recursively calling

        Returns:
            Node (DecisionTree.Node): Returns decision or leaf node
        """
        # Gets number of rows (samples) and columns (features)
        num_samples, num_features = X.shape
        # Check Stopping Criteria
        if depth >= self.max_depth or num_samples <= 1:
            # If met it marks a leaf node, calculate value and create node
            leaf_value = self._calculate_leaf_value(y)
            return self.Node(value=leaf_value)

        # Find the best split
        best_feature, best_threshold = self._find_best_split(X, y, num_samples, num_features)
        # If no split can improve the outcome, create a leaf node
        if best_feature is None:
            return self.Node(value=self._calculate_leaf_value(y))

        # Split the dataset and recursively build left and right subtrees
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left_subtree = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_subtree = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return self.Node(feature_index=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def _calculate_leaf_value(self, y: np.ndarray) -> float:
        """Calculates the leaf value, which is the value instance attribute of the node class. 
        For a regression Decision Tree, this value is the mean of the output training data sorted 
        to this node.

        Args:
            y (np.ndarray): Output (observations) training data sorted to this leaf node.

        Returns:
            mean: Mean of the output training data sorted to this node. 
        """
        return np.mean(y)

    def _find_best_split(self, X: np.ndarray, y: np.ndarray, num_samples: int, num_features: int) -> int | float:
        """Finds the best feature and threshold to split on based 
        on the lowest mean square error (MSE).

        Args:
            X (np.ndarray): Input (features) training data sorted to this decision node.
            y (np.ndarray): Output (observations) training data sorted to this decision node
            num_samples (int): Number of rows in X
            num_features (int): Number of columns in X

        Returns:
            best_feature (int): The best feature for node to split on
            best_threshold (float): The best threshold of the best_feature for node to split on
        """
        # Initialize variables
        best_feature, best_threshold = None, None
        best_mse = np.inf
        # Loops through every feature (column) of X
        for feature_index in range(num_features):
            # Array of potential thresholds to split data on
            thresholds = np.unique(X[:, feature_index])
            # Calculate the MSE for the split of every potential threshold
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

    def _split(self, feature_values: int, threshold: float) -> np.ndarray:
        """Splits dataset into left_idxs/right_idxs based on threshold of a given feature

        Args:
            feature_values (_type_): The feature the dataset is split on
            threshold (float): Threshold/Value of the feature the dataset is split on

        Returns:
            left_idxs (type): Dataset split that are less than or equal to threshold 
                of chosen feature
            right_idxs (type): Dataset split that are more than threshold of chosen feature
        """
        left_idxs = np.where(feature_values <= threshold)[0]
        right_idxs = np.where(feature_values > threshold)[0]
        return left_idxs, right_idxs

    def _calculate_mse(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        """ Calculates the MSE of the left and right splits by the weighted variances

        Args:
            left_y (np.ndarray): Dataset split that are less than or equal to splitting condition
            right_y (np.ndarray): Dataset split that are more than splitting condition

        Returns:
            total_mse (float): The total weighted sum of the MSEs
        """
        total_left_mse = np.var(left_y) * len(left_y) if len(left_y) > 0 else 0
        total_right_mse = np.var(right_y) * len(right_y) if len(right_y) > 0 else 0
        total_mse = (total_left_mse + total_right_mse) / (len(left_y) + len(right_y))
        return total_mse

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns the Decision Trees prediction value for an array of data points 
        by looping over all points with the _transverse_tree method.

        Args:
            X (np.ndarray): Array of data points that we want predictions for

        Returns:
            predictions (np.ndarray): Array of predictions
        """
        # Predictions array to store predictions for each sample in X
        predictions = np.array([self._traverse_tree(x, self.tree) for x in X])
        return predictions

    def _traverse_tree(self, x: np.ndarray, node: 'DecisionTree.Node'):
        """Recursive method to transverse the tree for a single sample 'x' 
        until a leaf node is reached

        Args:
            x (np.ndarray): Single row/sample of data
            node (DecisionTree.Node): The node we're currently at on the tree

        Returns:
            Node.Value (float): The prediction for x
            self._transverse_tree (Function): Recursively calls itself to move on to next node
        """
        # Checks if node is a leaf node (only leaf nodes have a value attribute)
        if node.value is not None:
            return node.value
        # Navigates to next node based on current decision node's splitting condition
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# Boosted Gradient Class
class GradientBoostAll:
    """Class for creating Gradient Boosted Regression Trees ML model

    Instance Attributes:
        max_depth (int): Max amount of branches for a tree (stops splitting once hit)
        n_estimators (int): The amount of trees (iterations) that will be gradient boosted
        learning_rate (float): The step size made each iteration to minimize the loss
        trees (list): A list of all the trees made during boosting
    """
    def __init__(self, n_estimators: int = 25, max_depth: int = 1, learning_rate: float =.1):
        self.max_depth = max_depth # Max depth of the trees
        self.n_estimators = n_estimators # Number of trees
        self.learning_rate = learning_rate # Learning rate, step size for parameter update
        self.trees = [] # List of our trees

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """Fits boosted trees to the given data

        Args:
            X_train (pd.DataFrame): Training input data
            y_train (pd.DataFrame): Training output (observed) data
        """
        # Transform args to numpy
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()
        # Initialize variables: residuals = y_train to start
        residuals = np.copy(y_train)
        # Make n_estimator amount of decision tree
        for i in range(self.n_estimators):
            # Build and Fit Tree to data
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_train, residuals)
            # Save our tree
            self.trees.append(tree)
            # Make prediction
            f_hat = tree.predict(X_train)
            # Update residuals
            residuals = residuals - (self.learning_rate*f_hat)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Uses trained Gradient Boosted Tree model to make predictions on data

        Args:
            X_test (pd.DataFrame): Testing or actual data that predictions will be made on

        Returns:
            y_hat (np.ndarray): Predictions made by trained model
        """
        # Transform arg to numpy
        X_test = X_test.to_numpy()
        # Make sure class instance has been fit to data
        if not self.trees:
            raise ValueError("This instance of GradientBoostAll class hasn't been fit to data")
        # Initialize prediction to be same length as input data
        y_hat = np.zeros((X_test.shape[0], ))
        # Sum prediction from each tree
        for tree in self.trees:
            y_hat += self.learning_rate*tree.predict(X_test)
        return y_hat

"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
METHOD = "gini"

def impurity(labels):
    """Return a non-negative impurity score for a 1D array of integer class labels."""
    # TODO: score how mixed the labels are; 0 for a pure set, larger for more mixed sets.

    if len(labels) <= 1:
        return 0.0

    values, counts = np.unique(labels, return_counts=True)
    proportions = counts / len(labels)

    if METHOD == "gini":
        return 1 - np.sum(proportions ** 2)
        
    if METHOD == "entropy":
        np.where(proportions > 0.0, proportions, 1.0)
        return -np.dot(proportions, np.log(proportions))
    
    raise ValueError(f"Unknown method {METHOD}")

# Step 2 - split_dataset
import numpy as np

def split_dataset(features, labels, feature_index, threshold):
    # TODO: partition rows into left (feature <= threshold) and right (feature > threshold)

    mask = features[:, feature_index] <= threshold
    
    features_left = features[mask]
    labels_left = labels[mask]

    features_right = features[~mask]
    labels_right = labels[~mask]

    return features_left, labels_left, features_right, labels_right

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels):
    # TODO: return a score where higher means the children are purer than the parent.
    if len(parent_labels) == 0:
        return 0.0

    w_l = len(left_labels) / len(parent_labels)
    w_r = len(right_labels) / len(parent_labels)

    return impurity(parent_labels) - w_l * impurity(left_labels) - w_r * impurity(right_labels)

# Step 4 - best_split
import numpy as np

def best_split(features, labels, feature_indices):
    # TODO: search feature_indices for the (feature, threshold) that best improves purity.

    result = {"feature_index": None, "threshold": None, "score": 0.0}
    score_best = 0.0

    for feature_index in feature_indices:
        unique_features = np.unique(features[:, feature_index])
        thresholds = (unique_features[:-1] + unique_features[1:]) / 2

        for threshold in thresholds:
            _, left_labels, _, right_labels = split_dataset(features, labels, feature_index, threshold)
            
            if len(left_labels) == 0 or len(right_labels) == 0:
                continue
            
            score = split_score(labels, left_labels, right_labels)

            if score > score_best:
                score_best = score
                result = {"feature_index": feature_index, "threshold": threshold, "score": score}
                
    return result

# Step 5 - should_stop
def should_stop(labels, depth, max_depth, min_samples_split):
    """Return True if this node should become a leaf instead of splitting further."""
    # TODO: decide whether to stop growing based on purity, depth, and size...

    if depth >= max_depth:
        return True

    if len(labels) < min_samples_split:
        return True

    if len(np.unique(labels)) == 1:
        return True

    return False

# Step 6 - leaf_prediction
def leaf_prediction(labels):
    # TODO: choose a single class label to output for a leaf given the labels that reached it
    if len(labels) == 1:
        return labels[0]

    labels_unique, counts = np.unique(labels, return_counts=True)
    return int(labels_unique[np.argmax(counts)])

# Step 7 - build_tree
def build_tree(features, labels, max_depth=10, min_samples_split=2, feature_subset=None, depth=0):
    # TODO: recursively grow a decision tree, returning a nested dict of leaf/internal nodes.

    leaf = {"leaf": True, "prediction": leaf_prediction(labels)}

    if should_stop(labels, depth, max_depth, min_samples_split):
        return leaf

    feature_indices = range(features.shape[1]) if feature_subset is None else  feature_subset
    split = best_split(features, labels, list(feature_indices))
    
    feature_index = split["feature_index"]
    threshold = split["threshold"]
    
    if feature_index is None or threshold is None:
        return leaf
    
    features_left, labels_left, features_right, labels_right = split_dataset(features, labels, feature_index, threshold)

    if len(features_left) == 0 or len(features_right) == 0:
        return leaf

    depth += 1
    node_left = build_tree(features_left, labels_left, feature_subset=feature_subset, depth=depth)
    node_right = build_tree(features_right, labels_right, feature_subset=feature_subset, depth=depth)

    return {"leaf": False, "feature_index": feature_index, "threshold": threshold, "left":  node_left, "right": node_right}

# Step 8 - predict_example_tree (not yet solved)
# TODO: implement

# Step 9 - predict_tree (not yet solved)
# TODO: implement

# Step 10 - bootstrap_sample (not yet solved)
# TODO: implement

# Step 11 - feature_subset (not yet solved)
# TODO: implement

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement


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

NUM_STEPS = 10

def best_split(features, labels, feature_indices):
    # TODO: search feature_indices for the (feature, threshold) that best improves purity.

    result: dict[str, int | float | None] = {}

    for feature_index in feature_indices:
        t_min = float(np.min(features[:, feature_index]))
        t_max = float(np.max(features[:, feature_index]))
        t_range = t_max - t_min

        for threshold in np.arange(t_min, t_max, t_range / NUM_STEPS):
            _, left_labels, _, right_labels = split_dataset(features, labels, feature_index, threshold)
            score = split_score(labels, left_labels, right_labels)

            if score > 0:
                result.setdefault("feature_index", []).append(feature_index)
                result.setdefault("threshold", []).append(threshold)
                result.setdefault("score", []).append(score)
        
    if not result:
        return {"feature_index": None, "threshold": None, "score": 0.0}

    return result

# Step 5 - should_stop (not yet solved)
# TODO: implement

# Step 6 - leaf_prediction (not yet solved)
# TODO: implement

# Step 7 - build_tree (not yet solved)
# TODO: implement

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


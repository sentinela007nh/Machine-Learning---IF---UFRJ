import numpy
import matplotlib

# Decision tree

X_train = numpy.array(
    [
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 1],
        [0, 1, 1],
        [0, 0, 0],
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ]
)

y_train = numpy.array([1, 1, 0, 0, 1, 0, 0, 1, 1, 0])


def compute_entropy(p_1):
    if len(p_1) == 0:
        return 0

    if p_1 == 0 or p_1 == 1:
        entropy = 0
    else:
        counts = numpy.unique(p_1, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -numpy.sum(probabilities * numpy.log2(probabilities)) - (
            1 - probabilities.sum()
        ) * numpy.log2(1 - probabilities.sum())
        return entropy


def split_data(X, feature_index, node_line):
    mask_left = X[feature_index, node_line] == 1
    mask_right = X[feature_index, node_line] == 0
    left_indices = feature_index[mask_left]
    right_indices = feature_index[mask_right]

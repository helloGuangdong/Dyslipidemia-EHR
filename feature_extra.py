import numpy as np


def original_feature(data):
    feature_all = []
    feature_2years = np.array([0, 1.1])
    w, h = np.shape(data)
    for ii in range(w):
        curr_feature = data[ii, :]
        feature_2years = np.concatenate([feature_2years, curr_feature])
        if (ii+1)%2 == 0:
            feature_all.append(feature_2years[2:])
            feature_2years = np.array([0, 1.1])
            continue

    return feature_all


def original_diff(data):
    feature_all = []
    feature_2years = []
    w, h = np.shape(data)
    for ii in range(w):
        curr_feature = data[ii, :]
        feature_2years.append(curr_feature)
        if (ii+1)%2 == 0:
            feature_2years = np.array(feature_2years)
            diff_feature = feature_2years[1, :] - feature_2years[0, :]
            feature_all.append(diff_feature)
            feature_2years = []
            continue

    return feature_all


if '__main__' == __name__:
    matrix = np.random.rand(12, 10)

    original_diff(matrix)

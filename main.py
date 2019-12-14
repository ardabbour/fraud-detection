"""
    Fraud Detection
"""

import pandas as pd
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier


def get_col(array, column):
    """Returns the column of a 2D array."""

    return [row[column] for row in array]


def get_combos(itr1, itr2):
    """Returns all the combinations of elements between two iterables."""

    return np.array([np.array([x, y]) for x in itr1 for y in itr2])


def separate_data(data, labels_index):
    """Returns the data separated into a feature set and a label set. 'data':
    DataFrame object."""

    return data.drop([labels_index], axis=1), data[labels_index]


def correlation_encoding(data, feature_headings, label_heading):
    """Gets the correlation of the selected features with relation to the label,
    then replaces the features encoded as correlation values. The logic is that
    the algorithm will bias towards higher correlation values, thus possibly
    more relevant values."""

    for feature_heading in feature_headings:
        dic = (
            data[[feature_heading, label_heading]]
            .groupby([feature_heading], as_index=False)
            .mean()
            .sort_values(by=label_heading, ascending=False)
        )
        data[feature_heading] = data[feature_heading].replace(
            tuple(dic[feature_heading]), tuple(dic[label_heading])
        )

    return data


# Read data file.
DATA = pd.read_csv("data.csv")

# Extract date information
DATA["incident_date"] = pd.to_datetime(DATA["incident_date"], errors="coerce")
DATA["incident_month"] = DATA["incident_date"].dt.month
DATA["incident_day"] = DATA["incident_date"].dt.day

# Drop useless columns
DATA = DATA.drop(
    [
        "policy_number",
        "policy_bind_date",
        "incident_date",
        "incident_location",
        "auto_model",
    ],
    axis=1,
)


# Deal with unknown values
DATA = DATA.replace("?", np.NaN)
DATA["collision_type"].fillna(DATA["collision_type"].mode()[0], inplace=True)
DATA["property_damage"].fillna(False, inplace=True)
DATA["police_report_available"].fillna(False, inplace=True)

# Replace strings with True/False values
DATA = DATA.replace(("YES", "Y", "NO", "N"), (True, True, False, False))

# Encode based on correlation
DATA = correlation_encoding(
    DATA,
    [
        # "auto_model",
        "auto_make",
        "police_report_available",
        "property_damage",
        "incident_city",
        "incident_state",
        "authorities_contacted",
        "incident_severity",
        "collision_type",
        "incident_type",
        "insured_relationship",
        "insured_hobbies",
        "insured_occupation",
        "insured_education_level",
        "insured_sex",
        "policy_csl",
        "policy_state",
        "incident_month",
        "incident_day",
    ],
    "fraud_reported",
)


FEATURES, LABELS = separate_data(DATA, "fraud_reported")

TRAIN_FEATURES, TEST_FEATURES, TRAIN_LABELS, TEST_LABELS = train_test_split(
    FEATURES, LABELS
)

########################### Using neural networks ##############################
ALPHA_PARAMS = [10e-3, 10e-2, 10e-1]
HIDDEN_LAYER_SIZES_PARAMS = [4, 5, 6]
COMBOS = get_combos(ALPHA_PARAMS, HIDDEN_LAYER_SIZES_PARAMS)

SCORES = []
for combo in COMBOS:
    CLF = MLPClassifier(
        solver="lbfgs", alpha=combo[0], hidden_layer_sizes=(combo[1], 2), random_state=1
    ).fit(TRAIN_FEATURES, TRAIN_LABELS)
    SCORES.append(cross_val_score(CLF, TRAIN_FEATURES, TRAIN_LABELS, cv=3))
ACCURACIES = [np.mean(x) for x in SCORES]

# Train a classifier using the best C value obtained
BEST_PARAMS = COMBOS[ACCURACIES.index(max(ACCURACIES))]
CLF = MLPClassifier(
    solver="lbfgs", alpha=combo[0], hidden_layer_sizes=(combo[1], 2), random_state=1
)

# Get accuracy, precision, and recall for the training data.
TRAIN_PREDICTIONS = CLF.predict(TRAIN_FEATURES)
TRAIN_ACCURACY = np.round(accuracy_score(TRAIN_LABELS, TRAIN_PREDICTIONS), 4)
TRAIN_PRECISION = np.round(precision_score(TRAIN_LABELS, TRAIN_PREDICTIONS), 4)
TRAIN_RECALL = np.round(recall_score(TRAIN_LABELS, TRAIN_PREDICTIONS), 4)

# Get accuracy, precision, and recall for the test data.
TEST_PREDICTIONS = CLF.predict(TEST_FEATURES)
TEST_ACCURACY = np.round(accuracy_score(TEST_LABELS, TEST_PREDICTIONS), 4)
TEST_PRECISION = np.round(precision_score(TEST_LABELS, TEST_PREDICTIONS), 4)
TEST_RECALL = np.round(recall_score(TEST_LABELS, TEST_PREDICTIONS), 4)

# Print solution to terminal.
print()
print("-----------------------------------------------------------------------")
print("SOLUTION 2, part b:")
print("Alpha parameters tested: {}".format(ALPHA_PARAMS))
print("Hidden layer sizes parameters tested: {}".format(HIDDEN_LAYER_SIZES_PARAMS))
print("Highest cross-validation accuracy: {}".format(round(max(ACCURACIES), 4)))
print("Corresponding Alpha value: {}".format(BEST_PARAMS[0]))
print("Corresponding hidden layer sizes value: {}".format((BEST_PARAMS[1], 2)))
print()
print("For training data:")
print(
    "Accuracy: {}, Precision: {}, Recall: {}".format(
        TRAIN_ACCURACY, TRAIN_PRECISION, TRAIN_RECALL
    )
)
print()
print("For test data:")
print(
    "Accuracy: {}, Precision: {}, Recall: {}".format(
        TEST_ACCURACY, TEST_PRECISION, TEST_RECALL
    )
)

# Plot 3-fold cross-validation surface graph.
PLOT = pd.DataFrame({"X": get_col(COMBOS, 0), "Y": get_col(COMBOS, 1), "Z": ACCURACIES})


fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot_trisurf(np.log10(PLOT.X), np.log2(PLOT.Y), PLOT.Z)
ax.set_xlabel("log10(C)")
ax.set_ylabel("log2(gamma)")
ax.set_zlabel("Accuracy")
ax.set_title("3-Fold Cross Validation Accuracy Using the RBF Kernel")
plt.show()


########################### Using a nonlinear SVM ##############################
C_PARAMS = [10e-4, 10e-3, 10e-2, 10e-1, 10e0, 10e1, 10e2]
GAMMA_PARAMS = [2e-4, 2e-3, 2e-2, 2e-1, 2e0]
COMBOS = np.array([np.array([x, y]) for x in C_PARAMS for y in GAMMA_PARAMS])

SCORES = []
for combo in COMBOS:
    CLF = svm.SVC(kernel="rbf", C=combo[0], gamma=combo[1]).fit(
        TRAIN_FEATURES, TRAIN_LABELS
    )
    SCORES.append(cross_val_score(CLF, TRAIN_FEATURES, TRAIN_LABELS, cv=3))
ACCURACIES = [np.mean(x) for x in SCORES]

# Train a classifier using the best C value obtained
BEST_PARAMS = COMBOS[ACCURACIES.index(max(ACCURACIES))]
CLF = svm.SVC(kernel="rbf", C=combo[0], gamma=combo[1]).fit(
    TRAIN_FEATURES, TRAIN_LABELS
)

# Get accuracy, precision, and recall for the training data.
TRAIN_PREDICTIONS = CLF.predict(TRAIN_FEATURES)
TRAIN_ACCURACY = np.round(accuracy_score(TRAIN_LABELS, TRAIN_PREDICTIONS), 4)
TRAIN_PRECISION = np.round(precision_score(TRAIN_LABELS, TRAIN_PREDICTIONS), 4)
TRAIN_RECALL = np.round(recall_score(TRAIN_LABELS, TRAIN_PREDICTIONS), 4)

# Get accuracy, precision, and recall for the test data.
TEST_PREDICTIONS = CLF.predict(TEST_FEATURES)
TEST_ACCURACY = np.round(accuracy_score(TEST_LABELS, TEST_PREDICTIONS), 4)
TEST_PRECISION = np.round(precision_score(TEST_LABELS, TEST_PREDICTIONS), 4)
TEST_RECALL = np.round(recall_score(TEST_LABELS, TEST_PREDICTIONS), 4)

# Print solution to terminal.
print()
print("-----------------------------------------------------------------------")
print("SOLUTION 2, part b:")
print("C parameters tested: {}".format(C_PARAMS))
print("gamma parameters tested: {}".format(GAMMA_PARAMS))
print("Highest cross-validation accuracy: {}".format(round(max(ACCURACIES), 4)))
print("Corresponding C value: {}".format(BEST_PARAMS[0]))
print("Corresponding gamma value: {}".format(BEST_PARAMS[1]))
print()
print("For training data:")
print(
    "Accuracy: {}, Precision: {}, Recall: {}".format(
        TRAIN_ACCURACY, TRAIN_PRECISION, TRAIN_RECALL
    )
)
print()
print("For test data:")
print(
    "Accuracy: {}, Precision: {}, Recall: {}".format(
        TEST_ACCURACY, TEST_PRECISION, TEST_RECALL
    )
)

# Plot 3-fold cross-validation surface graph.
PLOT = pd.DataFrame({"X": get_col(COMBOS, 0), "Y": get_col(COMBOS, 1), "Z": ACCURACIES})


fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot_trisurf(np.log10(PLOT.X), np.log2(PLOT.Y), PLOT.Z)
ax.set_xlabel("log10(C)")
ax.set_ylabel("log2(gamma)")
ax.set_zlabel("Accuracy")
ax.set_title("3-Fold Cross Validation Accuracy Using the RBF Kernel")
plt.show()


#! /usr/bin/env python3

# Data processing tools
import pandas as pd
import numpy as np
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from category_encoders.target_encoder import TargetEncoder

# Visualization tools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Metrics and scoring tools
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Random forest classifier
from sklearn.ensemble import RandomForestClassifier


def get_col(array, column_index):
    """Returns the column of a 2D array."""

    return [row[column_index] for row in array]


def get_combos(itr1, itr2):
    """Returns all the combinations of elements between two iterables."""

    return [[x, y] for x in itr1 for y in itr2]


if __name__ == "__main__":
    # Read data file.
    DATA = pd.read_csv("data.csv")

    # Extract date information
    DATA["incident_date"] = pd.to_datetime(DATA["incident_date"], errors="coerce")
    DATA["incident_month"] = DATA["incident_date"].dt.month
    DATA["incident_day"] = DATA["incident_date"].dt.day

    DATA["policy_bind_date"] = pd.to_datetime(DATA["policy_bind_date"], errors="coerce")
    DATA["policy_bind_month"] = DATA["policy_bind_date"].dt.month
    DATA["policy_bind_day"] = DATA["policy_bind_date"].dt.day

    # Drop useless columns
    DATA.drop(
        ["policy_number", "policy_bind_date", "incident_date", "incident_location",],
        axis=1,
        inplace=True,
    )

    # Deal with unknown values
    DATA.replace("?", np.NaN, inplace=True)
    DATA["collision_type"].fillna(DATA["collision_type"].mode()[0], inplace=True)
    DATA["property_damage"].fillna(False, inplace=True)
    DATA["police_report_available"].fillna(False, inplace=True)

    # Replace strings with True/False values
    DATA = DATA.replace(("YES", "Y", "NO", "N"), (True, True, False, False))

    # Seperate the data into features and labels
    FEATURES, LABELS = DATA.drop(["fraud_reported"], axis=1), DATA["fraud_reported"]

    # Use target encoding with smoothing for categorical features (strings)
    FEATURES = TargetEncoder().fit(FEATURES, LABELS).transform(FEATURES, LABELS)

    # Use SMOTE oversampling with ENN undersampling to balance the dataset
    FEATURES, LABELS = SMOTEENN().fit_sample(FEATURES, LABELS.values.ravel())

    # Split the dataset into test and train datasets
    TRAIN_FEATURES, TEST_FEATURES, TRAIN_LABELS, TEST_LABELS = train_test_split(
        FEATURES, LABELS
    )

    # Create hyperparameter combinations to test using cross validation
    N_ESTIMATORS_PARAMS = [300, 500, 700, 900, 1100]
    CRITERION_PARAMS = ["gini", "entropy"]
    COMBOS = get_combos(N_ESTIMATORS_PARAMS, CRITERION_PARAMS)
    SCORES = []

    # Create a classifier with each combination of hyperparameters and measure its
    # cross-validation score.
    for combo in COMBOS:
        CLF = RandomForestClassifier(n_estimators=combo[0], criterion=combo[1]).fit(
            TRAIN_FEATURES, TRAIN_LABELS
        )
        SCORES.append(cross_val_score(CLF, TRAIN_FEATURES, TRAIN_LABELS))

    # Get the accuracies from the scores
    ACCURACIES = [np.mean(x) for x in SCORES]

    # Train a classifier using the best parameters
    BEST_PARAMS = COMBOS[ACCURACIES.index(max(ACCURACIES))]
    CLF = RandomForestClassifier(
        n_estimators=BEST_PARAMS[0], criterion=BEST_PARAMS[1]
    ).fit(TRAIN_FEATURES, TRAIN_LABELS)

    # Make predictions using the trained classifier
    TRAIN_PREDICTIONS = CLF.predict(TRAIN_FEATURES)
    TEST_PREDICTIONS = CLF.predict(TEST_FEATURES)

    # Get accuracy the train and test datasets.
    TRAIN_ACCURACY = np.round(accuracy_score(TRAIN_LABELS, TRAIN_PREDICTIONS), 2)
    TEST_ACCURACY = np.round(accuracy_score(TEST_LABELS, TEST_PREDICTIONS), 2)

    # Print report to terminal
    print("Estimator numbers tested: {}".format(N_ESTIMATORS_PARAMS))
    print("Criterions tested: {}".format(CRITERION_PARAMS))
    print("Highest cross-validation accuracy: {}".format(round(max(ACCURACIES), 4)))
    print("Corresponding estimator number: {}".format(BEST_PARAMS[0]))
    print("Corresponding criterion: {}".format(BEST_PARAMS[1]))
    print()
    print("Train dataset accuracy: {}".format(TRAIN_ACCURACY))
    print("Test dataset accuracy: {}".format(TEST_ACCURACY))
    print()
    print(classification_report(TEST_LABELS, TEST_PREDICTIONS))

    # Plot 5-fold cross-validation surface graph.
    CRITS = [0 if x == "gini" else 1 for x in get_col(COMBOS, 1)]
    PLOT = pd.DataFrame({"X": get_col(COMBOS, 0), "Y": CRITS, "Z": ACCURACIES})

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_trisurf(PLOT.X, PLOT.Y, PLOT.Z)
    ax.set_xlabel("Number of estimators")
    ax.set_ylabel("Criterion\n0: 'gini, 1: 'entropy'")
    ax.set_zlabel("Accuracy")
    ax.set_title("5-Fold Cross Validation Accuracy Using Random Forests")
    plt.savefig(fname="plot.pdf")
    plt.show()

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import sys

def train_ova_classifier(X_train, y_train):
    """
    Train three One vs All classifiers without using loops
    """
    y_1 = (y_train == 1).astype(int)
    y_2 = (y_train == 2).astype(int)
    y_3 = (y_train == 3).astype(int)

    clf_1 = LogisticRegression().fit(X_train, y_1)
    clf_2 = LogisticRegression().fit(X_train, y_2)
    clf_3 = LogisticRegression().fit(X_train, y_3)

    return clf_1, clf_2, clf_3

def apply_ova_classifier(clf_1, clf_2, clf_3, X_test):
    """
    Apply the three classifiers to the test data
    """
    prob_1 = clf_1.predict_proba(X_test)[:, 1]
    prob_2 = clf_2.predict_proba(X_test)[:, 1]
    prob_3 = clf_3.predict_proba(X_test)[:, 1]

    probs = np.column_stack((prob_1, prob_2, prob_3))
    predictions = np.argmax(probs, axis=1) + 1

    return predictions

def main():
    if len(sys.argv) != 3:
        print("Usage: python ova_diy.py <train_file> <test_file>")
        sys.exit(1)

    train_path = sys.argv[1]
    test_path = sys.argv[2]

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = train_data[['f1', 'f2', 'f3', 'f4']].values
    y_train = train_data['target'].values

    X_test = test_data[['f1', 'f2', 'f3', 'f4']].values

    clf_1, clf_2, clf_3 = train_ova_classifier(X_train, y_train)

    predictions = apply_ova_classifier(clf_1, clf_2, clf_3, X_test)

    output_df = test_data.copy()
    output_df['predicted_value'] = predictions

    output_df.to_csv('predictions.csv', index=False)


if __name__ == "__main__":
    main()

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np


DATA = 'data/'


def main():
    train_data = pd.read_csv(DATA+'cybersecurity_training.csv', delimiter='|')



    test_data = pd.read_csv(DATA+'cybersecurity_test.csv', delimiter='|')

    categorical_columns = ['client_code', 'categoryname', 'ip', 'ipcategory_name', 'ipcategory_scope',
                           'grandparent_category', 'weekday', 'dstipcategory_dominate', 'srcipcategory_dominate']

    # Perform one-hot encoding on the categorical columns
    train_data = pd.get_dummies(train_data, columns=categorical_columns)
    test_data = pd.get_dummies(test_data, columns=categorical_columns)

    # Ensure the test data has the same columns as the training data
    missing_cols = set(train_data.columns) - set(test_data.columns)
    for c in missing_cols:
        test_data[c] = 0
    test_data = test_data[train_data.columns]

    # Select features and target
    features = train_data.drop(['notified', 'alert_ids'], axis=1)
    target = train_data['notified']

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

    # Instantiate the model
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    preds_val = model.predict_proba(X_val)[:, 1]



    # Calculate the ROC AUC score
    roc_score = roc_auc_score(y_val, preds_val)
    print(f"Validation ROC AUC Score: {roc_score}")

    # Make predictions on the test set
    test_features = test_data.drop(['notified', 'alert_ids'], axis=1)

    preds_test = model.predict_proba(test_features)[:, 1]
    np.savetxt(DATA + 'result2.txt', preds_test, delimiter='\n', fmt='%1.3f')



if __name__ == '__main__':
    main()
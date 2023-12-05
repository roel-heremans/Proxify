import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.util import get_training_data_between, plotly_data, resample_df


if __name__ == "__main__":
    # Assuming 'data' contains your dataset with columns 'Timestamp', 'Temperature', and 'GroundTruth'
    # 'GroundTruth' has binary labels indicating pump usage (1 for in use, 0 for not in use)
    file_names = {
        '01': 'Consolidated UG Data Jan 2023',
        '02': 'Kaliro Use Data (Kakosi Budumba) 230912'
    }
    run_file_id = '01'
    feature_list = ['Temperature', 'Temp_diff', 'Temp_extrema']

    data_files = {key: os.path.join('data', '{}.xlsx'.format(value)) for key, value in file_names.items()}
    data_train = get_training_data_between(data_files[run_file_id],
                              start_datetime = pd.to_datetime('2023-01-13 10:49:00'),
                              stop_datetime = pd.to_datetime('2023-01-13 18:00:00'))

    # Prepare features (Temperature) and target (GroundTruth)
    X_train = data_train[feature_list]  # Features
    y_train = data_train['GroundTruth']    # Target

    # Make predictions
    data_test = get_training_data_between(data_files[run_file_id],
                              start_datetime = pd.to_datetime('2023-02-02 08:44:00'),
                              stop_datetime = pd.to_datetime('2023-02-02 13:49:00'))

    X_test = data_test[feature_list]  # Features
    y_test = data_test['GroundTruth']    # Target

    # Convert data into DMatrix format (XGBoost's internal optimized data structure)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Define parameters for XGBoost
    params = {
        'objective': 'binary:logistic',  # Binary classification
        'max_depth': 3,
        'learning_rate': 0.1,
        'eval_metric': 'error'  # Evaluation metric
    }

    # Train the XGBoost model
    num_round = 100  # Number of boosting rounds
    bst = xgb.train(params, dtrain, num_round)

    # Predict on the train set
    y_pred_prob = bst.predict(dtrain)
    # Convert predicted probabilities to binary predictions
    y_pred_train = [1 if pred > 0.5 else 0 for pred in y_pred_prob]

    # Predict on the test set
    y_pred_prob = bst.predict(dtest)
    # Convert predicted probabilities to binary predictions
    y_pred_test = [1 if pred > 0.5 else 0 for pred in y_pred_prob]


    # Evaluate accuracy, precision, recall, and F1-score On Train
    accuracy_train = accuracy_score(y_train, y_pred_train)
    precision_train = precision_score(y_train, y_pred_train)
    recall_train = recall_score(y_train, y_pred_train)
    f1_train = f1_score(y_train, y_pred_train)

    df_eval_train = data_train[['Temperature', 'Temp_diff', 'GroundTruth']]
    df_eval_train['Prediction'] = y_pred_train

    col_name_dict = {'temp_col_name': 'Temperature', 'gt_col_name': 'GroundTruth', 'pred_col_name': 'Prediction'}
    dict_for_plot_title = {'file': [run_file_id,file_names[run_file_id]],
                           'model': ['XGBoost', 'OnTrainData'],
                           'eval_train': [accuracy_train,precision_train,recall_train,f1_train]}

    plotly_data(df_eval_train, col_name_dict, dict_for_plot_title)




    # Evaluate accuracy, precision, recall, and F1-score On Test
    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test)
    recall_test = recall_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test)

    df_eval_test = data_test[['Temperature', 'GroundTruth']]
    df_eval_test['Prediction'] = y_pred_test

    col_name_dict = {'temp_col_name': 'Temperature', 'gt_col_name': 'GroundTruth', 'pred_col_name': 'Prediction'}
    dict_for_plot_title = {'file': [run_file_id,file_names[run_file_id]],
                           'model': ['XGBoost', 'OnTestData'],
                           'eval_test': [accuracy_test,precision_test,recall_test,f1_test]}

    plotly_data(df_eval_test, col_name_dict, dict_for_plot_title)


    print(f"Accuracy: {accuracy_train:.2f} -> {accuracy_test:.2f}")
    print(f"Precision: {precision_train:.2f} -> {precision_test:.2f}")
    print(f"Recall: {recall_train:.2f} -> {recall_train:.2f}")
    print(f"F1: {f1_train:.2f} -> {f1_test:.2f}")
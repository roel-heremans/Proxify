import os
import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from scipy.signal import find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

pd.options.mode.chained_assignment = None


def create_output_directories():
    ''' Checks if the directories are already existing, if not existing they will get created
    '''
    to_create_paths = ['plots', 'plotly']
    for my_path in to_create_paths:
        if not os.path.exists(os.path.join(os.getcwd(), my_path)):
            # Create the directory
            os.makedirs(os.path.join(os.getcwd(), my_path))

def getting_extrema(df, col_name):

    window_size = 2
    df['Temp_smooth'] = df['Temperature'].rolling(window=window_size).mean()
    df['Temp_smooth'].fillna(method='bfill', inplace=True)
    temperatures = df['Temp_smooth']

    #maxima_indices, _ = find_peaks(temperatures, prominence=0.3, distance=4)
    #minima_indices, _ = find_peaks(-temperatures, prominence=0.3, distance=5)
    maxima_indices, _ = find_peaks(temperatures,  distance=3)
    minima_indices, _ = find_peaks(-temperatures,  distance=3)

    # Todo: check that the maxima_indices and minima_indices are alternating
    # Ensure alternating pattern: a maxima followed by a minima and vice versa
    alternating_max_min = []
    for i in range(min(len(minima_indices), len(maxima_indices))):
        alternating_max_min.append(maxima_indices[i])
        alternating_max_min.append(minima_indices[i])

    # Generate an array with alternating maxima and minima
    alternating_temperatures = temperatures[np.sort(alternating_max_min)]

    # Assign the calculated values to the DataFrame column 'Temp_extrema'
    df['Temp_extrema'] = 1
    for val1,val2 in zip(alternating_temperatures.index[0::2], alternating_temperatures.index[1::2]):
        df.loc[val1:val2, 'Temp_extrema'] = -1
    df.loc[alternating_temperatures.index[0::2], 'Temp_extrema'] = 2
    df.loc[alternating_temperatures.index[1::2], 'Temp_extrema'] = -2

    # Uncomment for debugging purpose
    #fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    #ax1.plot(df['Temp_smooth'])
    #ax1.plot(alternating_temperatures[0::2],marker='o',color='r', linestyle='')
    #ax1.plot(alternating_temperatures[1::2],marker='o',color='g', linestyle='')
    #ax2.plot(df["Temp_extrema"], marker='o', color='k', linestyle='')
    #plt.show()
    return df

def get_training_data_between(filename,
                              start_datetime = pd.to_datetime('2022-01-01 00:00:00'),
                              stop_datetime = pd.to_datetime('2025-01-01 00:00:00')):

    df, col_name_dict = import_xlsx_to_df(filename,
                                          timestamp_col_name='DateTime_EAT',
                                          temp_col_name='Celsius',
                                          gt_col_name='Use_event',
                                          gt_start_col_name='use_start',
                                          gt_end_col_name='use_end',
                                          batch_col_name='batch')
    df_filtered = select_date_time_range_from_df(df, start_datetime, stop_datetime)
    df_filtered.reset_index(inplace=True)

    col_mapping = {'Celsius': 'Temperature', 'Use_event': 'GroundTruth'}
    df_filtered = df_filtered.rename(columns=col_mapping)

    df_filtered_resampled = resample_df(df_filtered.set_index('Timestamp'), resample='1T')

    window_size = 15
    df_filtered_resampled['Temp_Rolling_Mean'] = df_filtered_resampled['Temperature'].rolling(window=window_size).mean()
    df_filtered_resampled['Temp_Rolling_Mean'].fillna(method='bfill', inplace=True)

    df_filtered_resampled['Temp_diff'] = df_filtered_resampled['Temperature'].diff()
    df_filtered_resampled['Temp_diff'].fillna(method='bfill', inplace=True)
    df_filtered_resampled = getting_extrema(df_filtered_resampled, 'Temperature')

    return df_filtered_resampled[['Temperature', 'Temp_diff', 'Temp_Rolling_Mean', 'Temp_extrema', 'GroundTruth']]


def import_xlsx_to_df(filename, timestamp_col_name='DateTime_EAT', temp_col_name='Celsius', gt_col_name='',
                      gt_start_col_name='', gt_end_col_name='', batch_col_name=''):
    '''
    Reads in the excel datafiles containing at least:
    -a column that reflects on the timestamp (formated like 2023-01-06 12:21:26)
    -a column containing the temperatures
    -optional: a column containing the annotation if the pump was on or off during that timestamp
    -optional: a column wiht the start pump times
    -optional: a column with the end pump times
    -optional: batch reflecting on one location or one sequnece of recordings without timestamp gaps

    :param filename: xlsx file to be processed (path and basename joined into filename)
    :param timestamp_col_name: the name of the column that contains the timestamp
    :return:
    '''
    df = pd.read_excel(filename)
    df[timestamp_col_name] = pd.to_datetime(df[timestamp_col_name])
    df = df.set_index(timestamp_col_name)
    df.index = df.index.rename('Timestamp')

    col_name_dict = {'temp_col_name': temp_col_name}
    col_name_optional_dict = {'gt_col_name': gt_col_name,
                              'gt_start_col_name':gt_start_col_name,
                              'gt_end_col_name': gt_end_col_name,
                              'batch_col_name': batch_col_name}

    for key,value in col_name_optional_dict.items():
        if value:   # only when non-empty string, the col_name_dict gets updated
            col_name_dict.update({key: value})

    return df, col_name_dict


def my_local_extrema():
    # Assuming 'data' contains your dataset with columns 'Timestamp', 'Temperature', and 'GroundTruth'
    # 'GroundTruth' has binary labels indicating pump usage (1 for in use, 0 for not in use)
    file_names = {
        '01': 'Consolidated UG Data Jan 2023',
        '02': 'Kaliro Use Data (Kakosi Budumba) 230912'
    }
    run_file_id = '01'
    feature_list = ['Temperature', 'Temp_diff', 'Temp_Rolling_Mean']
    data_files = {key: os.path.join('data', '{}.xlsx'.format(value)) for key, value in file_names.items()}
    data_train = get_training_data_between(data_files[run_file_id],
                              start_datetime = pd.to_datetime('2023-01-13 10:49:00'),
                              stop_datetime = pd.to_datetime('2023-01-13 18:00:00'))

    # Making predictions on the training set
    y_train = data_train['GroundTruth']
    y_pred_train = data_train['Temp_extrema']!=1

    # Make predictions
    data_test = get_training_data_between(data_files[run_file_id],
                              start_datetime = pd.to_datetime('2023-02-02 08:44:00'),
                              stop_datetime = pd.to_datetime('2023-02-02 13:49:00'))

    # Making predictions on the test set
    y_test = data_test['GroundTruth']
    y_pred_test = data_test['Temp_extrema']!=1


    # Evaluate accuracy, precision, recall, and F1-score On Train
    accuracy_train = accuracy_score(y_train, y_pred_train)
    precision_train = precision_score(y_train, y_pred_train)
    recall_train = recall_score(y_train, y_pred_train)
    f1_train = f1_score(y_train, y_pred_train)

    df_eval_train = data_train[['Temperature', 'Temp_diff', 'GroundTruth']]
    df_eval_train['Prediction'] = y_pred_train

    col_name_dict = {'temp_col_name': 'Temperature', 'gt_col_name': 'GroundTruth', 'pred_col_name': 'Prediction'}
    dict_for_plot_title = {'file': [run_file_id,file_names[run_file_id]],
                           'model': ['LocalExtrema', 'OnTrainData'],
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
                           'model': ['LocalExtrema', 'OnTestData'],
                           'eval_test': [accuracy_test,precision_test,recall_test,f1_test]}

    plotly_data(df_eval_test, col_name_dict, dict_for_plot_title)

    print("\nLocal Extrema:\n**************")
    print(f"Accuracy: {accuracy_train:.2f} -> {accuracy_test:.2f}")
    print(f"Precision: {precision_train:.2f} -> {precision_test:.2f}")
    print(f"Recall: {recall_train:.2f} -> {recall_test:.2f}")
    print(f"F1: {f1_train:.2f} -> {f1_test:.2f}")

    res_dict = {'Accuracy': [accuracy_train, accuracy_test],
                'Precision': [precision_train, precision_test],
                'Recall': [recall_train, recall_test],
                'F1': [f1_train, f1_test]}

    return res_dict


def my_logistic_regression():
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

    # Initialize the Logistic Regression model
    model = LogisticRegression(class_weight='balanced')

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the same Training set to see if the model has at all some potential
    y_pred_train = model.predict(X_train)

    # Evaluate accuracy, precision, recall, and F1-score on Training set
    accuracy_train = accuracy_score(y_train, y_pred_train)
    precision_train = precision_score(y_train, y_pred_train)
    recall_train = recall_score(y_train, y_pred_train)
    f1_train = f1_score(y_train, y_pred_train)

    df_eval_train = data_train[['Temperature', 'Temp_diff', 'GroundTruth']]
    df_eval_train['Prediction'] = y_pred_train

    col_name_dict = {'temp_col_name': 'Temperature', 'gt_col_name': 'GroundTruth', 'pred_col_name': 'Prediction'}
    dict_for_plot_title = {'file': [run_file_id,file_names[run_file_id]],
                           'model': ['LogReg', 'OnTrainData'],
                           'eval_train': [accuracy_train, precision_train, recall_train, f1_train]}
    plotly_data(df_eval_train, col_name_dict, dict_for_plot_title)


    # Make predictions on Test set
    data_test = get_training_data_between(data_files[run_file_id],
                              start_datetime = pd.to_datetime('2023-02-02 08:44:00'),
                              stop_datetime = pd.to_datetime('2023-02-02 13:49:00'))
    X_test = data_test[feature_list]  # Features
    y_test = data_test['GroundTruth']    # Target
    y_pred = model.predict(X_test)

    # Evaluate accuracy, precision, recall, and F1-score on Test data
    accuracy_test = accuracy_score(y_test, y_pred)
    precision_test = precision_score(y_test, y_pred)
    recall_test = recall_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred)

    df_eval = data_test[['Temperature', 'Temp_diff', 'GroundTruth']]
    df_eval['Prediction'] = y_pred

    col_name_dict = {'temp_col_name': 'Temperature', 'gt_col_name': 'GroundTruth', 'pred_col_name': 'Prediction'}
    dict_for_plot_title = {'file': [run_file_id,file_names[run_file_id]],
                           'model': ['LogReg', 'OnTestData'],
                           'eval_test': [accuracy_test, precision_test, recall_test, f1_test]}
    plotly_data(df_eval, col_name_dict, dict_for_plot_title)

    print("\nLogistic Regression:\n********************")
    print(f"Accuracy: {accuracy_train:.2f} -> {accuracy_test:.2f}")
    print(f"Precision: {precision_train:.2f} -> {precision_test:.2f}")
    print(f"Recall: {recall_train:.2f} -> {recall_test:.2f}")
    print(f"F1: {f1_train:.2f} -> {f1_test:.2f}")

    res_dict = {'Accuracy': [accuracy_train, accuracy_test],
                'Precision': [precision_train, precision_test],
                'Recall': [recall_train, recall_test],
                'F1': [f1_train, f1_test]}
    return res_dict


def my_xgb():
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

    print("\nXGBoost:\n*********")
    print(f"Accuracy: {accuracy_train:.2f} -> {accuracy_test:.2f}")
    print(f"Precision: {precision_train:.2f} -> {precision_test:.2f}")
    print(f"Recall: {recall_train:.2f} -> {recall_test:.2f}")
    print(f"F1: {f1_train:.2f} -> {f1_test:.2f}")

    res_dict = {'Accuracy': [accuracy_train, accuracy_test],
                'Precision': [precision_train, precision_test],
                'Recall': [recall_train, recall_test],
                'F1': [f1_train, f1_test]}

    return res_dict


def resample_df(df, resample='1T'):
    # Ensure the index is in DateTime format
    df.index = pd.to_datetime(df.index)

    # Resample to 1-minute intervals
    resampled_df = df.resample(resample).ffill()
    return resampled_df


def select_date_time_range_from_df(df, start_datetime, stop_datetime):

    # Select data within the specified time range
    return df.loc[(df.index >= start_datetime) & (df.index <= stop_datetime)]


def plotly_data(df, col_name_dict, dict_for_plot_title):
    fig = go.Figure()

    # Plotting the 'Celcius' column against the timestamp index
    fig.add_trace(go.Scatter(x=df.index, y=df[col_name_dict['temp_col_name']], mode='markers+lines', name='Temperature', line=dict(color='blue')))

    # Customize plot title
    batch_id = dict_for_plot_title.get('batch','')
    file_id = dict_for_plot_title['file'][0]
    file_str = dict_for_plot_title['file'][1]
    eval_train = dict_for_plot_title.get('eval_train','')
    eval_test = dict_for_plot_title.get('eval_test','')
    my_model = dict_for_plot_title.get('model','')

    # constructing the title (adding parts of title sequencially for existing keywords)
    graph_title = 'File{}: {}<br>Temp over Time'.format(file_id, file_str)
    if batch_id:
        graph_title = graph_title + ': Batch={}'.format(batch_id)
    if my_model:
        graph_title = graph_title + '   Model: {} {}'.format(my_model[0], my_model[1])
    if eval_train:
        graph_title = graph_title + \
                      '<br>Evaluation on Train: Acc={:.2f}, Prec={:.2f}, Recall={:.2f}, F1={:.2f}'.format(eval_train[0],
                                                                                                 eval_train[1],
                                                                                                 eval_train[2],
                                                                                                 eval_train[3])
    if eval_test:
        graph_title = graph_title + \
                      '<br>Evaluation on Test: Acc={:.2f}, Prec={:.2f}, Recall={:.2f}, F1={:.2f}'.format(eval_test[0],
                                                                                                         eval_test[1],
                                                                                                         eval_test[2],
                                                                                                         eval_test[3])


    fig.update_layout(title=graph_title, xaxis_title='Timestamp', yaxis_title='Celcius', hovermode='x')

    # Plotting rectangles indicating when the pump was annotated to be ON
    gt_col_name = col_name_dict.get('gt_col_name', '')
    pred_col_name = col_name_dict.get('pred_col_name', '')

    # Setting the hight of the rectangles (from mean to max in case the prediction is present as well)
    # (from min to max if the prediction is not present)
    if pred_col_name:
        y0 = df[col_name_dict['temp_col_name']].mean()
        y1 = df[col_name_dict['temp_col_name']].max()
    else:
        y0 = df[col_name_dict['temp_col_name']].min()
        y1 = df[col_name_dict['temp_col_name']].max()

    if gt_col_name:
        on_intervals = []
        on_status = False
        bin_size = df.index[1] - df.index[0]

        for i, status in enumerate(df[gt_col_name].fillna(0)):
            if status == 1 and not on_status:
                start = df.index[i]
                on_status = True
            elif status == 0 and on_status:
                end = df.index[i - 1]
                on_status = False
                on_intervals.append((start, end))

        if on_status:
            on_intervals.append((start, start))

        for start, end in on_intervals:
            fig.add_shape(type="rect", x0=start - pd.Timedelta(bin_size / 2), y0=y0,
                          x1=end + pd.Timedelta(bin_size), y1=y1,
                          line=dict(color="red", width=1), fillcolor="red", opacity=0.3)
        # Add a custom legend entry for the line shape
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color="red", width=10), fillcolor="red", opacity=0.3, name='GT: On-Off'))

    # Plotting rectangles indicating when the pump prediction was ON
    if pred_col_name:
        y0 = df[col_name_dict['temp_col_name']].min()
        y1 = df[col_name_dict['temp_col_name']].mean()
        pred_on_intervals = []
        pred_on_status = False
        bin_size = df.index[1] - df.index[0]

        for i, status in enumerate(df[pred_col_name].fillna(0)):
            if status == 1 and not pred_on_status:
                start = df.index[i]
                pred_on_status = True
            elif status == 0 and pred_on_status:
                end = df.index[i - 1]
                pred_on_status = False
                pred_on_intervals.append((start, end))

        if pred_on_status:
            pred_on_intervals.append((start, start))

        for start, end in pred_on_intervals:
            fig.add_shape(type="rect", x0=start - pd.Timedelta(bin_size / 2), y0=y0,
                          x1=end + pd.Timedelta(bin_size), y1=y1,
                          line=dict(color="blue", width=1), fillcolor="blue", opacity=0.3)
        # Add a custom legend entry for the line shape
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color="blue", width=10), fillcolor="blue", opacity=0.3, name='Pred: On-Off'))


    # Adding annotations
    y_center = df[col_name_dict['temp_col_name']].mean()
    gt_start_col_name = col_name_dict.get('gt_start_col_name', '')
    gt_end_col_name = col_name_dict.get('gt_end_col_name', '')
    if gt_start_col_name and gt_end_col_name:
        assert len(df[gt_start_col_name]) == len(df[gt_end_col_name]), \
            'Something is wrong with the start and stop of the annotations. Please verify'

        for ind, (start, end) in enumerate(zip(df[gt_start_col_name], df[gt_end_col_name])):
            if (isinstance(start, str)) and (isinstance(end, str)):
                start = datetime.datetime.strptime(start, '%H:%M:%S').time()
                end = datetime.datetime.strptime(end, '%H:%M:%S').time()

            if (isinstance(start, datetime.time)) and (isinstance(end, datetime.time)):
                start_date = pd.to_datetime(df.index[ind]).replace(hour=start.hour, minute=start.minute,
                                                                   second=start.second)
                end_date = pd.to_datetime(df.index[ind]).replace(hour=end.hour, minute=end.minute,
                                                                 second=end.second)

                # Add an annotation with an arrow between points
                fig.add_shape(type='line',x0=start_date, y0=y_center, x1=end_date, y1=y_center,
                              line=dict(color='red', width=2, dash='solid'))
        # Add a custom legend entry for the line shape
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='red', width=2, dash='solid'), name='GT: Start-stop'))


    # Update layout and show the plot
    fig.update_layout(height=600, width=800, showlegend=True)

    plot_name = 'temp_graph_f{}'.format(dict_for_plot_title['file'][0])
    if batch_id:
        plot_name = plot_name + '_batch{:02d}'.format(batch_id)
    if my_model:
        plot_name = plot_name + '_{}_{}'.format(my_model[0], my_model[1])


    fig.write_html(os.path.join('plotly',plot_name+'.html'))


def plot_data(df, col_name_dict, dict_for_plot_title):
    '''Creates a simple figure based on a dataframe input
    :param df:
    :param col_name_dict:
    :param dict_for_plot_title:
    :return:
    '''

    #fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10, 6),sharex=True)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    grap_title = 'File: {}\nTemp over Time: Batch={}'.format(dict_for_plot_title['file'][1],dict_for_plot_title['batch'])
    ax1.set_title(grap_title)

    # Plotting the 'Celcius' column against the timestamp index
    ax1.plot(df.index, df[col_name_dict['temp_col_name']], marker='o', linestyle='-', color='blue')  # Adjust color if needed

    # Customize ax1 plot
    ax1.set_ylabel('Celcius')
    ax1.grid(True)

    y_min,y_max = ax1.get_ylim()

    # Plotting ground truth on ax2 in the case the ground truth column does exist
    gt_col_name = col_name_dict.get('gt_col_name','')
    if gt_col_name:
        # Plotting rectangles indicating when the pump is 'on'
        on_intervals = []
        on_status = False
        bin_size = df.index[1]-df.index[0]

        for i, status in enumerate(df[gt_col_name].fillna(0)):
            if status == 1 and not on_status:
                start = df.index[i]
                on_status = True
            elif status == 0 and on_status:
                end = df.index[i-1]
                on_status = False
                on_intervals.append((start, end))

        # the case where the last measurement was annotated as pump on (need to add on_intervals)
        if on_status:
            on_intervals.append((start,start))

        for start, end in on_intervals:
            rect = Rectangle((start - pd.Timedelta(bin_size / 2), y_min),
                             (end - start + pd.Timedelta(bin_size)), y_max-y_min, color='red', alpha=0.3)
            ax1.add_patch(rect)

        # Adding the annotaion on a seperate axis (can be removed later on)
        #ax2.plot(df.index, df[gt_col_name], marker='o', linestyle='-', color='red')  # Adjust color if needed
        #ax2.set_title('Pump Status')
        #ax2.set_ylabel('Ground Truth')
        #ax2.set_xlabel('Timestamp')
        #ax2.grid(True)

    gt_start_col_name = col_name_dict.get('gt_start_col_name','')
    gt_end_col_name = col_name_dict.get('gt_end_col_name','')
    if gt_start_col_name and gt_end_col_name:
        assert len(df[gt_start_col_name]) == len(df[gt_end_col_name]), \
            'Something is wrong with the start and stop of the annotations. Please verify'

        y_center = (y_max+y_min)/2
        for ind, (start, end) in enumerate(zip(df[gt_start_col_name], df[gt_end_col_name])):
            if (isinstance(start, str)) and (isinstance(end, str)):
                start = datetime.datetime.strptime(start, '%H:%M:%S').time()
                end = datetime.datetime.strptime(end, '%H:%M:%S').time()

            if (isinstance(start, datetime.time)) and (isinstance(end, datetime.time)):
                start_date = pd.to_datetime(df.index[ind]).replace(hour=start.hour, minute=start.minute, second=start.second)
                end_date = pd.to_datetime(df.index[ind]).replace(hour=end.hour, minute=end.minute, second=end.second)
                ax1.annotate('', xy=(start_date, y_center), xytext=(end_date, y_center),
                         arrowprops=dict(facecolor='red', arrowstyle='|-|', alpha=0.3))
    plt.tight_layout()
    batch_id = dict_for_plot_title.get('batch','')
    if batch_id:
        plot_name = 'temp_graph_f{}_batch{:02d}.png'.format(dict_for_plot_title['file'][0], batch_id)
    else:
        plot_name = 'temp_graph_f{}.png'.format(dict_for_plot_title['file'][0])

    plt.savefig(os.path.join('plots',plot_name))

def plot_evaluation(res):
    '''
    Comparing the different evaluation metrices for the different models
    :param res: is a dict containing the results of the different models
    :return:
    '''

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    markers = ['+', 'o']  # Markers for train and test data
    colors = ['blue', 'green', 'red']  # Different colors for each model

    models = list(res.keys())
    metrics = list(res[models[0]].keys())
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for idx, metric in enumerate(metrics):
        ax = axs[positions[idx]]
        for i, (model, values) in enumerate(res.items()):
            train_value, test_value = values[metric]
            ax.plot(i, train_value, marker='+', color=colors[i], label=f'{model} Train', markersize=8)
            ax.plot(i, test_value, marker='o', color=colors[i], label=f'{model} Test', markersize=8)

        ax.set_xticks(range(len(res)))
        ax.set_xticklabels(res.keys()) #, rotation=45)
        ax.set_title(metric)
        ax.set_ylim(0, 1)  # Set y-limits to 0 and 1
        ax.grid()

        # Remove x-labels for subplots at the top
        if idx < 2:
            ax.set_xticklabels([])

        # Remove legend from subplots
        ax.legend().remove()

    # Create a common legend outside the subplots
    #handles, labels = ax.get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper right')

    # Shrink current axis
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.96, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(os.path.join('plots','evaluation.png'))
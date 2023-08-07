import pickle

import pandas as pd
import xgboost as xgb
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    file_to_load = 'site261_new_XGB_0'

# to open the pickle file that contains the model and use the pretrained  to predict
    with open(os.path.join('output',file_to_load+'.pkl'), 'rb') as file:
        objects_loaded = pickle.load(file)

        # extract the objects from the dictionary
        loaded_model = objects_loaded['model']
        loaded_X_train = objects_loaded['X_train']
        loaded_y_train = objects_loaded['y_train']
        loaded_X_test = objects_loaded['X_test']
        loaded_y_test = objects_loaded['y_test']
    train_dmatrix = xgb.DMatrix(loaded_X_train, loaded_y_train)
    test_dmatrix = xgb.DMatrix(loaded_X_test, loaded_y_test)

    df_train = pd.DataFrame({'target': loaded_y_train, 'y_pred': loaded_model.predict(train_dmatrix)})
    df_test = pd.DataFrame({'target': loaded_y_test, 'y_pred': loaded_model.predict(test_dmatrix)})

    # calculate the residual between the actual and predicted time series from training
    residual_train = df_train['target'] - df_train['y_pred']
    residual_test = df_test['target'] - df_test['y_pred']

    # calculate the mean and standard deviation of the residual
    mean = np.mean(residual_test)
    std_dev = np.std(residual_test)

    # define the desired confidence level
    confidence_level = 0.995  # 95% confidence level

    # calculate the upper and lower bounds of the confidence interval
    z = norm.ppf((1 + confidence_level) / 2)
    upper_bound = mean + z * std_dev
    lower_bound = mean - z * std_dev

    # define anomalies as points in time where the residual falls outside the confidence interval
    anomalies = np.where((residual_test > upper_bound) | (residual_test < lower_bound))[0]
    df_anomalies = df_test.iloc[anomalies]

    # find the row with the minimum value of y in this dataframe
    min_row = df_anomalies.loc[df_anomalies['target'].idxmin()]

    fig, ax = plt.subplots()
    df_test.plot(y=['target','y_pred'], ax=ax)
    # calculate the y-range of the plot
    y_min, y_max = ax.get_ylim()
    line_length = 0.1
    line_height = (y_max - y_min) * line_length
    # iterate over the tick indices and add ticklines
    for index in anomalies:
        # add a horizontal line to the plot at the y-position of the tickline
        ax.axvline(x=df_test.index[index], ymax=line_height, color='red', linestyle='--', linewidth=0.5)
    ax.set_title(file_to_load)
    plt.show()
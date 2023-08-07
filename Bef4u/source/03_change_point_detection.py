import pickle
import pandas as pd
import xgboost as xgb
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from utils import get_data_X_y
import os
from statsmodels.tsa.seasonal import seasonal_decompose


def time_serie_decomposition(df, ch_dict):
    # we have each 4 measurement/hour, ==> 24x4=96
    period = 96
    model = 'additive'

    df = df.dropna()
    result = seasonal_decompose(df['y'], model=model, period=period)
    # extract the trend, seasonal, and residual components
    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid

    # calculate the mean and standard deviation of the residual component
    residual_mean = residual.mean()
    residual_std = residual.std()

    # calculate the z-scores of the residual component
    z_scores = (residual - residual_mean) / residual_std

    # detect anomalies based on a threshold of 3 standard deviations
    anomalies = z_scores[abs(z_scores) > 3]

    # plot the original time-series data, the trend, the seasonal component, and the residual component
    fig, ax = plt.subplots(nrows=4, figsize=(12, 8), sharex=True)

    ax[0].plot(df['y'])
    ax[0].set_ylabel('Original')
    ax[0].grid()
    ax[1].plot(trend)
    ax[1].set_ylabel('Trend')
    ax[1].grid()
    ax[2].plot(seasonal)
    ax[2].set_ylabel('Seasonal')
    ax[2].grid()
    ax[3].plot(residual)
    ax[3].plot(z_scores)
    ax[3].set_ylabel('Res/z-score')
    ax[3].grid()

    title_txt = "Time Series Decomposition - Site: {}, Meter: {}, \nPeriod: {}, Model: {} ".format(
        ch_dict['site'], ch_dict['target'], period, model)
    ax[0].set_title(title_txt)

    # plot the detected anomalies in red
    ax[0].plot(anomalies.index, anomalies, 'ro')
    ax[3].plot(anomalies.index, anomalies, 'ro')
    figure_out = os.path.join('output', 'TSD_site' + str(ch_dict['site']) +
                              '_target' + str(ch_dict['target']) +
                              '_period' + str(period) +
                              '_model(' + model + ')' +
                              '.png')
    plt.savefig(figure_out)
    plt.show()
    plt.close()


def statistical_control_chart(df, ch_dict):
    window_size = ch_dict['window_size']
    control_factor = ch_dict['control_factor']
    rolling_mean = df['y'].rolling(window_size).mean()
    rolling_std = df['y'].rolling(window_size).std()

    # Calculate the upper and lower control limits
    upper_limit = rolling_mean + control_factor * rolling_std
    lower_limit = rolling_mean - control_factor * rolling_std

    # Identify the anomalies
    anomalies = df[(df['y'] > upper_limit) | (df['y'] < lower_limit)]

    # Plot the results
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(df.index, df['y'], color='blue', label='Process variable')
    ax.plot(rolling_mean.index, rolling_mean, color='green', label='Rolling mean')
    ax.plot(upper_limit.index, upper_limit, color='red', linestyle='--', label='Upper control limit')
    ax.plot(lower_limit.index, lower_limit, color='red', linestyle='--', label='Lower control limit')
    ax.scatter(anomalies.index, anomalies['y'], color='red', label='Anomalies')
    ax.legend()
    title_txt = "Statistical Control Chart - Site: {}, Meter: {}, \nWindow: {}, Control_factor: {} ".format(
        ch_dict['site'], ch_dict['target'], window_size, control_factor)
    ax.set_title(title_txt)
    figure_out = os.path.join('output', 'SCC_site' + str(ch_dict['site']) +
                              '_target' + str(ch_dict['target']) +
                              '_win(' + str(ch_dict['window_size']) + ')' +
                              '_control_f(' + str(control_factor) + ')' +
                              '.png')
    plt.savefig(figure_out)
    plt.show()
    plt.close()


def plot_change_point_result(df, ch_dict):
    control_factor = ch_dict['control_factor']
    df.loc[:, 'z_score_new'] = (df['y_inputed_std']-df['y_pred']-df['win_test_res_mu'])/df['win_test_res_std']
    anomalies = np.where(df['z_score'] > control_factor)[0]
    inputed = np.where(df['y'] != df['y_inputed_std'])[0]
    inputed_detected = list(set(inputed).intersection(set(np.where(abs(df['z_score_new']) > control_factor)[0])))
    inputed_not_detected = list(set(inputed).intersection(set(np.where(abs(df['z_score_new']) <= control_factor)[0])))
    acc = len(inputed_detected)/(len(inputed_detected)+len(inputed_not_detected))

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16, 8), sharex=True)
    # plot each data series on a different subplot
    ax1.plot(df.index, df['y_inputed_std'])
    ax1.plot(df.index, df['y_pred'])
    ax2.plot(df.index, df['z_score_new'])

    # set the plot titles and axis labels
    ax1.set_ylabel('Meter')
    ax2.set_ylabel('z-score')
    ax2.set_xlabel('Timestamp')
    ax1.legend(['y', 'y_pred'])
    ax1.grid()
    ax2.grid()
    y_min, y_max = ax1.get_ylim()
    line_length = 0.1
    line_height = (y_max - y_min) * line_length
    for index in anomalies:
        ax1.axvline(x=df.index[index], ymax=line_height, color='blue', linestyle='--', linewidth=0.5)
    for index in inputed_detected:
        ax1.axvline(x=df.index[index], ymax=line_height, color='green', linestyle='--', linewidth=0.5)
    for index in inputed_not_detected:
        ax1.axvline(x=df.index[index], ymax=line_height, color='red', linestyle='--', linewidth=0.5)

    y_min, y_max = ax2.get_ylim()
    line_length = 0.1
    line_height = (y_max - y_min) * line_length
    for index in anomalies:
        # add a horizontal line to the plot at the y-position of the tickline
        ax2.axvline(x=df.index[index], ymax=line_height, color='blue', linestyle='--', linewidth=1)
    for index in inputed_detected:
        ax2.axvline(x=df.index[index], ymax=line_height, color='green', linestyle='--', linewidth=1)
    for index in inputed_not_detected:
        ax2.axvline(x=df.index[index], ymax=line_height, color='red', linestyle='--', linewidth=1)

    title_txt = "Site: {}, Meter: {}, Window: ({}, {}) ".format(
        ch_dict['site'], ch_dict['target'], ch_dict['window_size'], ch_dict['test_size'])
    ax1.set_title(title_txt)
    title_txt = "Anomaly: z-score > {}, Acc={:.2f},\ninputed std: {}, at interval=100".format(
        control_factor, acc, ch_dict['inpute_std'], ch_dict['inpute_each'])
    ax2.set_title(title_txt)
    # show the plot
    plt.subplots_adjust(hspace=0.3)

    figure_out = os.path.join('output', 'CPD_site' + str(ch_dict['site']) +
                              '_target' + str(ch_dict['target']) +
                              '_win(' + str(ch_dict['window_size']) + ',' + str(ch_dict['test_size']) + ')' +
                              '_inpute(' + str(ch_dict['inpute_each']) + ',' + str(ch_dict['inpute_std']) + ')' +
                              '.png')
    plt.savefig(figure_out)
    plt.close()


def run_change_point_detection(ch_dict):
    site, target_id = ch_dict['site'], ch_dict['target']
    window_size, test_size = ch_dict['window_size'], ch_dict['test_size']

    X_train, y_train, X_test, y_test, ev_name_mapper, meters_name_mapper = get_data_X_y(site, target_id)

    # to test you can remove the comment of the next line (so that it does not run on the whole data file)
    #y_test = y_test.iloc[:850]
    y_test_std = y_test.std()

    # Creation of empty DataFrame with the same index as the test data
    time_stamps = y_test.index
    df_out = pd.DataFrame(data=[], index=time_stamps, columns=[])


    # Loop through the time series with the rolling window
    len_y_test = len(y_test)
    for i, ii in enumerate(range(window_size, len_y_test)):
        if i % ch_dict['print_each'] == 0:
            print("{}/{}".format(i+window_size, len_y_test))
        # Get the current window
        X_window_train = X_test.iloc[i:ii-test_size]
        y_window_train = y_test[i:ii-test_size]
        X_window_test = X_test.iloc[ii-test_size:ii]
        y_window_test = y_test[ii-test_size:ii]
        model = xgb.XGBRegressor()
        model.fit(X_window_train, y_window_train)

        df_window = pd.DataFrame({'y_test_window': y_test[i:ii], 'y_test_window_pred': model.predict(X_test.iloc[i:ii])})

        # Use the model to predict the next data point
        X_next = np.array([X_test.iloc[ii]])
        y_pred = model.predict(X_next)

        # Calculate the residual
        residual = y_test[ii] - y_pred
        if i % ch_dict['inpute_each'] == 0:
            if residual > 0:
                y_inputed = y_test[ii] + ch_dict['inpute_std']*y_test_std
            else:
                y_inputed = y_test[ii] - ch_dict['inpute_std']*y_test_std
        else:
            y_inputed = y_test[ii]

        # Compute the Z-score of the residual within the window
        window_test_residuals = y_window_test - model.predict(X_window_test)
        z_score = (residual - np.mean(window_test_residuals)) / np.std(window_test_residuals)
        df_out.loc[time_stamps[ii],
        ['y', 'y_pred', 'residual', 'win_test_res_mu', 'win_test_res_std', 'z_score', 'y_inputed_std']] = \
            [y_test[ii], y_pred[0], residual[0], np.mean(window_test_residuals), np.std(window_test_residuals),
             z_score[0], y_inputed]

    with open(pickle_out, 'wb') as file:
        pickle.dump(df_out, file)

if __name__ == "__main__":

    # dict contaiing all the neceswsary variables to run Change Point Detection
    #for ws in [400, 600, 1000, 1200]:
    #    for ts in [50, 100]:
    for ws in [1200]:
        for ts in [100]:
            print(ws, ts)
            print('**************')

            ch_dict = {'site': '261_new',
                       'target': 0,
                       'window_size': ws,
                       'test_size': ts,
                       'inpute_each': 100,
                       'inpute_std': 3,
                       'control_factor': 2,
                       'print_each': 1000}

            pickle_out = os.path.join('output', 'CPD_site' + str(ch_dict['site']) +
                                      '_target' + str(ch_dict['target']) +
                                      '_win(' + str(ch_dict['window_size']) + ',' + str(ch_dict['test_size']) + ')' +
                                      '_inpute(' + str(ch_dict['inpute_each']) + ',' + str(ch_dict['inpute_std']) + ')' +
                                      '.pkl')
            ch_dict.update({'pickle_out': pickle_out})
            # comment next line out when pickle exists (no need to run)
            run_change_point_detection(ch_dict)
            # to open the pickle file that contains the result of the run_change_detection
            with open(pickle_out, 'rb') as file:
                df = pickle.load(file)
            plot_change_point_result(df, ch_dict)
            time_serie_decomposition(df, ch_dict)
            statistical_control_chart(df, ch_dict)



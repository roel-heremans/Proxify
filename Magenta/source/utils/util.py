
import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from scipy.signal import find_peaks

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
    temperature = df['Temp_smooth']

    maxima_indices, _ = find_peaks(temperature, prominence=0.1, distance=4)
    minima_indices, _ = find_peaks(-temperature, prominence=0.1, distance=5)

    plt.plot(df['Temp_smooth'])
    plt.plot(df.iloc[maxima_indices]['Temp_smooth'],marker='o',color='r', linestyle='')
    plt.plot(df.iloc[minima_indices]['Temp_smooth'],marker='o',color='g', linestyle='')
    plt.show()
    # Ensure alternating pattern: alternating_max_min contains alternating maxima and minima
    alternating_max_min = []
    if len(maxima_indices) > 0:
        alternating_max_min.append(maxima_indices[0])
    for i in range(1,min(len(minima_indices), len(maxima_indices))):
        alternating_max_min.append(minima_indices[i-1])
        alternating_max_min.append(maxima_indices[i])

    alternating_max_min = np.sort(alternating_max_min)

    #plt.plot(df['Temp_smooth'])
    #plt.plot(df.iloc[alternating_max_min[0::2]]['Temp_smooth'],marker='o',color='r', linestyle='')
    #plt.plot(df.iloc[alternating_max_min[1::2]]['Temp_smooth'],marker='o',color='g', linestyle='')
    #plt.show()



    # Assign the calculated values to the DataFrame column 'Temp_extrema'
    df['Temp_extrema'] = 1
    df['Temp_extrema'].iloc[alternating_max_min[0::2]] = 2
    df['Temp_extrema'].iloc[alternating_max_min[1::2]] = -2
    df['Temp_extrema'].iloc[0:alternating_max_min[0]] = 1

    for val1,val2 in zip(alternating_max_min[0::2], alternating_max_min[1::2]):
        df['Temp_extrema'].iloc[val1+1:val2] = -1

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


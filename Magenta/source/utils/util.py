
import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import plotly.graph_objects as go


def create_output_directories():
    ''' Checks if the directories are already existing, if not existing they will get created
    '''
    to_create_paths = ['plots', 'plotly']
    for my_path in to_create_paths:
        if not os.path.exists(os.path.join(os.getcwd(), my_path)):
            # Create the directory
            os.makedirs(os.path.join(os.getcwd(), my_path))


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

    col_name_dict = {'temp_col_name': temp_col_name}
    col_name_optional_dict = {'gt_col_name': gt_col_name,
                              'gt_start_col_name':gt_start_col_name,
                              'gt_end_col_name': gt_end_col_name,
                              'batch_col_name': batch_col_name}

    for key,value in col_name_optional_dict.items():
        if value:   # only when non-empty string, the col_name_dict gets updated
            col_name_dict.update({key: value})

    return df, col_name_dict


def plotly_data(df, col_name_dict, dict_for_plot_title):
    fig = go.Figure()

    # Plotting the 'Celcius' column against the timestamp index
    fig.add_trace(go.Scatter(x=df.index, y=df[col_name_dict['temp_col_name']], mode='markers+lines', name='Temperature', line=dict(color='blue')))

    # Customize plot title
    batch_id = dict_for_plot_title['batch']
    file_id = dict_for_plot_title['file'][0]
    file_str = dict_for_plot_title['file'][1]
    grap_title = 'File{}: {}<br>Temp over Time: Batch={}'.format(file_id, file_str, batch_id)
    fig.update_layout(title=grap_title, xaxis_title='Timestamp', yaxis_title='Celcius', hovermode='x')

    # Plotting rectangles indicating when the pump is 'on'
    gt_col_name = col_name_dict.get('gt_col_name', '')
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
            fig.add_shape(type="rect", x0=start - pd.Timedelta(bin_size / 2), y0=min(df[col_name_dict['temp_col_name']]),
                          x1=end + pd.Timedelta(bin_size), y1=max(df[col_name_dict['temp_col_name']]),
                          line=dict(color="red", width=1), fillcolor="red", opacity=0.3)
        # Add a custom legend entry for the line shape
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color="red", width=10), fillcolor="red", opacity=0.3, name='GT: On-Off'))


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
    plot_name = 'temp_graph_f{}_batch{:02d}.html'.format(dict_for_plot_title['file'][0], batch_id)
    fig.write_html(os.path.join('plotly',plot_name))


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
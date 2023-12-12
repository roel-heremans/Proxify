import dash
import pandas as pd
import numpy as np
from itertools import groupby
from scipy.signal import find_peaks
from dash import dcc, html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import os


config_dict = {'smooth_factor': 5,
               'resample_string': '1T',
               'dist_for_maxima': 3,
               'dist_for_minima': 3,
               'peak_prominence': 0.1,
               'timestamp_col_name':'DateTime_EAT',
               'temp_col_name':'Celsius',
               'gt_col_name':'Use_event',
               'detect_start_time': '08:00:00',
               'detect_stop_time': '18:00:00'}

def get_alternating_values(maxima_indices, minima_indices):
    merged_indices_res = []
    source_info = []

    max_idx = 0
    min_idx = 0

    while max_idx < len(maxima_indices) and min_idx < len(minima_indices):
        max_val = maxima_indices[max_idx]
        min_val = minima_indices[min_idx]

        if min_val <= max_val:
            merged_indices_res.append(min_val)
            source_info.append('min')
            min_idx += 1
        else:
            merged_indices_res.append(max_val)
            source_info.append('max')
            max_idx += 1

    # Add the remaining elements from either list
    merged_indices_res.extend(maxima_indices[max_idx:])
    source_info.extend(['max'] * (len(maxima_indices) - max_idx))

    merged_indices_res.extend(minima_indices[min_idx:])
    source_info.extend(['min'] * (len(minima_indices) - min_idx))

    successive_indices = []

    # Loop through the indices to find successive elements
    for i in range(len(source_info) - 1):
        if source_info[i] == source_info[i + 1]:
            successive_indices.append(i+1)

    for i in successive_indices[::-1]:
        removed = merged_indices_res.pop(i)
        print('removed idx: {}, value: {}'.format(i, removed))

    if merged_indices_res[0] == maxima_indices[0]:
        max_is_first = 1
    else:
        max_is_first = 0

    # check if the last is a maximum:  if so remove it from the list
    if merged_indices_res[-1] == maxima_indices[-1]:
        merged_indices_res.pop()

    return merged_indices_res, max_is_first

def getting_extrema(temperatures, config_dict):

    # the following is/was a variable that can be set to 1 to try the list of masks (but did not work so far)
    create_list = 0

    # selection of the start and stop times for which the extrema need to be considered
    start_time = pd.to_datetime(config_dict['detect_start_time']).time()
    stop_time = pd.to_datetime(config_dict['detect_stop_time']).time()

    if create_list:
        # Get unique dates from the DataFrame
        unique_dates = pd.Series(temperatures.index).dt.date.unique()
        print('unique dates')
        print(unique_dates)

        #creating a list of masks per unique date
        mask_list = []

        # Create masks for each unique date from the DataFrame based on start and stop times
        for date in unique_dates:
            start_datetime = pd.to_datetime(str(date) + ' ' + str(start_time))
            stop_datetime = pd.to_datetime(str(date) + ' ' + str(stop_time))

            mask = (temperatures.index >= start_datetime) & (temperatures.index <= stop_datetime)
            mask_list.append(mask)

            # Set 'Temp_extrema' values to zero where the index falls outside the interval
            temperatures.loc[mask] = temperatures.min()
        print('temperatures')
        print(temperatures)

    if not create_list:
        # Create a mask for values outside the time interval
        mask = (temperatures.index.time < start_time) | (temperatures.index.time > stop_time)
        print('mask')
        print(mask)

    # Set 'Temp_extrema' values to zero where the index falls outside the interval
    #temperatures.loc[mask] = temperatures.min()
    df = temperatures.to_frame().copy()

    maxima_indices, _ = find_peaks(temperatures,  distance=config_dict['dist_for_maxima'],
                                   prominence=config_dict['peak_prominence'])
    minima_indices, _ = find_peaks(-temperatures,  prominence=config_dict['peak_prominence'],
                                   distance=config_dict['dist_for_minima'])


    print('max_indices')
    print(maxima_indices)

    print('min_indices')
    print(minima_indices)


    alternating_max_min, max_is_first = get_alternating_values(maxima_indices, minima_indices)

    alternating_temperatures = temperatures[alternating_max_min]

    if len(alternating_temperatures) >= 2:
        # find out if the first element in the alternating_temperatures is a min or a max by comparing its value with the
        # next one.
        first_is_maximum = False
        if alternating_temperatures[0] > alternating_temperatures[1]:
            first_is_maximum = True


        if first_is_maximum:
            df['Temp_extrema'] = 1
            for val1,val2 in zip(alternating_temperatures.index[0::2], alternating_temperatures.index[1::2]):
                df.loc[val1:val2, 'Temp_extrema'] = -1
            df.loc[alternating_temperatures.index[0::2], 'Temp_extrema'] = 2
            df.loc[alternating_temperatures.index[1::2], 'Temp_extrema'] = -2
        else:
            df['Temp_extrema'] = -1
            for val1,val2 in zip(alternating_temperatures.index[0::2], alternating_temperatures.index[1::2]):
                df.loc[val1:val2, 'Temp_extrema'] = 1
            df.loc[alternating_temperatures.index[0::2], 'Temp_extrema'] = -2
            df.loc[alternating_temperatures.index[1::2], 'Temp_extrema'] = 2
    else:
        df['Temp_extrema'] = 1

    if create_list:
        # setting the Temp_extrema value to 0 outside the selected (start-stop)-times
        for mask in mask_list:
            df.loc[mask,'Temp_extrema'] = 0
    else:
        df.loc[mask,'Temp_extrema'] = 0

    return df['Temp_extrema']

def get_data(filename, config_dict):

    df, col_name_dict = import_xlsx_to_df(filename,
                                          timestamp_col_name=config_dict['timestamp_col_name'],
                                          temp_col_name=config_dict['temp_col_name'],
                                          gt_col_name=config_dict['gt_col_name'])

    # Rename columns
    col_mapping = {col_name_dict['temp_col_name']: 'Temperature',
                   col_name_dict['gt_col_name']: 'GroundTruth'}
    df= df.rename(columns=col_mapping)
    df = resample_df(df, resample=config_dict['resample_string'])

    df['Temp_smooth'] = df['Temperature'].rolling(window=config_dict['smooth_factor']).mean()
    df['Temp_smooth'].fillna(method='bfill', inplace=True)

    df_extrema = getting_extrema(df['Temp_smooth'], config_dict)
    df = pd.concat([df, df_extrema], axis=1)

    return df

def get_plotly_fig(df, config_dict):

    # Create the base line plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Temperature'], mode='lines', name='Temp Unaltered'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Temp_smooth'], mode='lines', name='Temp Smoothed'))
    loc_max = df[df["Temp_extrema"] == 2].copy()
    fig.add_trace(go.Scatter(x=loc_max.index, y=loc_max['Temp_smooth'], mode='markers', name='Local Maxima',
                             marker=dict(color='red', symbol='circle')))
    loc_min = df[df["Temp_extrema"] == -2].copy()
    fig.add_trace(go.Scatter(x=loc_min.index, y=loc_min['Temp_smooth'], mode='markers', name='Local Minima',
                             marker=dict(color='green', symbol='circle')))

    # pump is on when the Temp_extrema is between a maxima and a minima
    resolution_min = int(config_dict['resample_string'][:-1])
    bin_width = pd.Timedelta(minutes=resolution_min)/2
    min_temp = df['Temp_smooth'].min()

    y_pred = df['Temp_extrema']==-1
    y_pred_on = y_pred.loc[y_pred]
    if len(y_pred_on) > 0:
        for start,stop in get_start_stops(y_pred_on):
            bin_start = start - bin_width / 2
            bin_end = stop + bin_width / 2
            fig.add_shape(type="rect", x0=bin_start, y0=min_temp-0.5, x1=bin_end, y1=min_temp,
                          line=dict(color='blue', width=1), fillcolor='blue', opacity=0.8)


    min_temp -=0.6
    y = df['GroundTruth'].dropna()
    if len(y)> 0:
        for start,stop in get_start_stops(y):
            bin_start = start - bin_width / 2
            bin_end = stop + bin_width / 2
            fig.add_shape(type="rect", x0=bin_start, y0=min_temp-0.5, x1=bin_end, y1=min_temp,
                          line=dict(color='red', width=1), fillcolor='red', opacity=0.8)

    # Add invisible traces for legend
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                             marker=dict(color='red', opacity=0.8,  symbol='square', size=10), name='Ground Truth'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                             marker=dict(color='blue', opacity=0.8, symbol='square', size=10), name='Prediction'))
    return fig

def get_start_stops(y):

    groups = []
    for k, g in groupby(enumerate(y.index), lambda ix: ix[0] - ix[1].minute):
        timestamps = [ix[1] for ix in g]
        if len(timestamps) == 1:
            groups.append((timestamps[0], timestamps[0]))
        else:
            groups.append((timestamps[0], timestamps[-1]))

    return groups


def import_xlsx_to_df(filename,
                      timestamp_col_name='DateTime_EAT',
                      temp_col_name='Celsius',
                      gt_col_name='',
                      batch_col_name=''):
    '''
    Reads in the excel datafiles containing at least:
    -a column that reflects on the timestamp (formated like 2023-01-06 12:21:26)
    -a column containing the temperatures
    -optional: a column containing the annotation if the pump was on or off during that timestamp
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
                              'batch_col_name': batch_col_name}

    for key,value in col_name_optional_dict.items():
        if value:   # only when non-empty string, the col_name_dict gets updated
            col_name_dict.update({key: value})

    return df[col_name_dict.values()], col_name_dict

def resample_df(df, resample='1T'):
    # Ensure the index is in DateTime format
    df.index = pd.to_datetime(df.index)

    # Resample to 1-minute intervals
    resampled_df = df.resample(resample).ffill()
    return resampled_df



app = dash.Dash(__name__)

data_directory = 'data'
file_names = os.listdir(data_directory)
print('List: {}'.format(file_names))

# Define layout
app.layout = html.Div([
    dcc.Dropdown(
        id='file-dropdown',
        options=[{'label': name, 'value': name} for name in file_names],
        value=file_names[0] if file_names else None
    ),
    html.Hr(),  # Horizontal line for visual separation

    # Input fields for config_dict values
    html.Div([
        html.Div([
            html.Label('Smooth Factor: '),
            dcc.Input(id='smooth-factor', type='number', value=config_dict['smooth_factor'], style={'width': '30px'}),

            html.Label('  Resample String:'),
            dcc.Input(id='resample-string', type='text', value=config_dict['resample_string'], style={'width': '40px'}),
        ], style={'margin-bottom': '20px'}),  # Add bottom margin to create spac
        html.Div([
            html.Label('dist for maxima: '),
            dcc.Input(id='dist-for-maxima', type='number', value=config_dict['dist_for_maxima'], style={'width': '40px'}),
            html.Label('  dist for minima: '),
            dcc.Input(id='dist-for-minima', type='number', value=config_dict['dist_for_minima'], style={'width': '40px'}),
            html.Label('  prominence: '),
            dcc.Input(id='peak-prominence', type='number', value=config_dict['peak_prominence'], style={'width': '40px'}),
        ], style={'margin-bottom': '20px'}),
        html.Div([
            html.Label('timestamp col name: '),
            dcc.Input(id='timestamp-col-name', type='text', value=config_dict['timestamp_col_name']),
            html.Label('  temp col name: '),
            dcc.Input(id='temp-col-name', type='text', value=config_dict['temp_col_name']),
            html.Label('  ground truth col name: '),
            dcc.Input(id='gt-col-name', type='text', value=config_dict['gt_col_name']),
        ], style={'margin-bottom': '20px'}),
        html.Div([
            html.Label('detect between start time: '),
            dcc.Input(id='detect-between-start-time', type='text', value=config_dict['detect_start_time']),
            html.Label('  detect between stop time: '),
            dcc.Input(id='detect-between-stop-time', type='text', value=config_dict['detect_stop_time']),
        ], style={'margin-bottom': '20px'}),
    ]),

    # Display the graph and duration output in separate Div elements
    html.Div([ dcc.Graph(id='graph', config={'editable': True, 'editSelection': True}) ]),
    html.Div(id='duration-output'),

])

# Callback to update the graph based on the selected file
@app.callback(
    Output('graph', 'figure'),
    [Input('file-dropdown', 'value'),
     Input('smooth-factor', 'value'),
     Input('resample-string', 'value'),
     Input('dist-for-maxima', 'value'),
     Input('dist-for-minima', 'value'),
     Input('peak-prominence', 'value'),
     Input('timestamp-col-name', 'value'),
     Input('temp-col-name', 'value'),
     Input('gt-col-name', 'value'),
     Input('detect-between-start-time', 'value'),
     Input('detect-between-stop-time', 'value')]
)
def update_graph(selected_file, smooth_factor, resample_string,
                 dist_for_maxima, dist_for_minima, peak_prominence,
                 timestamp_col_name, temp_col_name, gt_col_name, detect_start_time, detect_stop_time):
    if selected_file:
        print('File selected is: {}'.format(selected_file))
        # Update config_dict values based on user input
        config_dict['smooth_factor'] = smooth_factor
        config_dict['resample_string'] = resample_string
        config_dict['dist_for_maxima'] = dist_for_maxima
        config_dict['dist_for_minima'] = dist_for_minima
        config_dict['peak_prominence'] = peak_prominence
        config_dict['timestamp_col_name'] = timestamp_col_name
        config_dict['temp_col_name'] = temp_col_name
        config_dict['gt_col_name'] = gt_col_name
        config_dict['detect_start_time'] = detect_start_time
        config_dict['detect_stop_time'] = detect_stop_time
        # Read data from the selected file using your read_data function
        # This assumes your read_data function takes the file name and returns a DataFrame
        df = get_data(os.path.join(data_directory, selected_file), config_dict)


        # generate graph
        fig = get_plotly_fig(df, config_dict)

        return fig
    else:
        return {}

# Callback to update the duration output
@app.callback(
    Output('duration-output', 'children'),
    [Input('file-dropdown', 'value'),
     Input('graph', 'selectedData')]
)
def update_duration_output(selected_file, selected_data):
    if selected_file:
        df = get_data(os.path.join(data_directory, selected_file), config_dict)
        data_sampling = (df.index[1]-df.index[0]).total_seconds() // 60
        on_state_duration_full, on_state_count_full, shortest_full, longest_full = calculate_on_state_duration(df)

        duration_output = html.Div([
            html.Label('Original Data Sampling: ', style={'margin-right': '5px'}),
            html.Label(f'{data_sampling} minutes', style={'margin-right': '20px'}),
            html.Br(),
            html.Label('"ON" duration: ', style={'margin-right': '5px'}),
            html.Label(f'{on_state_duration_full} minutes', style={'margin-right': '20px'}),
            html.Label('count: ', style={'margin-right': '5px'}),
            html.Label(f'{on_state_count_full} times', style={'margin-right': '20px'}),
            html.Label('Shortest: ', style={'margin-right': '5px'}),
            html.Label(f'{shortest_full} minutes', style={'margin-right': '20px'}),
            html.Label('Longest: ', style={'margin-right': '5px'}),
            html.Label(f'{longest_full} minutes'),
            #html.Br(),
            #html.Label('"ON" duration (Box Selected Range): '),
            #html.Label(f'{on_state_duration_range} minutes')
        ])
        if selected_data:
            selected_points = selected_data['points']
            x_values = [point['x'] for point in selected_points]

            x_values_min = min(x_values)
            x_values_max = max(x_values)
            if pd.to_datetime(x_values_min).date() == pd.to_datetime(x_values_max).date():
                x_values_max = pd.to_datetime(x_values_max).time()

            # Get the data within the selected range
            df_selected = df[(df.index >= min(x_values)) & (df.index <= max(x_values))]
            on_state_duration_range, on_state_count_range, shortest, longest = calculate_on_state_duration(df_selected)

            # Modify duration_output if selected_data is present
            duration_output = html.Div([
                html.Label('Original Data Sampling: ', style={'margin-right': '5px'}),
                html.Label(f'{data_sampling} minutes', style={'margin-right': '20px'}),
                html.Br(),
                html.Label('"ON" duration: ', style={'margin-right': '5px'}),
                html.Label(f'{on_state_duration_full} minutes', style={'margin-right': '20px'}),
                html.Label('count: ', style={'margin-right': '5px'}),
                html.Label(f'{on_state_count_full} times', style={'margin-right': '20px'}),
                html.Label('Shortest: ', style={'margin-right': '5px'}),
                html.Label(f'{shortest_full} minutes', style={'margin-right': '20px'}),
                html.Label('Longest: ', style={'margin-right': '5px'}),
                html.Label(f'{longest_full} minutes'),
                html.Br(),
                html.Br(),
                html.Label('Box Selected Range: {} - {}'.format(x_values_min, x_values_max)),
                html.Br(),
                html.Label('"ON" duration: ', style={'margin-right': '5px'}),
                html.Label(f'{on_state_duration_range} minutes', style={'margin-right': '20px'}),
                html.Label('count: ', style={'margin-right': '5px'}),
                html.Label(f'{on_state_count_range} times', style={'margin-right': '20px'}),
                html.Label('Shortest: ', style={'margin-right': '5px'}),
                html.Label(f'{shortest} minutes', style={'margin-right': '20px'}),
                html.Label('Longest: ', style={'margin-right': '5px'}),
                html.Label(f'{longest} minutes'),
            ])

        return duration_output

    else:
        return {}


def calculate_on_state_duration(df):
    # Assuming 'Temp_extrema' column indicates the on state

    on_state = df[df['Temp_extrema'] == -1]
    durations = []

    longest = None
    shortest = None
    print('Calculate_on_state_duration')
    for start, stop in get_start_stops(on_state):
        print(start, stop)
        delta_time = pd.to_datetime(stop) - pd.to_datetime(start)

        # Initialize shortest and longest with first delta_time
        if longest is None or delta_time > longest:
            longest = delta_time
        if shortest is None or delta_time < shortest:
            shortest = delta_time

        durations.append((stop - start).seconds // 60)  # Convert seconds to minutes


    return sum(durations), len(durations), int(shortest.total_seconds() // 60), int(longest.total_seconds() // 60)

if __name__ == '__main__':
    app.run_server(debug=True)
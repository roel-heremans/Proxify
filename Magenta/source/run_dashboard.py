import dash
import pandas as pd
import numpy as np
from itertools import groupby
from scipy.signal import find_peaks
from dash import dcc, html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import os
from utils.dash_utils import get_alternating_values, getting_extrema, get_data, get_plotly_fig, get_start_stops, \
    import_xlsx_to_df, resample_df


config_dict = {'file_dropdown': ' ',
               'smooth_factor': 5,
               'resample_string': '1T',
               'poly_fit_deg': 20,
               'dist_for_maxima': 3,
               'dist_for_minima': 3,
               'peak_prominence': 0.1,
               'res_thres_plus': 0.5,
               'res_thres_minus': -0.5,
               'timestamp_col_name':'DateTime_EAT',
               'temp_col_name':'Celsius',
               'gt_col_name':'Use_event',
               'detect_start_time': '00:00:00',
               'detect_stop_time': '23:59:00'}


app = dash.Dash(__name__)

data_directory = 'data'
file_names = sorted(os.listdir(data_directory))
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
            dcc.Input(id='smooth-factor', type='number', value=config_dict['smooth_factor'], style={'width': '50px'}),
            html.Label('  Resample String:'),
            dcc.Input(id='resample-string', type='text', value=config_dict['resample_string'], style={'width': '40px'}),
            html.Label('  Polynomial fit (degree):'),
            dcc.Input(id='poly_fit_deg', type='number', value=config_dict['poly_fit_deg'], style={'width': '40px'})
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
            html.Label('Threshold Seasonal Residu (+): '),
            dcc.Input(id='res-thres-plus', type='number', value=config_dict['res_thres_plus'], style={'width': '40px'}),
            html.Label('Threshold Seasonal Residu (-): '),
            dcc.Input(id='res-thres-minus', type='number', value=config_dict['res_thres_minus'], style={'width': '40px'}),
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
     Input('poly_fit_deg', 'value'),
     Input('dist-for-maxima', 'value'),
     Input('dist-for-minima', 'value'),
     Input('peak-prominence', 'value'),
     Input('res-thres-plus', 'value'),
     Input('res-thres-minus', 'value'),
     Input('timestamp-col-name', 'value'),
     Input('temp-col-name', 'value'),
     Input('gt-col-name', 'value'),
     Input('detect-between-start-time', 'value'),
     Input('detect-between-stop-time', 'value')]
)
def update_graph(selected_file, smooth_factor, resample_string, poly_fit_deg,
                 dist_for_maxima, dist_for_minima, peak_prominence, res_thres_plus, res_thres_minus,
                 timestamp_col_name, temp_col_name, gt_col_name, detect_start_time, detect_stop_time):

    if selected_file:
        print('update_graph when selected_file')
        print('File selected is: {}'.format(selected_file))
        config_dict['file_dropdown'] = selected_file
        config_dict['smooth_factor'] = smooth_factor
        config_dict['resample_string'] = resample_string
        config_dict['poly_fit_deg'] = poly_fit_deg
        config_dict['dist_for_maxima'] = dist_for_maxima
        config_dict['dist_for_minima'] = dist_for_minima
        config_dict['peak_prominence'] = peak_prominence
        config_dict['res_thres_plus'] = res_thres_plus
        config_dict['res_thres_minus'] = res_thres_minus
        config_dict['timestamp_col_name'] = timestamp_col_name
        config_dict['temp_col_name'] = temp_col_name
        config_dict['gt_col_name'] = gt_col_name
        config_dict['detect_start_time'] = detect_start_time
        config_dict['detect_stop_time'] = detect_stop_time

        # Read data from the selected file using your read_data function
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
     Input('graph', 'selectedData'),
     Input('smooth-factor', 'value'),
     Input('resample-string', 'value'),
     Input('poly_fit_deg', 'value'),
     Input('dist-for-maxima', 'value'),
     Input('dist-for-minima', 'value'),
     Input('peak-prominence', 'value'),
     Input('detect-between-start-time', 'value'),
     Input('detect-between-stop-time', 'value')]
)
def update_duration_output(selected_file, selected_data, smooth_factor, resample_string, poly_fit_deg, dist_for_maxima,
                           dist_for_minima, peak_prominence, detect_start_time, detect_stop_time):
    if selected_file:
        df = get_data(os.path.join(data_directory, selected_file), config_dict)
        data_sampling = (df.index[1]-df.index[0]).total_seconds() // 60
        on_state_duration_full, on_state_count_full, shortest_full, longest_full = \
            recalculate_on_state_duration(df,config_dict)

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

    for start, stop in get_start_stops(on_state):
        delta_time = pd.to_datetime(stop) - pd.to_datetime(start)

        # Initialize shortest and longest with first delta_time
        if longest is None or delta_time > longest:
            longest = delta_time
        if shortest is None or delta_time < shortest:
            shortest = delta_time

        durations.append((stop - start).seconds // 60)  # Convert seconds to minutes


    return sum(durations), len(durations), int(shortest.total_seconds() // 60), int(longest.total_seconds() // 60)

def recalculate_on_state_duration(df, config_dict):
    # Use the updated config_dict to recalculate the values
    on_state_duration_full, on_state_count_full, shortest_full, longest_full = calculate_on_state_duration(df)
    return on_state_duration_full, on_state_count_full, shortest_full, longest_full

if __name__ == '__main__':
    app.run_server(debug=True)
import os
import glob
import warnings

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import timedelta
from plotly.subplots import make_subplots

# Suppress the RankWarning
warnings.filterwarnings('ignore', category=np.RankWarning)

def correct_pump_on_window(time_series, ambient_h2o_dropdown, window_size):

    slopes_before = [np.nan] * window_size
    slopes_after = [np.nan] * window_size

    for i in range(window_size, len(time_series)-window_size):
        slopes_before.append(time_series[i] - time_series[i-window_size])
        slopes_after.append(time_series[i+window_size]-time_series[i])
    diff = np.array(slopes_after) - np.array(slopes_before)


    if ambient_h2o_dropdown == 'amb_gt_h2o':    # during the time that the ambient temperature is cooling down:
                                                # for this case we want to check if there is a breakpoint where the
                                                # slope before is bigger (less negative) then the slope after
                                                # (more negative) the breakpoint (so looking for minima)
        break_points = find_peaks(-diff, prominence=0.5)
    else:   # during the time that the ambient temperature is rising:
            # then the slope after the breakpoint (so looking for maxima)
            # for this case we want to check if there is a breakpoint where the slope before is smaller
        break_points = find_peaks(diff, prominence=0.5)


    if len(break_points):
        break_point_idx = break_points[0]
    else:
        break_point_idx = 0

    return break_point_idx

def create_necessary_paths():
    to_create_paths = [ os.path.join('plotly','dashb'), os.path.join('plotly','season'), 'tables']
    for my_path in to_create_paths:
        if not os.path.exists(os.path.join(os.getcwd(), my_path)):
            # Create the directory
            os.makedirs(os.path.join(os.getcwd(), my_path))

def extract_gui_output(res_dict):

    gui_output ={}

    durations = []
    longest = None
    shortest = None

    for pump_on_id, value in res_dict.items():
        delta_time = pd.to_datetime(value.index[-1]) - pd.to_datetime(value.index[0])

        # Initialize shortest and longest with first delta_time
        if longest is None or delta_time > longest:
            longest = delta_time
        if shortest is None or delta_time < shortest:
            shortest = delta_time

        durations.append((value.index[-1] - value.index[0]).seconds // 60)  # Convert seconds to minutes

    if durations:
        gui_output = {'Nr of pump usage': len(durations),
                      'Total duration (min)': sum(durations),
                      'Shortest (min)': int(shortest.total_seconds() // 60),
                      'Longest (min)': int(longest.total_seconds() // 60)
                      }

    return gui_output

def get_alternating_values(maxima_indices, minima_indices):
    '''
    This function serves to get alternating values between maxima_indices and minima_indices. For instance:
    maxima_indices = [13, 22, 24, 38]
    minima_indices = [15, 27, 42, 53]
    the result would be [13, 15, 22, 27, 38, 42]
    :param maxima_indices: list that contains the indices where the maxima are located
    :param minima_indices: list that contians the indices where the minma are located
    :return: list that contains the alternating indices between max and minima or between min and maxima
    '''
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

    # indices would not correspond anymore to the one mentioned by successive _indices when doing this operation from
    # the beginning to the end, need to do this from last to first, hence [::-1]
    for i in successive_indices[::-1]:
        removed = merged_indices_res.pop(i)
        #print('removed idx: {}, value: {}'.format(i, removed))

    if merged_indices_res[0] == maxima_indices[0]:
        max_is_first = 1
    else:
        max_is_first = 0

    # check if the last is a maximum:  if so remove it from the list
    if max_is_first:
        if merged_indices_res[-1] in maxima_indices:
            merged_indices_res.pop()
    else:
        if merged_indices_res[-1] in minima_indices:
            merged_indices_res.pop()

    return merged_indices_res, max_is_first

def get_bin_width(df, config_dict):
    min_temp = df['Temp_smooth'].min()
    if config_dict['resample_string'][-1] == 'T':
        resolution_min = int(config_dict['resample_string'][:-1])
        bin_width = pd.Timedelta(minutes=resolution_min)
    else:
        print('get_plotly_fig: Code need update to get the correct bin_width (Resample String is something different than minutes "T")')
    return min_temp, bin_width

def get_prediction_result_extrema(df, config_dict):
    result_extrema = {}

    _, bin_width = get_bin_width(df, config_dict)

    if config_dict['ambient_h2o_dropdown'] == 'amb_gt_h2o':  # looking at dropping temperatures for pump ON state
        y_pred = df['Temp_extrema'] == -1
        y_pred_on = y_pred.loc[y_pred]
    else:                                    # amb_lt_h2o    # looking at increasing temperatures for pump ON state
        y_pred = df['Temp_extrema'] == 1
        y_pred_on = y_pred.loc[y_pred]

    if len(y_pred_on) > 0:
        for pump_usage_id, (start,stop) in enumerate(get_start_stops(y_pred_on)):
            bin_start = start - bin_width
            bin_end = stop + bin_width
            new_idx = correct_pump_on_window(df.loc[bin_start:bin_end, 'Temp_smooth'].values, config_dict['ambient_h2o_dropdown'], 5)
            if new_idx.any():
                bin_start += new_idx[0] * bin_width
            result_extrema.update({pump_usage_id: df.loc[bin_start:bin_end,'Temp_smooth']})

    return result_extrema

def get_prediction_result_seasonal(df, config_dict):
    result_seasonal = {}
    _, bin_width = get_bin_width(df, config_dict)
    if config_dict['ambient_h2o_dropdown'] == 'amb_gt_h2o':
        if 'resid' in df:
            y = df['resid'].dropna()
            y = y[y < config_dict['res_thres_minus']]
            if len(y)> 0:
                for pump_usage_id, (start,stop) in enumerate(get_start_stops(y)):
                    bin_start = start - bin_width / 2
                    bin_end = stop + bin_width / 2
                    result_seasonal.update({pump_usage_id: df.loc[bin_start:bin_end,'Temp_smooth']})
    else:
        if 'resid' in df.columns:
            y = df['resid'].dropna()
            y = y[y > config_dict['res_thres_plus']]
            if len(y) > 0:
                for pump_usage_id, (start, stop) in enumerate(get_start_stops(y)):
                    bin_start = start - bin_width / 2
                    bin_end = stop + bin_width / 2
                    result_seasonal.update({pump_usage_id: df.loc[bin_start:bin_end,'Temp_smooth']})
    return result_seasonal

def get_data(filename, config_dict):

    df, col_name_dict = import_xlsx_to_df(filename,
                                          timestamp_col_name=config_dict['timestamp_col_name'],
                                          temp_col_name=config_dict['temp_col_name'],
                                          gt_col_name=config_dict['gt_col_name'])
    original_data_sampling = (df.index[1]-df.index[0]).total_seconds() // 60

    # Rename columns
    col_mapping = {col_name_dict['temp_col_name']: 'Temperature'}
    if 'gt_col_name' in col_name_dict:
        col_mapping.update({col_name_dict['gt_col_name']: 'GroundTruth'})

    df= df.rename(columns=col_mapping)

    # Put sampling frequency of incoming data to the Value written in the Resample String: "1T" means each minute,
    # '5T' each 5 minutes,...
    df = resample_df(df, resample=config_dict['resample_string'])
    df['Temperature'].fillna(method='bfill', inplace=True)
    df['Temperature'].fillna(method='ffill', inplace=True)

    # proxy for the ambient temperature
    polynome = get_poly_fit(np.arange(len(df)),df['Temperature'].values,config_dict['poly_fit_deg'])
    df['poly_fit'] = polynome(np.arange(len(df)))


    df['Temp_smooth'] = df['Temperature'].rolling(window=config_dict['smooth_factor']).mean()

    # Shift the rolling mean by half the window size backward (this is possible because we are analyzing off-line)
    half_window_shift = config_dict['smooth_factor'] // 2
    df['Temp_smooth'] = df['Temp_smooth'].shift(-half_window_shift)
    df['Temp_smooth'].fillna(method='bfill', inplace=True)


    df_extrema = getting_extrema(df['Temp_smooth'], config_dict)

    df = pd.concat([df, df_extrema], axis=1)

    temperature_series = df['Temperature']
    season_period = 24                 # Assuming a daily seasonality
    if len(temperature_series) >= 2 * season_period:
        results = seasonal_decompose(temperature_series, model='additive', period=season_period)  # Assuming a daily seasonality
        df = pd.concat([df, results.trend, results.seasonal, results.resid], axis=1)


    df.reset_index(inplace=True)
    x = df[df['Temp_extrema']==2].index
    y = df[df['Temp_extrema']==2]['Temp_smooth'].values
    if (len(x) > 0) & (len(y) > 0):
        convex_envelope_max = np.interp(np.arange(df.shape[0]), x, y)
        df['convex_envelope_max'] = convex_envelope_max
        polynome = get_poly_fit(np.arange(len(df)),df['convex_envelope_max'].values,config_dict['poly_fit_deg'])
        df['poly_fit_max'] = polynome(np.arange(len(df)))

    x = df[df['Temp_extrema']==-2].index
    y = df[df['Temp_extrema']==-2]['Temp_smooth'].values
    if (len(x) > 0) & (len(y) > 0):
        convex_envelope_min = np.interp(np.arange(df.shape[0]), x, y)
        df['convex_envelope_min'] = convex_envelope_min
        polynome = get_poly_fit(np.arange(len(df)),df['convex_envelope_min'].values,config_dict['poly_fit_deg'])
        df['poly_fit_min'] = polynome(np.arange(len(df)))

    df.set_index('Timestamp', inplace=True)

    return df, original_data_sampling

def get_files_from_dir(data_dir, file_extension):
    return file_extension, sorted(glob.glob(os.path.join(data_dir, '*' + file_extension)))

def get_plotly_fig(df, config_dict):

    # Create the base line plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Temperature'], mode='lines', name='Temp Unaltered'))
    fig.add_trace(go.Scatter(x=df.index, y=df['poly_fit'], mode='lines',
                             name='Temp Ambient Proxy (poly-{})'.format(config_dict['poly_fit_deg'])))
    
    # The blue band around the poly_fit
    if ('poly_fit_min' in df) & ('poly_fit_min' in df):
        start_index = df[df['Temp_extrema']!=0].index[0]
        end_index = df[df['Temp_extrema']!=0].index[-1]
        fig.add_trace(go.Scatter(x=df.loc[start_index:end_index].index,
                                 y=df.loc[start_index:end_index,'poly_fit_min'],
                                 fill=None, mode='lines', line=dict(color='blue'),
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=df.loc[start_index:end_index].index,
                                 y=df.loc[start_index:end_index,'poly_fit_max'],
                                 fill='tonexty', mode='lines', line=dict(color='blue'),
                                 showlegend=False))
        # Update layout if needed
        fig.update_layout(
            title='Filled Area Between Bounds',
            xaxis=dict(title='X-axis Label'),
            yaxis=dict(title='Y-axis Label')
        )
    # Adding Temp_smooth on which the min and max are calculated
    fig.add_trace(go.Scatter(x=df.index, y=df['Temp_smooth'], mode='lines',
                             name='Temp Smoothed ({})'.format(config_dict['smooth_factor'])))
    # Adding the min and max
    loc_max = df[df["Temp_extrema"] == 2].copy()
    fig.add_trace(go.Scatter(x=loc_max.index, y=loc_max['Temp_smooth'], mode='markers', name='Local Maxima',
                             marker=dict(color='red', symbol='circle')))
    loc_min = df[df["Temp_extrema"] == -2].copy()
    fig.add_trace(go.Scatter(x=loc_min.index, y=loc_min['Temp_smooth'], mode='markers', name='Local Minima',
                             marker=dict(color='green', symbol='circle')))

    min_temp, bin_width = get_bin_width(df, config_dict)
    res_extrema_dict = get_prediction_result_extrema(df, config_dict)
    for key, value in res_extrema_dict.items():
        fig.add_shape(type="rect", x0=value.index[0], y0=min_temp-0.5, x1=value.index[-1], y1=min_temp,
                  line=dict(color='blue', width=1), fillcolor='blue', opacity=0.8)

    min_temp -=0.6
    res_seasonal_dict = get_prediction_result_seasonal(df, config_dict)
    for key, value in res_seasonal_dict.items():
        if config_dict['ambient_h2o_dropdown'] == 'amb_gt_h2o':
            fig.add_shape(type="rect", x0=value.index[0], y0=min_temp-0.5, x1=value.index[-1], y1=min_temp,
                          line=dict(color='orange', width=1), fillcolor='orange', opacity=0.8)

        else:
            fig.add_shape(type="rect", x0=value.index[0], y0=min_temp - 0.5, x1=value.index[-1], y1=min_temp,
                          line=dict(color='green', width=1), fillcolor='green', opacity=0.8)

    min_temp -=0.6
    if 'GroundTruth' in df.columns:
        y = df['GroundTruth'].dropna()
        if len(y)> 0:
            for start,stop in get_start_stops(y):
                bin_start = start - bin_width / 2
                bin_end = stop + bin_width / 2
                fig.add_shape(type="rect", x0=bin_start, y0=min_temp-0.5, x1=bin_end, y1=min_temp,
                              line=dict(color='red', width=1), fillcolor='red', opacity=0.8)



    # Add invisible traces for legend
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                             marker=dict(color='red', opacity=0.8,  symbol='square', size=10),
                             name='Ground Truth'))

    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                             marker=dict(color='blue', opacity=0.8, symbol='square', size=10),
                             name='Pred Local-Extr'))

    if config_dict['ambient_h2o_dropdown'] == 'amb_gt_h2o':
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(color='orange', opacity=0.8, symbol='square', size=10),
                                 name='Pred Season-Res - (H2O < Amb)'))
    else:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(color='green', opacity=0.8, symbol='square', size=10),
                                 name='Pred Season-Res + (H2O > Amb)'))

    # Set title for the figure
    fig.update_layout(title=config_dict['file_dropdown'])

    # Update layout to include grid lines
    fig.update_layout(
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGrey'),  # Customize x-axis grid
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGrey'),  # Customize y-axis grid
    )
    return fig

def get_poly_fit(x,y,degree):

    # Fitting a polynomial curve to the data
    coefficients = np.polyfit(x, y, degree)
    polynomial = np.poly1d(coefficients)
    # Generating the fitted curve
    return polynomial

def get_start_stops(y):
    # Assuming 'y.index' contains timestamps in ascending order
    start_stop_pairs = []
    current_group = [y.index[0]]  # Initialize the first timestamp as the start of the group

    for i in range(1, len(y.index)):
        time_diff = y.index[i] - y.index[i - 1]

        # Check if the difference between consecutive timestamps is 1 minute
        if time_diff == timedelta(minutes=1):
            current_group.append(y.index[i])  # Add to the current group
        else:
            # If there's a jump, append the current group as a start-stop pair
            start_stop_pairs.append((current_group[0], current_group[-1]))
            current_group = [y.index[i]]  # Start a new group

    # Append the last group if it's not already added
    if current_group:
        start_stop_pairs.append((current_group[0], current_group[-1]))


    return start_stop_pairs

def getting_extrema(temperatures, config_dict):

    # selection of the start and stop times for which the extrema need to be considered
    start_time = pd.to_datetime(config_dict['detect_start_time']).time()
    stop_time = pd.to_datetime(config_dict['detect_stop_time']).time()

    # Create a mask for values outside the time interval
    mask = (temperatures.index.time < start_time) | (temperatures.index.time > stop_time)

    # Set 'Temp_extrema' values to zero where the index falls outside the interval
    temperatures.loc[mask] = temperatures.min()
    df = temperatures.to_frame().copy()

    maxima_indices, _ = find_peaks(temperatures,
                                   distance=config_dict['dist_for_maxima'], prominence=config_dict['peak_prominence'])
    minima_indices, _ = find_peaks(-temperatures,
                                   distance=config_dict['dist_for_minima'], prominence=config_dict['peak_prominence'])

    if (len(maxima_indices) > 0) & (len(minima_indices) > 0):
        alternating_max_min, max_is_first = get_alternating_values(maxima_indices, minima_indices)
        alternating_temperatures = temperatures[alternating_max_min]
    else:
        alternating_temperatures = []

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

    df.loc[mask,'Temp_extrema'] = 0

    # Find the index of the first occurrence of value 2 in 'Temp_extrema' column
    index_first_value_positive_2 = df.index[df['Temp_extrema'] == 2].min()
    index_last_value_negative_2 = df.index[df['Temp_extrema'] == -2].max()


    if index_first_value_positive_2 is not pd.NaT:
        # Set values before the first occurrence of 2 to 0
        df.loc[df.index < index_first_value_positive_2, 'Temp_extrema'] = 0

    if index_last_value_negative_2 is not pd.NaT:
        # Set values before the first occurrence of 2 to 0
        df.loc[df.index > index_last_value_negative_2, 'Temp_extrema'] = 0

    return df['Temp_extrema']

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

    if gt_col_name == '':
        col_name_optional_dict = {}
    else:
        col_name_optional_dict = {'gt_col_name': gt_col_name}

    for key,value in col_name_optional_dict.items():
        if value:   # only when non-empty string, the col_name_dict gets updated
            col_name_dict.update({key: value})

    return df[col_name_dict.values()], col_name_dict

def make_trend_plotly(df):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05)

    # Plot original series and poly fit
    fig.add_trace(go.Scatter(x=df.index, y=df['Temperature'], mode='lines', name='Original'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['poly_fit'], mode='lines', name='Poly Fit', line=dict(dash='dash', color='red')), row=1, col=1)

    # Plot trend component
    if 'trend' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['trend'], mode='lines', name='Trend'), row=2, col=1)

        # Plot seasonal component
        fig.add_trace(go.Scatter(x=df.index, y=df['seasonal'], mode='lines', name='Seasonal'), row=3, col=1)

        # Plot residue component
        fig.add_trace(go.Scatter(x=df.index, y=df['resid'], mode='lines', name='Residue'), row=4, col=1)

    # Plot Ground Truth and Pred
    if 'GroundTruth' in df:
        fig.add_trace(go.Scatter(x=df.index, y=df['GroundTruth'], mode='lines', name='Ground Truth'), row=4, col=1)
    if 'trend' in df.columns:
        threshold = 0.5  # Replace this with your desired threshold value
        df['pred'] = np.where(df['resid'] > threshold, threshold, np.nan)
        fig.add_trace(go.Scatter(x=df.index, y=df['pred'], mode='lines', name='Pred'), row=4, col=1)

    fig.update_layout(height=800, width=800, title_text="Trend Components")  # Set the overall layout

    return fig

def process(config_dict):

    # file_dropdown =   'Bangladesh_2023-12-12.xlsx',
    #                   'Consolidated UG Data Jan 2023 Batch01.xlsx',
    #                   'Consolidated UG Data Jan 2023 Batch02.xlsx',
    #                   'Consolidated UG Data Jan 2023 Batch03a.xlsx',
    #                   'Consolidated UG Data Jan 2023 Batch03b.xlsx',
    #                   'Consolidated UG Data Jan 2023 Batch04.xlsx',
    #                   'Kaliro Use Data (Kakosi Budumba) Batch05.xlsx'



    data_directory = config_dict['dir_to_process']
    selected_file = config_dict['file_dropdown']

    df, _ = get_data(os.path.join(data_directory, selected_file), config_dict)

    res_dict_extrema = get_prediction_result_extrema(df, config_dict)
    gui_output_extrema = extract_gui_output(res_dict_extrema)

    res_dict_seasonal = get_prediction_result_seasonal(df, config_dict)
    gui_output_seasonal = extract_gui_output(res_dict_seasonal)


    return df, res_dict_extrema, res_dict_seasonal

def resample_df(df, resample='1T'):
    # Ensure the index is in DateTime format
    df.index = pd.to_datetime(df.index)

    # Resample to 1-minute intervals
    resampled_df = df.resample(resample).ffill()
    return resampled_df



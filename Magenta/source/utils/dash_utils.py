import pandas as pd
import numpy as np
from itertools import groupby
from scipy.signal import find_peaks
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose

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
        #print('removed idx: {}, value: {}'.format(i, removed))

    if merged_indices_res[0] == maxima_indices[0]:
        max_is_first = 1
    else:
        max_is_first = 0

    # check if the last is a maximum:  if so remove it from the list
    if merged_indices_res[-1] == maxima_indices[-1]:
        merged_indices_res.pop()

    return merged_indices_res, max_is_first

def getting_extrema(temperatures, config_dict):

    # selection of the start and stop times for which the extrema need to be considered
    start_time = pd.to_datetime(config_dict['detect_start_time']).time()
    stop_time = pd.to_datetime(config_dict['detect_stop_time']).time()

    # Create a mask for values outside the time interval
    mask = (temperatures.index.time < start_time) | (temperatures.index.time > stop_time)

    # Set 'Temp_extrema' values to zero where the index falls outside the interval
    temperatures.loc[mask] = temperatures.min()
    df = temperatures.to_frame().copy()

    maxima_indices, _ = find_peaks(temperatures,  distance=config_dict['dist_for_maxima'],
                                   prominence=config_dict['peak_prominence'])
    minima_indices, _ = find_peaks(-temperatures,  prominence=config_dict['peak_prominence'],
                                   distance=config_dict['dist_for_minima'])

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

def get_data(filename, config_dict):

    df, col_name_dict = import_xlsx_to_df(filename,
                                          timestamp_col_name=config_dict['timestamp_col_name'],
                                          temp_col_name=config_dict['temp_col_name'],
                                          gt_col_name=config_dict['gt_col_name'])

    # Rename columns
    col_mapping = {col_name_dict['temp_col_name']: 'Temperature'}
    if 'gt_col_name' in col_name_dict:
        col_mapping.update({col_name_dict['gt_col_name']: 'GroundTruth'})

    df= df.rename(columns=col_mapping)

    print('Polynomial fit (degree = {})'.format(type(config_dict['poly_fit_deg'])))
    # proxy for the ambient temperature
    polynome = get_poly_fit(np.arange(len(df)),df['Temperature'].values,config_dict['poly_fit_deg'])
    df['poly_fit'] = polynome(np.arange(len(df)))


    df = resample_df(df, resample=config_dict['resample_string'])
    df['Temperature'].fillna(method='bfill', inplace=True)
    df['Temperature'].fillna(method='ffill', inplace=True)

    df['Temp_smooth'] = df['Temperature'].rolling(window=config_dict['smooth_factor']).mean()
    # Shift the rolling mean by half the window size backward
    half_window_shift = config_dict['smooth_factor'] // 2
    df['Temp_smooth'] = df['Temp_smooth'].shift(-half_window_shift)
    df['Temp_smooth'].fillna(method='bfill', inplace=True)

    df_extrema = getting_extrema(df['Temp_smooth'], config_dict)
    df = pd.concat([df, df_extrema], axis=1)

    temperature_series = df['Temperature']
    results = seasonal_decompose(temperature_series, model='additive', period=24)  # Assuming a daily seasonality
    df = pd.concat([df, results.trend, results.seasonal, results.resid], axis=1)


    df.reset_index(inplace=True)
    x = df[df['Temp_extrema']==2].index
    y = df[df['Temp_extrema']==2]['Temp_smooth'].values
    convex_envelope_max = np.interp(np.arange(df.shape[0]), x, y)
    df['convex_envelope_max'] = convex_envelope_max
    polynome = get_poly_fit(np.arange(len(df)),df['convex_envelope_max'].values,config_dict['poly_fit_deg'])
    df['poly_fit_max'] = polynome(np.arange(len(df)))

    x = df[df['Temp_extrema']==-2].index
    y = df[df['Temp_extrema']==-2]['Temp_smooth'].values
    convex_envelope_min = np.interp(np.arange(df.shape[0]), x, y)
    df['convex_envelope_min'] = convex_envelope_min
    polynome = get_poly_fit(np.arange(len(df)),df['convex_envelope_min'].values,config_dict['poly_fit_deg'])
    df['poly_fit_min'] = polynome(np.arange(len(df)))

    df.set_index('Timestamp', inplace=True)

    return df

def get_plotly_fig(df, config_dict):

    # Create the base line plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Temperature'], mode='lines', name='Temp Unaltered'))
    fig.add_trace(go.Scatter(x=df.index, y=df['poly_fit'], mode='lines',
                             name='Temp Ambient Proxy (poly-{})'.format(config_dict['poly_fit_deg'])))
    start_index = df[df['Temp_extrema']!=0].index[0]
    end_index = df[df['Temp_extrema']!=0].index[-1]
    fig.add_trace(go.Scatter(x=df.loc[start_index:end_index].index, y=df.loc[start_index:end_index,'poly_fit_min'], fill=None, mode='lines', line=dict(color='blue'), name='Lower Bound'))
    fig.add_trace(go.Scatter(x=df.loc[start_index:end_index].index, y=df.loc[start_index:end_index,'poly_fit_max'], fill='tonexty', mode='lines', line=dict(color='blue'), name='Upper Bound'))

    # Update layout if needed
    fig.update_layout(
        title='Filled Area Between Bounds',
        xaxis=dict(title='X-axis Label'),
        yaxis=dict(title='Y-axis Label')
    )
    fig.add_trace(go.Scatter(x=df.index, y=df['Temp_smooth'], mode='lines',
                             name='Temp Smoothed ({})'.format(config_dict['smooth_factor'])))
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
    if 'resid' in df.columns:
        y = df['resid'].dropna()
        y = y[y < config_dict['res_thres_minus']]
        if len(y)> 0:
            for start,stop in get_start_stops(y):
                bin_start = start - bin_width / 2
                bin_end = stop + bin_width / 2
                fig.add_shape(type="rect", x0=bin_start, y0=min_temp-0.5, x1=bin_end, y1=min_temp,
                              line=dict(color='orange', width=1), fillcolor='orange', opacity=0.8)

    min_temp -=0.6
    if 'GroundTruth' in df.columns:
        y = df['GroundTruth'].dropna()
        if len(y)> 0:
            for start,stop in get_start_stops(y):
                bin_start = start - bin_width / 2
                bin_end = stop + bin_width / 2
                fig.add_shape(type="rect", x0=bin_start, y0=min_temp-0.5, x1=bin_end, y1=min_temp,
                              line=dict(color='red', width=1), fillcolor='red', opacity=0.8)

    min_temp -=0.6
    if 'resid' in df.columns:
        y = df['resid'].dropna()
        y = y[y > config_dict['res_thres_plus']]
        if len(y)> 0:
            for start,stop in get_start_stops(y):
                bin_start = start - bin_width / 2
                bin_end = stop + bin_width / 2
                fig.add_shape(type="rect", x0=bin_start, y0=min_temp-0.5, x1=bin_end, y1=min_temp,
                              line=dict(color='green', width=1), fillcolor='green', opacity=0.8)

    # Add invisible traces for legend
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                             marker=dict(color='red', opacity=0.8,  symbol='square', size=10), name='Ground Truth'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                             marker=dict(color='blue', opacity=0.8, symbol='square', size=10), name='Pred Local-Extr (H2O < Amb)'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                             marker=dict(color='orange', opacity=0.8, symbol='square', size=10), name='Pred Season-Res - (H2O < Amb)'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                             marker=dict(color='green', opacity=0.8, symbol='square', size=10), name='Pred Season-Res + (H2O > Amb)'))


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

    if gt_col_name == '':
        col_name_optional_dict = {}
    else:
        col_name_optional_dict = {'gt_col_name': gt_col_name}

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



import os
import warnings

import numpy as np
import pandas as pd

from utils.dash_utils import process, extract_gui_output, make_trend_plotly, get_plotly_fig, create_necessary_paths
from utils.dash_utils import get_files_from_dir

# Suppress the RankWarning
warnings.filterwarnings('ignore', category=np.RankWarning)


# Specify all the parameters here, as well as the director that needs to be processed
config_dict = \
    {
        'dir_to_process': 'data/',
        'file_extension': '.xlsx',
        'smooth_factor': 5,
        'resample_string': '1T',
        'poly_fit_deg': 20,
        'dist_for_maxima': 3,
        'dist_for_minima': 3,
        'peak_prominence': 0.1,
        'ambient_h2o_dropdown': 'amb_gt_h2o',
        'res_thres_minus': -0.5,
        'res_thres_plus': 0.5,
        'timestamp_col_name':'DateTime_EAT',
        'temp_col_name':'Celsius',
        'gt_col_name':'Use_event',
        'detect_start_time': '00:00:00',
        'detect_stop_time': '23:59:00'
    }

def get_duration(start, end):
    duration = None
    if (start is not None) and (end is not None):
        duration = (end-start).total_seconds() // 60
    return duration

def get_meta_info(config_dict, file_names):
    directory_name = config_dict['dir_to_process']
    file_extension = config_dict['file_extension']

    data = {
        'directory_name': [directory_name] + [''],
        'file_extension': [file_extension] + [''],
        '': ['']*2
    }

    # adding the config_dict to the meta-info
    data.update(
        {
            'Smooth Factor': [''] + [config_dict['smooth_factor']],
            'Resample String': [''] + [config_dict['resample_string']],
            'Polynomil fit (degree)': [''] + [config_dict['poly_fit_deg']],
            'Dist for maxima': [''] + [config_dict['dist_for_maxima']],
            'Dist for minima': [''] + [config_dict['dist_for_minima']],
            'Prominence': [''] + [config_dict['peak_prominence']],
            'Select Option': [''] + [config_dict['ambient_h2o_dropdown']],
            'Threshold Seasonal Residu (+)': [''] + [config_dict['res_thres_plus']],
            'Threshold Seasonal Residu (-)': [''] + [config_dict['res_thres_minus']],
            'Timestamp col name': [''] + [config_dict['timestamp_col_name']],
            'Temp col name': [''] + [config_dict['temp_col_name']],
            'Ground truth col name': [''] + [config_dict['gt_col_name']],
            'Detect between start time': [''] + [config_dict['detect_start_time']],
            'Detect between stop time': [''] + [config_dict['detect_stop_time']]
        }
    )

    data.update(
        {
        **{f"{i + 1:03d}": [file_name.replace(config_dict['dir_to_process'],'')] + [''] for i, file_name in enumerate(file_names)}
        }
    )

    df = pd.DataFrame(data).T
    return df

def transform_to_df(i, naked_file_name, res_dict_extrema, res_dict_seasonal):
    gui_output_extrema = extract_gui_output(res_dict_extrema)
    gui_output_seasonal = extract_gui_output(res_dict_seasonal)

    data = {
        'file_name': [naked_file_name] + ['']*6,
        '': ['']*7
    }

    # combining the statistical-results of both method and adding them to the data-dict
    combined_gui_dict = {}
    for key in gui_output_extrema.keys():
        combined_gui_dict[key] = [gui_output_extrema[key], gui_output_seasonal.get(key, None)]
    for key, value in gui_output_seasonal.items():
        if key not in gui_output_extrema:
            combined_gui_dict[key] = [None, value]

    for key, value in combined_gui_dict.items():
        data.update({key: [''] + [value[0]] + [''] * 2 + [value[1]] + [''] * 2 })

    # combining the (start-stop)-results of both method and adding them to the data-dict
    combined_start_stop_dict = {}
    data.update({'---------': ['', '', 'Method:', 'Extrema', '', 'Method:', 'Seasonal']})
    data.update({'Pump On ID': [''] + ['Dt (min)', 'start', 'end'] * 2})
    for key in res_dict_extrema.keys():

        seasonal_start, seasonal_end = (
            res_dict_seasonal[key].index[0], res_dict_seasonal[key].index[-1]
        ) if key in res_dict_seasonal else (None, None)
        extrema_start, extrema_end = (res_dict_extrema[key].index[0], res_dict_extrema[key].index[-1])
        seasonal_duration = get_duration(seasonal_start, seasonal_end)
        extrema_duration = get_duration(extrema_start, extrema_end)
        combined_start_stop_dict.update({key: [extrema_duration, extrema_start, extrema_end,
                                               seasonal_duration, seasonal_start, seasonal_end]})
    for key, value in res_dict_seasonal.items():
        if key not in res_dict_extrema:
            combined_start_stop_dict.update({key: [None, None, None,
                                                   get_duration(value.index[0], value.index[-1]),
                                                   value.index[0], value.index[-1]]})

    for key, value in combined_start_stop_dict.items():
        data.update({key: [''] + value })
    # transform data-dict to dataframe so that it can be added to the final table
    df = pd.DataFrame(data).T
    return df

def adjust_column_widths_automatically(writer, tab_df_dict):
    # Access the xlsxwriter workbook and worksheet objects

    for tab_name, df in tab_df_dict.items():
        worksheet = writer.sheets[tab_name]

        # set column width of the index to the content max length
        col_length = df.index.astype(str).map(len).max()
        worksheet.set_column(0, 0, col_length + 2)

        # Set column widths based on column content length
        for i, col in enumerate(df.columns):
            col_length = max(df[col].astype(str).map(len).max(), len(str(col)))
            worksheet.set_column(i+1, i+1, col_length + 2)  # Adding extra space for aesthetics

    return writer

if __name__ == '__main__':

    all_results_dict = {}
    create_necessary_paths()
    meta_info_name = 'meta-info'
    #
    f_ext, file_names = get_files_from_dir(config_dict['dir_to_process'], config_dict['file_extension'])
    print("Processing all files ending with '{}' located in directory: '{}'".format(f_ext, config_dict['dir_to_process']))

    # open output table and write the meta_info in the corresponding tab
    result_excel_file = os.path.join('tables','processed.xlsx')
    writer = pd.ExcelWriter(result_excel_file, engine="xlsxwriter")
    meta_info_df = get_meta_info(config_dict, file_names)
    meta_info_df.to_excel(writer, sheet_name=meta_info_name, index=True)
    tab_df_dict = {meta_info_name: meta_info_df}

    total_files_to_process = len(file_names)

    for i, file in enumerate(file_names):
        naked_file_name = file.replace(config_dict['dir_to_process'],'')
        print("{}/{}: {}".format(i+1, total_files_to_process, naked_file_name))

        config_dict.update({'file_dropdown': naked_file_name})
        df, res_dict_extrema, res_dict_seasonal = process(config_dict)
        df_for_excel = transform_to_df(i, naked_file_name, res_dict_extrema, res_dict_seasonal)
        tab_df_dict.update({'{:03d}'.format(i+1): df_for_excel})
        df_for_excel.to_excel(writer, sheet_name='{:03d}'.format(i+1), index=True)


        # Create the seasonality plot in the plots directory
        file_name = naked_file_name.replace(config_dict['file_extension'],'.html')
        fig = make_trend_plotly(df)
        fig.write_html(os.path.join('plotly','season', f"{i+1:03d}_" + file_name))

        # Create the plotly figure in the plotly directory
        file_name = naked_file_name.replace(config_dict['file_extension'],'.html')
        fig = get_plotly_fig(df, config_dict)
        fig.write_html(os.path.join('plotly', 'dashb', f"{i+1:03d}_" +file_name))

    writer = adjust_column_widths_automatically(writer, tab_df_dict)
    writer.close()
    print("\nCheck-out the results in {}".format(result_excel_file))

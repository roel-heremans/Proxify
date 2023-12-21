
import warnings
import numpy as np
import matplotlib.pyplot as plt
from utils.dash_utils import get_plotly_fig, make_trend_plot
from utils.dash_utils import process, create_necessary_paths


# Suppress the RankWarning
warnings.filterwarnings('ignore', category=np.RankWarning)

config_dict = {'dir_to_process': 'data',
                   'file_dropdown': 'Consolidated UG Data Jan 2023 Batch03b.xlsx',
                   'smooth_factor': 5,
                   'resample_string': '1T',
                   'poly_fit_deg': 6,
                   'dist_for_maxima': 3,
                   'dist_for_minima': 3,
                   'peak_prominence': 0.1,
                   'ambient_h2o_dropdown': 'amb_gt_h2o',
                   'res_thres_minus': -0.5,
                   'res_thres_plus': 0.5,
                   'timestamp_col_name': 'DateTime_EAT',
                   'temp_col_name': 'Celsius',
                   'gt_col_name': 'Use_event',
                   'detect_start_time': '00:00:00',
                   'detect_stop_time': '23:59:00'}

if __name__ == "__main__":
    create_necessary_paths()

    df, res_dict_extrema, res_dict_seasonal = process(config_dict)
    fig1 = make_trend_plot(df)
    fig2 = get_plotly_fig(df, config_dict)




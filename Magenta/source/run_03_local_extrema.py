
import warnings
import numpy as np
import matplotlib.pyplot as plt
from utils.dash_utils import get_plotly_fig
from utils.dash_utils import process


# Suppress the RankWarning
warnings.filterwarnings('ignore', category=np.RankWarning)

def make_trend_plot(df):

    # Create subplots with shared x-axis
    fig, axs = plt.subplots(4, 1, figsize=(10, 6), sharex=True)

    # Plot original series
    axs[0].plot(df['Temperature'], label='Original')
    axs[0].plot(df['poly_fit'], label='Poly Fit', linestyle='--', color='red')
    axs[0].legend(loc='upper left')

    # Plot trend component
    if 'trend' in df.columns:                    # then also seasonal and residue can be plotted
        axs[1].plot(df['trend'], label='Trend')
        axs[1].legend(loc='upper left')

        # Plot seasonal component
        axs[2].plot(df['seasonal'], label='Seasonal')
        axs[2].legend(loc='upper left')

        # Plot residue component
        axs[3].plot(df['resid'], label='Residue')
        axs[3].legend(loc='upper left')

    # Plot Ground Truth
    axs[3].plot(df['GroundTruth'], label='GT')
    axs[3].legend(loc='upper left')

    if 'trend' in df.columns:
        threshold = 0.5  # Replace this with your desired threshold value
        # Creating the 'pred' column based on the condition
        df['pred'] = np.where(df['resid'] > threshold, threshold, np.nan)
        axs[3].plot(df['pred'], label='Pred')
        axs[3].legend(loc='upper left')

    plt.tight_layout()  # Adjusts spacing between subplots for better visualization
    fig.show()
    return fig


if __name__ == "__main__":
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

    df, res_dict_extrema, res_dict_seasonal = process(config_dict)
    fig1 = make_trend_plot(df)
    fig2 = get_plotly_fig(df, config_dict)




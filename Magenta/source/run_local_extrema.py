import os
import numpy as np
from scipy.optimize import curve_fit
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from utils.dash_utils import get_alternating_values, getting_extrema, get_data, get_plotly_fig, get_start_stops, \
    import_xlsx_to_df, resample_df


def fourier_series(x, *a):
    n = len(a) //2 # Extracting the number of harmonics from the length of a
    w = 2 * np.pi / 24  # Frequency of the cycle, assuming a 24-hour cycle

    ret = a[0]  # Initial value (mean temperature)

    for i in range(n):
        ret += a[2*i + 1] * np.cos((i + 1) * w * x) + a[2*i + 2] * np.sin((i + 1) * w * x)

    return ret

def make_trend_plot(df):

    df = my_curve_fit(df)
    # Create subplots with shared x-axis
    fig, axs = plt.subplots(4, 1, figsize=(10, 6), sharex=True)

    # Plot original series
    axs[0].plot(df['Temperature'], label='Original')
    axs[0].plot(df['poly_fit'], label='Poly Fit', linestyle='--', color='red')
    axs[0].legend(loc='upper left')

    # Plot trend component
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

    threshold = 0.5  # Replace this with your desired threshold value

    # Creating the 'pred' column based on the condition
    df['pred'] = np.where(df['resid'] > threshold, threshold, np.nan)
    axs[3].plot(df['pred'], label='Pred')
    axs[3].legend(loc='upper left')

    plt.tight_layout()  # Adjusts spacing between subplots for better visualization
    plt.show()
    return fig

def my_curve_fit(df):
    x = np.arange(len(df))  # Assuming the index represents the x-axis (time)
    y = df['Temperature'].values  # Assuming 'Temperature' column contains the y-axis values

    # Degree of the polynomial for fitting the curve (change this as needed)
    degree = 25

    # Fitting a polynomial curve to the data
    coefficients = np.polyfit(x, y, degree)
    polynomial = np.poly1d(coefficients)

    # Generating the fitted curve
    df['poly_fit'] = polynomial(x)

    #------------------ Fourier solution --------------------------------------
    # Initial guess for Fourier coefficients
    T = 24 # hours
    min_temp = 15
    initial_guess = [30, 0, 0, 0, 0]  # Assuming 2 harmonics, set to appropriate number

    # Bounds for parameters
    lower_bounds = [15] + [-np.inf] * 4  # Adjust according to your data range and number of harmonics
    upper_bounds = [32] + [np.inf] * 4  # Adjust according to your data range and number of harmonics

    # Fit the Fourier series to the data
    params, _ = curve_fit(fourier_series, x, y, p0=initial_guess, bounds=(lower_bounds, upper_bounds))

    # Generate fitted curve using the obtained Fourier coefficients
    df['fourier_fit'] = fourier_series(x, *params)

    return df


if __name__ == "__main__":
    config_dict = {'file_dropdown': 'Bangladesh_2023-12-12.xlsx',
                   'smooth_factor': 15,
                   'resample_string': '1T',
                   'poly_fit_deg': 20,
                   'ma_size': 60,
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
                   'detect_stop_time': '23:59:00'}

    data_directory = 'data'
    selected_file = 'Bangladesh_2023-12-12.xlsx'
    #selected_file = 'Consolidated UG Data Jan 2023 Batch03b.xlsx'
    #selected_file = 'Kaliro Use Data (Kakosi Budumba) Batch05.xlsx'
    df = get_data(os.path.join(data_directory, selected_file), config_dict)
    fig = make_trend_plot(df)
    fig, result_dict_extrema = get_plotly_fig(df, config_dict)
    a = 1



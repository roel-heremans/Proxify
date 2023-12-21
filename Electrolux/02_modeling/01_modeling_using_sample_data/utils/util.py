import pandas as pd
import os
import math
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import plotly.io as pio


def adding_scr_info_to_df(df):
    '''
    :param df:
              date  qty_sold  sca_m0  sca_m1  ...  scf_m9  scf_m10  scf_m11  scf_m12
        0   201601    111920     821    1074  ...     868      951     1021      513
        1   201602    133784     970    1034  ...     954     1038     1066      550
        2   201603    132199     873     963  ...     812      853      928      432
        ...
        with columns: ['date', 'qty_sold', 'sca_m0', 'sca_m1', 'sca_m2', 'sca_m3', 'sca_m4',
       'sca_m5', 'sca_m6', 'sca_m7', 'sca_m8', 'sca_m9', 'sca_m10', 'sca_m11',
       'sca_m12', 'scf_m0', 'scf_m1', 'scf_m2', 'scf_m3', 'scf_m4', 'scf_m5',
       'scf_m6', 'scf_m7', 'scf_m8', 'scf_m9', 'scf_m10', 'scf_m11',
       'scf_m12']


    :return: adding following columns to the df: 'scr', 'scrf', 'scr_rol_win03', 'scrf_rol_win03' 'scr_rol_win12', 'scrf_rol_win12'
    scr = actual service call ratio (uses the actual number of service calls and actual sales)
    scrf = forecasted service call ratio (uses the forecasted service calls and the actual sales (because the sales forecast was not available))
    '''

    # df2 = get_scr_df(df)
    ###################
    scr_df = []

    # first 12 months are needed to calculate the SCR based on the history of the sales and service call forcasts
    if len(df) > 12:
        for i in range(12, len(df)):
            scr = extract_scr(df, i, 2)
            scrf_as = extract_scr(df, i, 15)
            scrf_fs = extract_scr(df, i, 29, qty_sold_colid=28)
            scr_df.append([df.iloc[i, 0], scr, scrf_as, scrf_fs])

    df_out = pd.DataFrame(scr_df, columns=['date', 'scr', 'scrf_as', 'scrf_fs'])

    #############
    merged_df = df.merge(df_out, left_on='date', right_on='date', how='left')
    merged_df.set_index('date', inplace=True)

    return merged_df


def adding_scrxQ_info_to_df(df):
    '''
    Adding Last Quarter and Last Year scr, scrf_as, scr_sf information to the df

    :param df: dataframe with following columns:
    ['qty_sold', 'sca_m0', 'sca_m1', 'sca_m2', 'sca_m3', 'sca_m4', 'sca_m5',
       'sca_m6', 'sca_m7', 'sca_m8', 'sca_m9', 'sca_m10', 'sca_m11', 'sca_m12',
       'scf_m0', 'scf_m1', 'scf_m2', 'scf_m3', 'scf_m4', 'scf_m5', 'scf_m6',
       'scf_m7', 'scf_m8', 'scf_m9', 'scf_m10', 'scf_m11', 'scf_m12', 'scr',
       'scrf', 'vol_diff']

    :return: adding Last Quarter and Last Year scr and scrf information to the df
        scr_rol_win03 and scrf_rol_win03 = same for the average (over 3 months --> Last Quarter) rolling window
        scr_rol_win12 and scrf_rol_win12 = same for the average (over 12 months --> Last Year) rolloing window
    '''

    for winsize in [3, 12]:
        for col_name in ['scr', 'scrf_as', 'scrf_fs']:
            rolling_average = df[col_name].rolling(window=winsize).mean()
            df['{}_rol_win{:02d}'.format(col_name, winsize)] = rolling_average

    return df


def check_on_zeros(df, replacing_value=1):
    '''if there is one 0 value in the df for one of the $m_i$'s, the model will crash, that is why we
    replace the 0's by 1's. The nr_interventions, lets you know how many times that was the case.
    When there are too many zeros that have been changed, the analysis can be stopped based on the number of zeros that have been changed.

    :param df: dataframe containing the sc for m_i (i=0,1,..12)
    :param replacing_value: default value is 1 but can be set to something else
    :return: same df with changed values'''

    # Check for 0 values in the DataFrame
    zero_locations = (df == 0)

    # Get row and column indices where 0 values occur
    rows, cols = zero_locations.to_numpy().nonzero()

    # Print the row and column indices
    for row, col in zip(rows, cols):
        df.iat[row, col] = replacing_value
    return df, len(rows)


def create_necessary_paths():
    to_create_paths = ['pickles', 'plots/CtrlPlots', 'plots/mape_summary', 'plots/SalesForecast', 'plots/SCR',
                       'plots/SCR/Regression/line', 'plots/SCR/Regression/group',
                       'plots/SCR/Regression/subgroup', 'plotly', 'tables']
    for my_path in to_create_paths:
        if not os.path.exists(os.path.join(os.getcwd(), my_path)):
            # Create the directory
            os.makedirs(os.path.join(os.getcwd(), my_path))


def create_regression_control_plot(df, title='', hierarchy=''):
    '''Service Call Control plot one per predicted batch id. Actually not used, the call to this function is commented out in the main. But can be usefull to debug the code when necessary.
    :param df:
    :param title
    :param hierarchy
    :return plots are created in plots/CtrlPlots/Regr_CtrlPlot
    '''

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.errorbar(df.index, df['Actual'], marker='o', color='k', linestyle="None", label='Actual Sales')
    ax1.errorbar(df.index, df['Forecast'], marker='o', color='r', label='Forecast Sales')
    # ax1.set_xticklabels([])
    ax1.set_ylabel('Service Calls')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid()

    vol_diff = (df['Actual'] - df['Forecast']) / df['Actual'] * 100
    mape = np.mean(np.abs(vol_diff))
    pos_vol_diff = [val if val > 0 else 0 for val in vol_diff]
    neg_vol_diff = [val if val < 0 else 0 for val in vol_diff]

    ax2.bar(df.index, pos_vol_diff, color='green')
    ax2.bar(df.index, neg_vol_diff, color='red')

    # Add a text annotation with result information
    str1 = "MAPE: {:.2f}%".format(mape)
    ax2.text(0.5, 1.05, str1, transform=ax2.transAxes, ha='center', va='center', fontsize=18)

    # ax2.set_ylim([-15,15])
    # ax2.set_xticks(df['date'])
    # ax2.set_xticklabels(df['date'], rotation=90)
    ax2.set_ylabel('Percentage Error')
    ax2.grid()
    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'CtrlPlots', 'Regr_CtrlPlot_{}_{}.png'.format(hierarchy, title[-6:])))
    plt.close()


def create_regression_mape_plot(df, title='', hierarchy='', ):
    ''' The call to this function is alsoo commented out in the main function. But can be activated by uncommenting the corresponding line of code in the main function. The plot shows the MAPE as a function of the predicted batches. When next_pred_id =[0,3,6,12] is an array then
    this figure will show you a mape value for each predicted batch_id. One graph per next_pred_id. The legend
     will show different color lines per next_pred_id. This graph shows you that the average MAPE value increases
      when you predict further into the future.'''
    res = df.T
    fig, ax = plt.subplots(figsize=(10, 5))
    lines = res.plot(ax=ax, legend=True)

    # Get the lines for the legend
    lines = lines.lines

    # Customize the labels in the legend
    labels = ['{}: Avg={:.1f}+-{:.1f}'.format(res_col, res[res_col].mean(), res[res_col].std()) for res_col in
              res.columns]

    # Add a custom legend
    ax.legend(lines, labels, loc="upper left")

    # Customize the plot as needed
    ax.set_xlabel("Date")
    ax.set_ylabel("MAPE")
    ax.set_title("{} - {}".format(hierarchy, title))
    ax.grid()
    plt.savefig(os.path.join("plots", "Regrs_{}.png".format(hierarchy)))
    plt.close()


def create_scr_figures(scr_df, **kwargs):
    ''' Creates the Service Call Ratio figure. The call to this function is also commented out in the main, since a better visualization is made that also shows the uncertainty band due to the sales uncertainty. See function: create_scr_figures_qtySold_study'''

    method = kwargs.pop('method', 'Regression - MarkovChain')
    hierarchy = kwargs.pop('hierarchy', 'line')
    qty_sold_scf = kwargs.pop('qty_sold_scf', 1)
    title_addition = '- Qty Sold ScaleF={}'.format(qty_sold_scf)

    scr_df.index = pd.to_datetime(scr_df.index, format='%Y%m')
    scr_df.index = scr_df.index.strftime('%Y%m')

    print('    --> create_scr_figures --')
    fig, ax = plt.subplots(2, 1, figsize=(15, 5), sharex=True)
    ax[0].plot(scr_df.index, scr_df['scr'], marker='o', color='k', label='Actual')
    ax[0].plot(scr_df.index, scr_df['scrf'], marker='o', color='b', label='Predicted')
    ax[0].set_xlabel('Month')
    ax[0].set_ylabel('Serv Call Ratio')
    # if 'dishcare' in hierarchy:
    #    ax[0].set_ylim([0.027,0.08])
    # elif 'fabriccare' in hierarchy:
    #    ax[0].set_ylim([0.02,0.06])
    # elif 'foodpreparation' in hierarchy:
    #    ax[0].set_ylim([0.01,0.031])
    # elif 'foodpreservation' in hierarchy:
    #    ax[0].set_ylim([0.01,0.045])
    ax[0].grid()
    ax[0].legend()
    ax[0].set_xlim([scr_df.index[12], scr_df.index[-1]])

    vol_diff = scr_df['vol_diff']

    mape = np.nanmean(abs(vol_diff))
    mdape = np.nanmedian(abs(vol_diff))
    mape_std = np.nanstd(abs(vol_diff))
    qty_sold_mean = scr_df['qty_sold'].mean()

    # scr_dict.update({'{}_{}'.format(method, hierarchy): [mape, mdape, mape_std]})

    ax[0].set_title('{}: {} -\n <q_sold>={:.0f} - MAPE={:.1f}+-{:.1f} {}'.format(
        hierarchy, method, qty_sold_mean, mape, mape_std, title_addition)
    )
    pos_vol_diff = [val if val > 0 else 0 for val in vol_diff]
    neg_vol_diff = [val if val < 0 else 0 for val in vol_diff]
    ax[1].bar(scr_df.index, pos_vol_diff, color='green')
    ax[1].bar(scr_df.index, neg_vol_diff, color='red')
    ax[1].set_ylabel('(Act-Pre)/Act [%]')
    # ax[1].set_ylim([-35,30])
    ax[1].set_xticklabels(scr_df.index, rotation=90)
    ax[1].grid()
    plt.tight_layout()

    fig_name = os.path.join('plots', 'SCR', 'Regression', get_hierarchy_dirname(hierarchy),
                            'SCR_{}_{}_scf{}.png'.format(
                                method, hierarchy, qty_sold_scf))
    plt.savefig(fig_name)
    plt.close()


def create_scr_figures_qtySold_study(scr_dict, **kwargs):
    ''' This code shows the confidence level around the forecasted SCR versus the Actual SCR value. The
     confidence level comes from the +-x percent scaling of the actual sales volume. In order to be able to use this
     code a dictionary is expected as input that contains for each analysis (for e different scale factor applied on
     the sales volume e.g. qty_sold_scfs = [0.9, 1.0, 1.1]

    :param scr_dict: with keys that look like: dict_keys(['QtySoldScf0.9', 'QtySoldScf1.0', 'QtySoldScf1.1'])
                     the value per key represents a dataframe with the following columns ("sca" stands for service
                     calls actuals, "scf" for sc forecasted, "scr" for service call ratio and "scrf" the forecasted scr):
                     Index(['qty_sold', 'sca_m0', 'sca_m1', 'sca_m2', 'sca_m3', 'sca_m4', 'sca_m5',
                    'sca_m6', 'sca_m7', 'sca_m8', 'sca_m9', 'sca_m10', 'sca_m11', 'sca_m12',
                    'scf_m0', 'scf_m1', 'scf_m2', 'scf_m3', 'scf_m4', 'scf_m5', 'scf_m6',
                    'scf_m7', 'scf_m8', 'scf_m9', 'scf_m10', 'scf_m11', 'scf_m12', 'scr',
                    'scrf', 'vol_diff', 'scr_rol_win03', 'scrf_rol_win03', 'scr_rol_win12',
                    'scrf_rol_win12'], dtype='object')
    :param kwargs: 'method': just used for creating the title of the figure and the filename of the saved figure
                   'hierarchy': used for creating the title of the figure and the filename of the saved figure but also
                                to set the ylim range for the SCR-plot (each prod line has a different scr range)
                   'qty_sold_scf': used for the title as well and to unpack the dataframes for the 3 dict-keys
    :return: saved figure
    '''

    # depending on the following flag a plot is created with just one or two subplots
    # the single subplot will only show the SCR forecast with the uncertainty band due to the Salse Forecast
    # the version with the 2 subplots will also show the volume difference, notice here that the volume difference is
    # taken as the difference between the actual SCR and the respective lower and upper values of the uncertainty band
    remove_volume_plot = kwargs.pop('remove_volume_plot', False)  # False leads to "v2" version in the file name, True leads to "v1" in the file name

    method = kwargs.pop('method', 'Regression - MarkovChain')
    hierarchy = kwargs.pop('hierarchy', 'line')
    qty_sold_scfs = kwargs.pop('qty_sold_scf', [1])

    xQ = kwargs.pop('xQ', 0)
    xq_map = {0: ['scr', 'scrf_as', 'scrf_fs', '0'],
              3: ['scr_rol_win03', 'scrf_as_rol_win03', 'scrf_fs_rol_win03', 'LQ'],
              12: ['scr_rol_win12', 'scrf_as_rol_win12', 'scrf_fs_rol_win12', 'LY']}
    vars_to_use = xq_map[xQ]

    scr_df = scr_dict['QtySoldScf1.0']

    vol_diff_min = get_vol_diff_from_df(scr_df[vars_to_use[0]], scr_df[vars_to_use[1]])
    vol_diff_max = get_vol_diff_from_df(scr_df[vars_to_use[0]], scr_df[vars_to_use[1]])

    pos_vol_diff_as = [val if val > 0 else 0 for val in vol_diff_min]
    neg_vol_diff_as = [val if val < 0 else 0 for val in vol_diff_max]

    vol_diff_min = get_vol_diff_from_df(scr_df[vars_to_use[0]], scr_df[vars_to_use[2]])
    vol_diff_max = get_vol_diff_from_df(scr_df[vars_to_use[0]], scr_df[vars_to_use[2]])

    pos_vol_diff_fs = [val if val > 0 else 0 for val in vol_diff_min]
    neg_vol_diff_fs = [val if val < 0 else 0 for val in vol_diff_max]

    scr_df.index = pd.to_datetime(scr_df.index, format='%Y%m')
    scr_df.index = scr_df.index.strftime('%Y%m')
    print('    --> create_scr_figures_qtySold_study Q{} --'.format(xq_map[xQ][-1]))

    if remove_volume_plot:
        fig, ax1 = plt.subplots(figsize=(15, 5))
        fig_name = os.path.join('plots', 'SCR', 'Regression', get_hierarchy_dirname(hierarchy),
                                'SCR_{}_{}_Q{}_scf_v1.png'.format(
                                    method, hierarchy, xQ))
    else:
        fig, ax = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        fig_name = os.path.join('plots', 'SCR', 'Regression', get_hierarchy_dirname(hierarchy),
                                'SCR_{}_{}_Q{}_scf_v2.png'.format(
                                    method, hierarchy, xQ))
        ax1 = ax[0]
    # Set up common elements for both cases
    ax1.plot(scr_df.index, scr_df[vars_to_use[0]], marker='o', color='k', label='Actual')
    ax1.plot(scr_df.index, scr_df[vars_to_use[2]], marker='o', color='g', label='SCRF_FS')
    ax1.plot(scr_df.index, scr_df[vars_to_use[1]], marker='o', color='b', label='SCRF_AS')
    # Fill the area between scr_min_df['scrf'] and scr_max_df['scrf'] with blue color and alpha value
    # ax1.fill_between(scr_df.index, scr_min_df[vars_to_use[1]], scr_max_df[vars_to_use[1]], color='blue', alpha=0.3)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Serv Call Ratio')
    ax1.grid()
    ax1.legend()
    ax1.set_xlim([scr_df.index[12], scr_df.index[-1]])
    get_vol_diff_from_df(scr_df[vars_to_use[0]], scr_df[vars_to_use[1]])
    mape_as, _, mape_as_std, _ = get_mape_from_series(
        get_vol_diff_from_df(scr_df[vars_to_use[0]], scr_df[vars_to_use[1]]))
    mape_fs, _, mape_fs_std, _ = get_mape_from_series(
        get_vol_diff_from_df(scr_df[vars_to_use[0]], scr_df[vars_to_use[2]]))
    qty_sold_mean = scr_df['qty_sold'].mean()
    ax1.set_title('{}: {} - Q{}\n <q_sold>={:.0f} - MAPE AS={:.1f}+-{:.1f} - MAPE FS={:.1f}+-{:.1f}'.format(
        hierarchy, method, xQ, qty_sold_mean, mape_as, mape_as_std, mape_fs, mape_fs_std)
    )
    if remove_volume_plot:
        ax1.set_xticklabels(scr_df.index, rotation=90)
    else:
        ax[1].bar(scr_df.index, pos_vol_diff_as, color='green')
        ax[1].bar(scr_df.index, neg_vol_diff_as, color='red')
        ax[1].set_ylabel('PE AS')
        ax[1].set_xticklabels(scr_df.index, rotation=90)
        ax[1].grid()
        ax[2].bar(scr_df.index, pos_vol_diff_fs, color='green')
        ax[2].bar(scr_df.index, neg_vol_diff_fs, color='red')
        ax[2].set_ylabel('PE FS')
        ax[2].set_xticklabels(scr_df.index, rotation=90)
        ax[2].grid()

    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()


def create_scr_sunburst_figure_hierarchy(sunburst_data, which, colmap):
    import plotly.express as px

    res = []
    for hierarchy, mape, mdape, mape_std, mdape_std, avg_qty_sold in sunburst_data:
        if not math.isnan(mape):
            h_split = hierarchy.split("_")
            res.append(h_split + [mape])
    df = pd.DataFrame(res, columns=['Line', 'Group', 'Subgroup', 'mape'])
    fig = px.sunburst(df, path=['Line', 'Group', 'Subgroup'], values='mape',
                      color='mape',
                      color_continuous_scale=colmap,
                      )
    fig.show()
    # Save the figure as an HTML file
    fig.write_html(os.path.join('plotly', "sunburst_{}.html".format(which)))


def extract_scr(df: pd.DataFrame, row_id: int, col_id: int, **kwargs) -> float:
    '''Given an input dataframe with at least 13 rows and at least 12 columns, this function extracts the SCR from it.
       Notice for an SCR calculation one needs 12 months (batches) sc data prior the actual date for which the scr is calculated.
       The service call data for the zeroth month (m0) until the 12th month (m12) need to be in consecutive column ids and the
       col_id indicates where the first month sc value can be found in the df.
       df looks like:
                       date qty_sold sc_m0 sc_m1 sc_m2  ... sc_m8 sc_m9 sc_m10 sc_m11 sc_m12
                0   201705   223191  1251  1182   773  ...   335   291    259    274    344
                1   201706   239887  1394  1255   855  ...   313   270    274    339    381
                2   201707   228594  1453  1305   850  ...   289   321    319    357    448
                3   201708   219271  1388  1169   698  ...   272   309    346    408    377
                4   201709   223009  1379  1151   556  ...   323   341    381    434    383
                5   201710   221593  1216   996   618  ...   406   376    416    416    398
                ...
                12  201805   216338   963  1030   696  ...   274   257    206    211    267
                13  201806   240976  1155  1174   753  ...   274   278    221    246    266
                14  201807   227943  1379  1114   708  ...   254   208    256    257    370
                15  201808   237674  1317  1066   613  ...   203   222    267    311    333
       In this example the scr can be calculated for the date=201805 which has a row_id=12, the col_id tells you
       the column id where the first months of service calls are located. In the example above
       col_id = 2 since there are 2 columns before column 'sc_m0'.
       So to get the scr value for date=201805 call the function as follows (the default qty_sold_colid=1 is taken):

        >> extract_scr(df, 12, 2)
    '''

    qty_sold_colid = kwargs.pop('qty_sold_colid', 1)
    debug = kwargs.pop('debug', 0)

    scr = (sum([df.iloc[row_id - j, col_id - 1 + j] for j in range(1, 2)]) + \
           sum([df.iloc[row_id - j, col_id - 1 + j] for j in range(2, 5)]) + \
           sum([df.iloc[row_id - j, col_id - 1 + j] for j in range(5, 13)]) \
           ) / \
          (0.25 * sum([df.iloc[row_id - j, qty_sold_colid] for j in range(1, 2)]) + \
           1 / 12 * sum([df.iloc[row_id - j, qty_sold_colid] for j in range(2, 5)]) + \
           1 / 16 * sum([df.iloc[row_id - j, qty_sold_colid] for j in range(5, 13)]) \
           )
    if debug:
        print('First factor: sales={}, sc={}'.format([df.iloc[row_id - j, col_id - 1 + j] for j in range(1, 2)],
                                                     [df.iloc[row_id - j, qty_sold_colid] for j in range(1, 2)]))
        print('Second factor: sales={}, sc={}'.format([df.iloc[row_id - j, col_id - 1 + j] for j in range(2, 5)],
                                                      [df.iloc[row_id - j, qty_sold_colid] for j in range(2, 5)]))
        print('Third factor: sales={}, sc={}'.format([df.iloc[row_id - j, col_id - 1 + j] for j in range(5, 13)],
                                                     [df.iloc[row_id - j, qty_sold_colid] for j in range(5, 13)]))

    return scr


def generate_exponential_weights(n, decay_rate=1):
    '''Exponential decaying weights, can be used for the batch_id weights. Most recent month will have bigger weight, hence the reverse at the end.

    :param n: in our case 11 (we have 12 months but only 11 are used for the one hot encoding)
    :param decay_rate: default =1
    :return: [2.87e-05, 7.807e-05, 0.00022, 0.00058, 0.0016, 0.0043, 0.012, 0.031, 0.086, 0.23, 0.63]
    '''
    weights = [np.exp(-decay_rate * i) for i in range(n)]
    total = sum(weights)
    normalized_weights = [w / total for w in weights]
    normalized_weights.reverse()
    return normalized_weights


def get_all_csv_files(in_dir, start_with='sc_table_', end_with='.csv'):
    '''
    Getting a list of files starting with a string defined in "start_with" and ending with a string defined in "end_with"
    This can be used to run over a list of different csv files.

    :param in_dir: directory in which csv search is done
    :param start_with: looking for all the files that start with the substring defined here
    :param end_with: and looking for all the files that also end with the substring defined here
    :return: list of files in the in_dir directory starting and ending with the substrings defined in start_with and end_with

    usage example: sorted(get_all_csv_files('data',start_with='sc_table_', end_with='.csv'))
    '''

    all_files_in_dir = os.listdir(in_dir)
    filtered_files = [file for file in all_files_in_dir if file.startswith(start_with) and file.endswith(end_with)]

    # remove the start_with part of the string as well as the end_with part
    stripped_file_names = [file[len(start_with):-len(end_with)] for file in filtered_files]

    return stripped_file_names


def get_hierarchy_dirname(hierarchy_str):
    '''
    Based on the hierarchy_str, a directory name is returned reflecting on if the hierarchy_str is a product line, group or subgroup.
    It counts the number of substrings "_all" in the hierarchy_str, when the count equals 2 the directroy will be 'line', when it equals 1 the directory name will be group and when the count equals 0 the dir name will be subgroup. This is used to save different hierarchy analysis in the corresponding subdirectories.

    :param hierarchy_str: e.g dishcare_all_all
    :return: for dishcare_all_all --> 'line', for dishcare_bislimlinedishwasher_all --> group
    '''
    all_count_map = {0: 'subgroup', 1: 'group', 2: 'line'}

    substr_all_count = hierarchy_str.count('_all')
    hierarchy_dir_name = all_count_map[substr_all_count]
    return hierarchy_dir_name


def get_mape_from_series(vol_diff):
    '''
    The Mean Absolute Percentage Error (MAPE) is calculated over the vol_diff panda series values and its STDEV, as well as the mdape (Median Abs Percentage Error).

    :param vol_diff: volume_difference is a pandas series that contains the percentage errors for each batch, the values can be negative as well as positive.
    :return mape, mdape, mapestd, mdape_std
    '''

    mape = np.nanmean(abs(vol_diff))
    mdape = np.nanmedian(abs(vol_diff))
    mape_std = np.nanstd(abs(vol_diff))
    mdape_std = np.nanstd(abs(vol_diff))

    return mape, mdape, mape_std, mdape_std


def get_mapes(serv_calls, pred_calls):
    '''
    When the inputs are numpy.ndarray for the actual service calls (serv_calls) and the predicted service calls (pred_calls)
    the mean absolute percentage error is calculated.

    :param serv_calls:
    array([ [ 821],
            [1074],
            [ 866],
            ...
            [ 794],
            [ 770]])

    :param pred_calls:
    array([ [ 764],
            [ 914],
            [ 727],
            ...
            [ 651],
            [ 696]])

    :return  array([11.95024241])
    '''
    # Find indices where serv_calls is zero to avoid division by zero
    zero_or_nan_indices = (serv_calls == 0) | np.isnan(serv_calls)

    # Replace zero or NaN values in serv_calls to avoid division by zero
    serv_calls[zero_or_nan_indices] = 1  # Set to 1 or any other non-zero value to avoid zero division

    # Calculate your metric (mean absolute percentage error in this case)
    absolute_percentage_error = np.mean(np.abs((serv_calls - pred_calls) / serv_calls * 100), axis=0)

    return absolute_percentage_error


def get_scr_df(df: pd.DataFrame) -> pd.DataFrame:
    ''' Extracts all the possible scr and forcasted scr from a given dataframe df of the form:
        df=
              date  qty_sold  sca_m0  sca_m1  ...  scf_m9  scf_m10  scf_m11  scf_m12
        0   201601    111920     821    1074  ...     868      951     1021      513
        1   201602    133784     970    1034  ...     954     1038     1066      550
        2   201603    132199     873     963  ...     812      853      928      432
        3   201604    127341     824     983  ...     697      774      751      344
        ...
        [79 rows x 28 columns]

        the output will be a dataframe with date (index), scr and scrf (forecasted SCR)
                      scr      scrf
            date
            201701  0.076555  0.073311
            201702  0.070084  0.068402
            201703  0.074905  0.069018
            201704  0.056880  0.057153
            201705  0.063427  0.059210

    '''

    scr_df = []

    # first 12 months are needed to calculate the SCR based on the history of the sales and service call forcasts
    if len(df) > 12:
        for i in range(12, len(df)):
            scr = extract_scr(df, i, 2)
            scrf = extract_scr(df, i, 15)
            scr_df.append([df.iloc[i, 0], scr, scrf])

    df_out = pd.DataFrame(scr_df, columns=['date', 'scr', 'scrf'])
    df_out.set_index('date', inplace=True)

    return df_out


def get_vol_diff_from_df(serie_act, serie_pred):
    return (serie_act - serie_pred) / serie_act * 100


def int_to_month(x):
    """Convert integer to month as string"""
    return [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ][x]


def load_service_call_data(*args):
    # Product LINE
    if args[0] == 'line':
        food_preservation_train_data = pd.DataFrame(
            columns=['date', 'qty_sold', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12',
                     'm13'])
        food_preservation_train_data['date'] = ['201705', '201706', '201707', '201708', '201709', '201710', '201711',
                                                '201712',
                                                '201801', '201802', '201803', '201804', '201805']
        food_preservation_train_data['qty_sold'] = [223191, 239887, 228594, 219271, 223009, 221593, 218133, 158218,
                                                    172558,
                                                    162798, 179055, 179193, 216338]
        food_preservation_train_data['m1'] = [1251, 1394, 1453, 1388, 1379, 1216, 1024, 910, 894, 951, 841, 910, 963]
        food_preservation_train_data['m2'] = [1182, 1255, 1305, 1169, 1151, 996, 975, 817, 777, 791, 795, 868, 1030]
        food_preservation_train_data['m3'] = [773, 855, 850, 698, 556, 618, 602, 508, 462, 556, 563, 610, 696]
        food_preservation_train_data['m4'] = [602, 658, 590, 471, 501, 447, 400, 388, 365, 444, 458, 521, 544]
        food_preservation_train_data['m5'] = [539, 496, 398, 412, 366, 368, 355, 316, 419, 432, 407, 425, 506]
        food_preservation_train_data['m6'] = [415, 383, 366, 359, 335, 331, 393, 405, 339, 413, 378, 387, 364]
        food_preservation_train_data['m7'] = [322, 350, 324, 247, 294, 364, 386, 376, 346, 326, 374, 312, 272]
        food_preservation_train_data['m8'] = [335, 313, 289, 272, 323, 406, 371, 380, 360, 364, 330, 232, 274]
        food_preservation_train_data['m9'] = [291, 270, 321, 309, 341, 376, 384, 335, 323, 283, 194, 228, 257]
        food_preservation_train_data['m10'] = [259, 274, 319, 346, 381, 416, 325, 351, 243, 213, 197, 199, 206]
        food_preservation_train_data['m11'] = [274, 339, 357, 408, 434, 416, 377, 278, 214, 222, 222, 212, 211]
        food_preservation_train_data['m12'] = [344, 381, 448, 377, 383, 398, 330, 237, 232, 193, 209, 182, 267]
        food_preservation_train_data['m13'] = [202, 239, 222, 183, 197, 177, 109, 135, 87, 113, 90, 118, 149]

        food_preservation_test_data = pd.DataFrame(
            columns=['date', 'qty_sold', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12',
                     'm13'])
        food_preservation_test_data['date'] = ['201806', '201807', '201808', '201809', '201810', '201811', '201812',
                                               '201901',
                                               '201902', '201903', '201904', '201905', '201906']
        food_preservation_test_data['qty_sold'] = [240976, 227943, 237674, 209602, 224913, 214440, 145946, 179610,
                                                   164794,
                                                   172637, 175147, 198479, 196969]
        food_preservation_test_data['m1'] = [1155, 1379, 1317, 1280, 1061, 864, 860, 754, 731, 695, 747, 855, 1038]
        food_preservation_test_data['m2'] = [1174, 1114, 1066, 923, 784, 791, 692, 640, 620, 675, 696, 818, 985]
        food_preservation_test_data['m3'] = [753, 708, 613, 486, 578, 464, 391, 357, 428, 395, 540, 570, 654]
        food_preservation_test_data['m4'] = [544, 477, 331, 370, 456, 333, 257, 333, 300, 384, 417, 466, 480]
        food_preservation_test_data['m5'] = [404, 322, 306, 285, 293, 244, 292, 235, 355, 352, 380, 403, 313]
        food_preservation_test_data['m6'] = [285, 317, 267, 281, 253, 282, 269, 275, 309, 322, 360, 287, 247]
        food_preservation_test_data['m7'] = [283, 303, 242, 209, 313, 278, 284, 266, 306, 266, 243, 258, 247]
        food_preservation_test_data['m8'] = [274, 254, 203, 280, 291, 289, 287, 250, 281, 246, 197, 238, 231]
        food_preservation_test_data['m9'] = [278, 208, 222, 257, 321, 337, 288, 264, 201, 187, 187, 201, 162]
        food_preservation_test_data['m10'] = [221, 256, 267, 276, 345, 306, 261, 203, 192, 167, 177, 169, 147]
        food_preservation_test_data['m11'] = [246, 257, 311, 336, 339, 304, 219, 183, 184, 159, 146, 110, 153]
        food_preservation_test_data['m12'] = [266, 370, 333, 327, 312, 245, 171, 186, 190, 144, 113, 160, 227]
        food_preservation_test_data['m13'] = [195, 167, 155, 176, 146, 123, 107, 72, 65, 37, 71, 98, 144]
    elif args[0] == 'group':
        food_preservation_train_data = pd.DataFrame(
            columns=['date', 'qty_sold', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12',
                     'm13'])
        food_preservation_train_data['date'] = ['201707', '201708', '201709', '201710', '201711', '201712', '201801',
                                                '201802', '201803', '201804', '201805', '201806', '201807']
        food_preservation_train_data['qty_sold'] = [23729, 25981, 20530, 20879, 20653, 18397, 17832, 15323, 16650,
                                                    18989, 23677, 25173, 26484]
        food_preservation_train_data['m1'] = [234, 204, 181, 156, 147, 107, 142, 113, 97, 136, 127, 185, 223]
        food_preservation_train_data['m2'] = [146, 132, 139, 122, 122, 105, 90, 88, 87, 92, 121, 160, 131]
        food_preservation_train_data['m3'] = [111, 70, 63, 70, 62, 54, 60, 62, 47, 53, 91, 81, 90]
        food_preservation_train_data['m4'] = [53, 56, 47, 64, 39, 46, 31, 50, 50, 66, 64, 58, 52]
        food_preservation_train_data['m5'] = [48, 63, 56, 47, 42, 36, 41, 46, 69, 48, 52, 59, 29]
        food_preservation_train_data['m6'] = [53, 48, 32, 39, 48, 41, 38, 40, 41, 56, 44, 23, 29]
        food_preservation_train_data['m7'] = [43, 35, 38, 47, 33, 37, 36, 42, 39, 30, 30, 23, 28]
        food_preservation_train_data['m8'] = [31, 40, 29, 43, 51, 53, 50, 42, 39, 22, 36, 21, 30]
        food_preservation_train_data['m9'] = [46, 32, 36, 40, 55, 40, 36, 25, 31, 26, 27, 36, 32]
        food_preservation_train_data['m10'] = [34, 28, 48, 36, 29, 46, 29, 26, 26, 20, 25, 21, 34]
        food_preservation_train_data['m11'] = [36, 44, 53, 38, 40, 27, 28, 17, 28, 24, 18, 29, 29]
        food_preservation_train_data['m12'] = [57, 40, 42, 51, 39, 25, 25, 20, 22, 13, 28, 34, 33]
        food_preservation_train_data['m13'] = [28, 23, 23, 17, 9, 16, 9, 9, 17, 15, 18, 19, 9]

        food_preservation_test_data = pd.DataFrame(
            columns=['date', 'qty_sold', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12',
                     'm13'])
        food_preservation_test_data['date'] = ['201806', '201807', '201808', '201809', '201810', '201811', '201812',
                                               '201901',
                                               '201902', '201903', '201904', '201905', '201906']
        food_preservation_test_data['qty_sold'] = [25173, 26484, 27886, 22984, 24367, 22148, 17384, 19826, 15599, 17373,
                                                   17698,
                                                   21296, 21159]
        food_preservation_test_data['m1'] = [185, 223, 214, 176, 132, 106, 132, 106, 89, 83, 94, 115, 145]
        food_preservation_test_data['m2'] = [160, 131, 128, 100, 94, 102, 68, 76, 62, 72, 82, 95, 137]
        food_preservation_test_data['m3'] = [81, 90, 61, 63, 53, 47, 44, 39, 34, 26, 58, 79, 89]
        food_preservation_test_data['m4'] = [58, 52, 43, 42, 83, 41, 30, 35, 28, 29, 33, 50, 44]
        food_preservation_test_data['m5'] = [59, 29, 35, 24, 23, 20, 27, 19, 38, 40, 42, 47, 31]
        food_preservation_test_data['m6'] = [23, 29, 23, 23, 14, 30, 36, 24, 41, 32, 36, 28, 26]
        food_preservation_test_data['m7'] = [23, 28, 25, 24, 40, 23, 38, 31, 23, 23, 24, 23, 19]
        food_preservation_test_data['m8'] = [21, 30, 22, 17, 23, 23, 23, 26, 25, 28, 16, 28, 22]
        food_preservation_test_data['m9'] = [36, 32, 22, 24, 24, 42, 38, 26, 30, 17, 20, 16, 16]
        food_preservation_test_data['m10'] = [21, 34, 28, 24, 31, 36, 30, 22, 16, 13, 23, 17, 13]
        food_preservation_test_data['m11'] = [29, 29, 31, 42, 28, 36, 24, 22, 15, 15, 18, 20, 12]
        food_preservation_test_data['m12'] = [34, 33, 41, 35, 25, 38, 14, 25, 24, 20, 8, 14, 23]
        food_preservation_test_data['m13'] = [19, 20, 20, 14, 14, 16, 14, 10, 4, 5, 7, 14, 19]
    elif args[0] == 'subgroup':
        food_preservation_train_data = pd.DataFrame(
            columns=['date', 'qty_sold', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12',
                     'm13'])
        food_preservation_train_data['date'] = ['201708', '201709', '201710', '201711', '201712', '201801', '201802',
                                                '201803', '201804', '201805', '201806', '201807', '201808']
        food_preservation_train_data['qty_sold'] = [25981, 20530, 20879, 20653, 18397, 17832, 15309, 16625, 18970,
                                                    23637, 25147, 26435, 27846]
        food_preservation_train_data['m1'] = [203, 181, 156, 147, 107, 142, 113, 97, 136, 126, 184, 222, 213]
        food_preservation_train_data['m2'] = [132, 139, 122, 122, 105, 90, 86, 87, 92, 120, 159, 131, 128]
        food_preservation_train_data['m3'] = [70, 63, 70, 62, 54, 60, 62, 47, 53, 90, 81, 90, 61]
        food_preservation_train_data['m4'] = [55, 47, 64, 39, 46, 31, 50, 50, 66, 64, 58, 52, 43]
        food_preservation_train_data['m5'] = [63, 56, 47, 42, 36, 41, 46, 69, 48, 52, 59, 28, 35]
        food_preservation_train_data['m6'] = [48, 31, 39, 48, 41, 38, 39, 41, 56, 44, 23, 29, 23]
        food_preservation_train_data['m7'] = [35, 38, 47, 33, 37, 36, 42, 39, 30, 29, 23, 28, 24]
        food_preservation_train_data['m8'] = [40, 28, 43, 51, 52, 50, 42, 39, 22, 36, 21, 30, 22]
        food_preservation_train_data['m9'] = [32, 36, 40, 55, 40, 36, 25, 31, 26, 27, 36, 32, 22]
        food_preservation_train_data['m10'] = [28, 48, 36, 29, 46, 29, 26, 26, 20, 25, 21, 34, 28]
        food_preservation_train_data['m11'] = [44, 53, 38, 40, 27, 28, 17, 27, 24, 18, 29, 28, 31]
        food_preservation_train_data['m12'] = [40, 42, 51, 39, 25, 25, 20, 22, 13, 28, 34, 33, 41]
        food_preservation_train_data['m13'] = [22, 23, 17, 9, 16, 9, 9, 17, 15, 18, 19, 20, 20]

        food_preservation_test_data = pd.DataFrame(
            columns=['date', 'qty_sold', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12',
                     'm13'])
        food_preservation_test_data['date'] = ['201806', '201807', '201808', '201809', '201810', '201811', '201812',
                                               '201901',
                                               '201902', '201903', '201904', '201905', '201906']
        food_preservation_test_data['qty_sold'] = [25147, 26435, 27846, 22954, 24323, 22121, 17354, 19793, 15566, 17334,
                                                   17680, 21264, 21129]
        food_preservation_test_data['m1'] = [184, 222, 213, 176, 132, 106, 132, 106, 89, 83, 93, 114, 144]
        food_preservation_test_data['m2'] = [159, 131, 128, 99, 94, 102, 67, 76, 62, 72, 82, 95, 137]
        food_preservation_test_data['m3'] = [81, 90, 61, 63, 53, 47, 44, 39, 34, 26, 58, 79, 89]
        food_preservation_test_data['m4'] = [58, 52, 43, 42, 83, 41, 30, 35, 28, 28, 33, 50, 44]
        food_preservation_test_data['m5'] = [59, 28, 35, 24, 23, 20, 27, 19, 38, 40, 42, 47, 31]
        food_preservation_test_data['m6'] = [23, 29, 23, 23, 13, 30, 36, 24, 41, 31, 35, 28, 26]
        food_preservation_test_data['m7'] = [23, 28, 24, 24, 40, 23, 38, 31, 23, 23, 24, 23, 19]
        food_preservation_test_data['m8'] = [21, 30, 22, 17, 23, 23, 23, 26, 25, 28, 16, 28, 22]
        food_preservation_test_data['m9'] = [36, 32, 22, 24, 24, 42, 36, 26, 30, 17, 20, 16, 16]
        food_preservation_test_data['m10'] = [21, 34, 28, 24, 31, 35, 30, 22, 16, 13, 23, 17, 13]
        food_preservation_test_data['m11'] = [29, 28, 31, 42, 28, 36, 23, 22, 15, 15, 18, 20, 12]
        food_preservation_test_data['m12'] = [34, 33, 41, 35, 25, 38, 14, 25, 24, 20, 8, 14, 23]
        food_preservation_test_data['m13'] = [19, 20, 20, 14, 14, 16, 14, 10, 4, 5, 7, 14, 19]
    else:
        print("choose on of the following options: line, group or subgroup")
    return food_preservation_train_data, food_preservation_test_data


def load_sunburst_data(which):
    if which == 'SCR':
        sunburst_data = [
            ['dishcare_all_all', 3.936798154054269, 3.099539496989019, 3.506190991372448, 3.506190991372448,
             136818.88607594935],
            ['dishcare_bifullsizedishwasherstainless_all', 3.899443131217493, 3.2590051457975986, 3.6073658414540497,
             3.6073658414540497, 92551.43037974683],
            ['dishcare_bislimlinedishwasher_all', 5.797070710217914, 4.708097928436914, 4.626717470108829,
             4.626717470108829, 16784.45569620253],
            ['dishcare_fsfullsizedishwasherstainless_all', 4.472826776135073, 3.8571428571428688, 4.0349478005426285,
             4.0349478005426285, 19695.98734177215],
            ['fabriccare_2460cmFLWashingMachineFS_all', 4.1125691477776725, 3.1639004149377534, 3.9035754620896066,
             3.9035754620896066, 100611.64556962025],
            ['fabriccare_2460cmFrontLoadDryerFreestanding_60cmElectricStandardCondenseDryer', 7.136245168163635,
             5.620043633341412, 5.416288632891441, 5.416288632891441, 12473.6625],
            ['fabriccare_2460cmFrontLoadDryerFreestanding_60cmElectricVentedDryer', 16.612345470214926,
             13.435828877005342, 12.761349660846264, 12.761349660846264, 3023.55],
            ['fabriccare_2460cmFrontLoadDryerFreestanding_60cmHeatPumpDryer', 7.811341262299535, 5.930596179613717,
             5.7041418579139735, 5.7041418579139735, 50738.9125],
            ['fabriccare_2460cmFrontLoadDryerFreestanding_all', 6.846257925901137, 5.920550038197095, 5.089967312928895,
             5.089967312928895, 66739.93670886075],
            ['fabriccare_BuiltinFrontLoadWashingMachine_BuiltinFrontLoadWashingMachine', 8.605147104790287,
             5.975524155400253, 7.718018284651161, 7.718018284651161, 6275.1],
            ['fabriccare_BuiltinFrontLoadWashingMachine_all', 7.930915006350024, 5.5900621118012435, 6.767012637260142,
             6.767012637260142, 6933.2531645569625],
            ['fabriccare_BuiltinWasherDryerFrontLoad_BuiltinWasherDryerFrontLoad', 9.807510715539884, 8.397681780874688,
             8.628841800878638, 8.628841800878638, 4200.9625],
            ['fabriccare_BuiltinWasherDryerFrontLoad_all', 9.756371877385753, 7.488986784140966, 8.576730389598175,
             8.576730389598175, 4393.139240506329],
            ['fabriccare_Dryer_ElectricCondenseTumbleDryer', 13.320444857076783, 13.324175824175832, 10.635449114736764,
             10.635449114736764, 10877.973684210527],
            ['fabriccare_Dryer_all', 12.932161036540359, 13.005406704313351, 9.961870267311422, 9.961870267311422,
             11100.842105263158],
            ['fabriccare_FSFullsizeWashingMachineFrontLoad_WashingMachineFrontLoad', 6.4694365506283384,
             5.349470535304597, 6.093245850873145, 6.093245850873145, 17202.8],
            ['fabriccare_FSFullsizeWashingMachineFrontLoad_all', 6.62618779463773, 5.115089514066501, 7.280453716742127,
             7.280453716742127, 18562.62162162162],
            ['fabriccare_FrontLoadWasherDryerFreestanding_FrontLoadWasherDryerFreestanding', 5.137931488847458,
             3.5825146870802245, 4.362663733456335, 4.362663733456335, 13649.775],
            ['fabriccare_FrontLoadWasherDryerFreestanding_WasherDryerFrontLoad', 11.117219810636541, 12.213302752293583,
             5.760004224229367, 5.760004224229367, 1254.6538461538462],
            ['fabriccare_FrontLoadWasherDryerFreestanding_all', 4.69946707622721, 2.890173410404626, 4.1993524861562,
             4.1993524861562, 14124.632911392406],
            ['fabriccare_HorizontalAxisTopLoadWashingMachine_HorizontalAxisTopLoadWashingMachine', 4.919029109473045,
             3.881147599525761, 3.7245103568819373, 3.7245103568819373, 28094.0],
            ['fabriccare_HorizontalAxisTopLoadWashingMachine_all', 5.000820116719365, 3.9383561643835647,
             3.8436457448303245, 3.8436457448303245, 28098.835443037973],
            ['fabriccare_SlimFrontLoadWashingMachineFS_SlimFrontLoadWashingMachineFS', 9.251199589158267,
             6.269690634243162, 10.693933307910108, 10.693933307910108, 20713.7625],
            ['fabriccare_SlimFrontLoadWashingMachineFS_all', 8.960013570312016, 6.243496357960468, 9.961173201093912,
             9.961173201093912, 21403.58227848101],
            ['fabriccare_all_all', 4.151076939639175, 3.665405632618751, 3.6552532508062847, 3.6552532508062847,
             260702.58227848102],
            ['foodpreparation_BuiltInOven_BuiltinCompactOven', 7.036413450345229, 5.161866294540322, 6.124814392645672,
             6.124814392645672, 12570.6125],
            ['foodpreparation_BuiltInOven_BuiltinDualCavityElectricOven', 9.897341398058643, 8.646372456267269,
             6.89123798375325, 6.89123798375325, 6318.9375],
            ['foodpreparation_BuiltInOven_BuiltinElectricOven', 5.375062039509915, 4.686584866606218, 4.49942800015828,
             4.49942800015828, 124340.275],
            ['foodpreparation_BuiltInOven_BuiltinSteamOven', 6.415040605695886, 4.433128263454471, 5.79431281177162,
             5.79431281177162, 14665.125],
            ['foodpreparation_BuiltInOven_all', 4.7935205036254445, 3.7754114230396945, 4.262921667670581,
             4.262921667670581, 160619.41772151898],
            ['foodpreparation_BuiltinHob_BuiltinGasHob', 11.593737569470768, 8.657463592233015, 10.689585728141283,
             10.689585728141283, 31577.8],
            ['foodpreparation_BuiltinHob_BuiltinInductionHob', 5.169914547160438, 4.018874643874641, 3.7702356665536225,
             3.7702356665536225, 82288.7875],
            ['foodpreparation_BuiltinHob_BuiltinMixedHob', 18.61987360164332, 17.232237539766697, 9.741762677453808,
             9.741762677453808, 1731.1625],
            ['foodpreparation_BuiltinHob_BuiltinRadiantHob', 6.125915189745204, 4.949878195615046, 4.54587584265598,
             4.54587584265598, 64138.1125],
            ['foodpreparation_BuiltinHob_all', 5.947097904471044, 4.880694143167031, 4.879797652920763,
             4.879797652920763, 183381.58227848102],
            ['foodpreparation_FreestandingCookerFrontControl_FSCookerFCElectCavityInductionHob', 6.962373225843513,
             5.897009966777407, 5.719583942072665, 5.719583942072665, 4823.5875],
            ['foodpreparation_FreestandingCookerFrontControl_FSCookerFCElectCavityRadiantHob', 6.4045245394330115,
             5.481262011531068, 4.8713986353789664, 4.8713986353789664, 21690.4875],
            ['foodpreparation_FreestandingCookerFrontControl_FSCookerFCElectDualCavityInducHob', 14.080346610123938,
             11.324786324786334, 12.011223234615434, 12.011223234615434, 1239.4625],
            ['foodpreparation_FreestandingCookerFrontControl_FSCookerFCElectDualCavityRadiaHob', 11.586815119344935,
             9.44940476190476, 10.461109569585975, 10.461109569585975, 3808.5625],
            ['foodpreparation_FreestandingCookerFrontControl_FSCookerFCElectricCavityGasHob', 10.890695757584814,
             8.244605000475332, 8.659026629647217, 8.659026629647217, 6973.1],
            ['foodpreparation_FreestandingCookerFrontControl_FSCookerFCGasCavityGasHob', 13.427343864411439,
             12.295839753466865, 8.966517460905447, 8.966517460905447, 6924.0625],
            ['foodpreparation_FreestandingCookerFrontControl_FSCookerFCGasDualCavityGasHob', 11.952842866405566,
             10.555555555555554, 8.742987639425355, 8.742987639425355, 1454.113924050633],
            ['foodpreparation_FreestandingCookerFrontControl_all', 3.885923543901643, 3.1847133757961785,
             3.4379539109382824, 3.4379539109382824, 52361.27848101266],
            ['foodpreparation_Hood_ChimneyDesignHood', 10.016479626203754, 7.545066413662242, 6.194318801430062,
             6.194318801430062, 5953.40625],
            ['foodpreparation_Hood_ChimneyStandardHood', 13.86104093174071, 11.842105263157892, 8.587663293073975,
             8.587663293073975, 9534.794871794871],
            ['foodpreparation_Hood_GroupHood', 10.872229593328193, 10.169491525423732, 8.637613602855225,
             8.637613602855225, 6512.461538461538],
            ['foodpreparation_Hood_PulloutHood', 10.475304261440813, 7.202108355655406, 8.843495928079953,
             8.843495928079953, 11323.95],
            ['foodpreparation_Hood_all', 6.456281959502139, 5.151915455746351, 6.010389292161947, 6.010389292161947,
             33150.794871794875],
            ['foodpreparation_OtherFoodPreparation_all', 8.578500707213578, 8.578500707213578, 4.2927864214992875,
             4.2927864214992875, 2399.0714285714284],
            ['foodpreparation_all_all', 4.661631472146733, 3.5130301481859987, 3.9881497257430194, 3.9881497257430194,
             415670.5189873418],
            ['foodpreservation_BuiltinFoodPreservation_BuiltinBottomFreezer', 6.057365744032895, 4.963370740267672,
             4.937392084435881, 4.937392084435881, 42928.1875],
            ['foodpreservation_BuiltinFoodPreservation_BuiltinTopFreezer', 15.523962023234322, 10.907729428405013,
             12.424743934431724, 12.424743934431724, 4773.8],
            ['foodpreservation_BuiltinFoodPreservation_BuiltinUnderCounterFreezer', 10.836538195122095,
             7.598039215686272, 8.76735842402683, 8.76735842402683, 3709.65],
            ['foodpreservation_BuiltinFoodPreservation_BuiltinUnderCounterRefrigerator', 6.165532169046059,
             4.666666666666675, 5.19183025612424, 5.19183025612424, 16005.7625],
            ['foodpreservation_BuiltinFoodPreservation_BuiltinUprightFreezer', 9.533044532839455, 8.484030589293743,
             6.956419637495031, 6.956419637495031, 2436.4],
            ['foodpreservation_BuiltinFoodPreservation_BuiltinUprightRefrigerator', 5.028964975338794,
             4.327473786236672, 4.436830976977645, 4.436830976977645, 17110.775],
            ['foodpreservation_BuiltinFoodPreservation_all', 4.352476457573821, 3.186484786090139, 3.976874320835023,
             3.976874320835023, 87012.5125],
            ['foodpreservation_ChestFreezer_LargeChestFreezer', 11.82702041154227, 9.577464788732392, 9.061323844969225,
             9.061323844969225, 8594.19642857143],
            ['foodpreservation_ChestFreezer_SmallChestFreezer', 14.165355066962766, 12.842278203723978,
             10.428641674485453, 10.428641674485453, 7225.096153846154],
            ['foodpreservation_ChestFreezer_all', 9.62392754757354, 7.5660401320802695, 7.7259211012966045,
             7.7259211012966045, 15382.017857142857],
            ['foodpreservation_FreestandingBottomFreezer_SmallBottomFreezer', 4.833620262113993, 3.351010075232493,
             4.286695537249871, 4.286695537249871, 20298.0],
            ['foodpreservation_FreestandingBottomFreezer_all', 4.712784691689093, 3.350970017636682, 4.435086272522913,
             4.435086272522913, 20413.73417721519],
            ['foodpreservation_FreestandingFreezer_FreestandingUprightFreezer', 6.545632589753217, 5.450602982237273,
             5.131688483330706, 5.131688483330706, 13097.075],
            ['foodpreservation_FreestandingFreezer_all', 6.565268268914286, 4.966887417218544, 5.322295841298798,
             5.322295841298798, 13177.974683544304],
            ['foodpreservation_FreestandingRefrigeratorFreezer_FreestandingBottomFreezer', 11.307119529434836,
             10.68279510635952, 8.045074104417399, 8.045074104417399, 3807.0178571428573],
            ['foodpreservation_FreestandingRefrigeratorFreezer_FreestandingTopFreeze', 22.356021638920947,
             21.21212121212121, 12.012461573153972, 12.012461573153972, 2610.836363636364],
            ['foodpreservation_FreestandingRefrigeratorFreezer_all', 10.068587105041967, 9.055192833282717,
             8.187720131841889, 8.187720131841889, 6242.706896551724],
            ['foodpreservation_FreestandingRefrigerator_FreestandingUprightRefrigerator', 5.937867676411749,
             5.483663587063587, 4.891370339205345, 4.891370339205345, 18126.25],
            ['foodpreservation_FreestandingRefrigerator_all', 5.875293972313533, 4.984487190851867, 4.620463520800608,
             4.620463520800608, 18292.51282051282],
            ['foodpreservation_FreestandingTopFreezer_SmallTopFreezer', 7.814076786148496, 6.597222222222225,
             6.468859730821011, 6.468859730821011, 11189.912280701754],
            ['foodpreservation_FreestandingTopFreezer_all', 8.030971778531994, 6.701428339555267, 6.4078764677297775,
             6.4078764677297775, 12150.551724137931],
            ['foodpreservation_FreestandingUnderCounter_FreestandingUnderCounterFreezer', 15.910441733236274,
             15.624999999999984, 10.067348402608927, 10.067348402608927, 5504.191780821918],
            ['foodpreservation_FreestandingUnderCounter_FreestandingUnderCounterRefrigerator', 14.239429860280074,
             11.42894461859979, 10.846181372614222, 10.846181372614222, 9609.236111111111],
            ['foodpreservation_FreestandingUnderCounter_all', 11.310353611291317, 9.032258064516126, 9.934486634819942,
             9.934486634819942, 13933.518987341773],
            ['foodpreservation_all_all', 4.242058088146517, 3.2639378969654285, 3.484523746087219, 3.484523746087219,
             177525.43037974683]]
    elif which == 'SCR_LQ':
        sunburst_data = [
            ['dishcare_all_all', 2.8416370335149237, 2.206695515389411, 2.3898824007250443, 2.3898824007250443,
             136818.88607594935],
            ['dishcare_bifullsizedishwasherstainless_all', 2.8393094660524625, 2.3926756669502303, 2.468443037720566,
             2.468443037720566, 92551.43037974683],
            ['dishcare_bislimlinedishwasher_all', 4.374200647772019, 3.7073884508129846, 3.3992819038080104,
             3.3992819038080104, 16784.45569620253],
            ['dishcare_fsfullsizedishwasherstainless_all', 2.8335115887602673, 2.200392563370045, 2.118195108167997,
             2.118195108167997, 19695.98734177215],
            ['fabriccare_2460cmFLWashingMachineFS_all', 2.7023267951748147, 2.3736451276935306, 2.3011450407468366,
             2.3011450407468366, 100611.64556962025],
            ['fabriccare_2460cmFrontLoadDryerFreestanding_60cmElectricStandardCondenseDryer', 5.159540681469005,
             5.0559766981607925, 3.838281696680068, 3.838281696680068, 12473.6625],
            ['fabriccare_2460cmFrontLoadDryerFreestanding_60cmElectricVentedDryer', 16.191703115680266,
             13.937895434006158, 11.07474697905924, 11.07474697905924, 3023.55],
            ['fabriccare_2460cmFrontLoadDryerFreestanding_60cmHeatPumpDryer', 6.969177436429662, 6.260632714482151,
             4.649675372484216, 4.649675372484216, 50738.9125],
            ['fabriccare_2460cmFrontLoadDryerFreestanding_all', 5.944150060788941, 5.089672780919662, 4.220652211944879,
             4.220652211944879, 66739.93670886075],
            ['fabriccare_BuiltinFrontLoadWashingMachine_BuiltinFrontLoadWashingMachine', 6.62897937768718,
             5.121037028402059, 5.433343439877837, 5.433343439877837, 6275.1],
            ['fabriccare_BuiltinFrontLoadWashingMachine_all', 5.941916785120362, 4.172793001092195, 4.720188424375107,
             4.720188424375107, 6933.2531645569625],
            ['fabriccare_BuiltinWasherDryerFrontLoad_BuiltinWasherDryerFrontLoad', 6.189736975526303,
             4.4140302536803215, 5.217265195179804, 5.217265195179804, 4200.9625],
            ['fabriccare_BuiltinWasherDryerFrontLoad_all', 6.168107078832866, 4.064984491367577, 5.309819094015983,
             5.309819094015983, 4393.139240506329],
            ['fabriccare_Dryer_ElectricCondenseTumbleDryer', 11.379101853714966, 12.269510747123064, 4.5705292189037925,
             4.5705292189037925, 10877.973684210527],
            ['fabriccare_Dryer_all', 10.78196672986985, 11.827549604549864, 4.888708490923525, 4.888708490923525,
             11100.842105263158],
            ['fabriccare_FSFullsizeWashingMachineFrontLoad_WashingMachineFrontLoad', 4.090276194485475,
             2.9150517175420756, 4.160364279486593, 4.160364279486593, 17202.8],
            ['fabriccare_FSFullsizeWashingMachineFrontLoad_all', 3.968119857932692, 3.344478865275782,
             3.381418669150868, 3.381418669150868, 18562.62162162162],
            ['fabriccare_FrontLoadWasherDryerFreestanding_FrontLoadWasherDryerFreestanding', 3.4009218971001833,
             3.136290420003189, 2.6193686220689383, 2.6193686220689383, 13649.775],
            ['fabriccare_FrontLoadWasherDryerFreestanding_WasherDryerFrontLoad', 10.881517182721629, 10.617647984710892,
             3.0993312700994453, 3.0993312700994453, 1254.6538461538462],
            ['fabriccare_FrontLoadWasherDryerFreestanding_all', 3.075301529237082, 2.427479191105179, 2.59377093507889,
             2.59377093507889, 14124.632911392406],
            ['fabriccare_HorizontalAxisTopLoadWashingMachine_HorizontalAxisTopLoadWashingMachine', 3.7574962646732897,
             3.4119884291749396, 2.6656407750540976, 2.6656407750540976, 28094.0],
            ['fabriccare_HorizontalAxisTopLoadWashingMachine_all', 3.9430861666737576, 3.2893750772988875,
             2.6096138460165657, 2.6096138460165657, 28098.835443037973],
            ['fabriccare_SlimFrontLoadWashingMachineFS_SlimFrontLoadWashingMachineFS', 6.761985259273424,
             5.027738553849174, 7.2602681669825975, 7.2602681669825975, 20713.7625],
            ['fabriccare_SlimFrontLoadWashingMachineFS_all', 6.38164413249704, 5.215865656344863, 4.908046682843383,
             4.908046682843383, 21403.58227848101],
            ['fabriccare_all_all', 3.1473986757997547, 2.1629652462209337, 2.520301387542889, 2.520301387542889,
             260702.58227848102],
            ['foodpreparation_BuiltInOven_BuiltinCompactOven', 5.619002854080394, 4.595748140081351, 4.69041839716934,
             4.69041839716934, 12570.6125],
            ['foodpreparation_BuiltInOven_BuiltinDualCavityElectricOven', 7.212915676217843, 6.198307069916308,
             5.490514682257221, 5.490514682257221, 6318.9375],
            ['foodpreparation_BuiltInOven_BuiltinElectricOven', 3.7930935098239926, 2.6504830273217106,
             3.541893700738491, 3.541893700738491, 124340.275],
            ['foodpreparation_BuiltInOven_BuiltinSteamOven', 5.04428879972448, 3.4970131886456306, 4.222395877386069,
             4.222395877386069, 14665.125],
            ['foodpreparation_BuiltInOven_all', 3.579749313991475, 2.509980043691558, 3.129811970598865,
             3.129811970598865, 160619.41772151898],
            ['foodpreparation_BuiltinHob_BuiltinGasHob', 10.089992953071851, 7.7559678059393615, 8.675193992310048,
             8.675193992310048, 31577.8],
            ['foodpreparation_BuiltinHob_BuiltinInductionHob', 4.117156469578821, 3.8358898562513972, 2.850411194623166,
             2.850411194623166, 82288.7875],
            ['foodpreparation_BuiltinHob_BuiltinMixedHob', 16.491885147514758, 16.52652196407661, 7.239895328733787,
             7.239895328733787, 1731.1625],
            ['foodpreparation_BuiltinHob_BuiltinRadiantHob', 3.5502375694314483, 2.9366832471964646, 3.125494199887399,
             3.125494199887399, 64138.1125],
            ['foodpreparation_BuiltinHob_all', 5.098890066901993, 4.166215285043182, 3.6744707298890766,
             3.6744707298890766, 183381.58227848102],
            ['foodpreparation_FreestandingCookerFrontControl_FSCookerFCElectCavityInductionHob', 4.876352539476239,
             4.02309069620695, 3.726904305478061, 3.726904305478061, 4823.5875],
            ['foodpreparation_FreestandingCookerFrontControl_FSCookerFCElectCavityRadiantHob', 4.474869757359753,
             3.997110816282843, 2.9066922160707347, 2.9066922160707347, 21690.4875],
            ['foodpreparation_FreestandingCookerFrontControl_FSCookerFCElectDualCavityInducHob', 9.839538074493646,
             9.157645637550068, 6.192693709536312, 6.192693709536312, 1239.4625],
            ['foodpreparation_FreestandingCookerFrontControl_FSCookerFCElectDualCavityRadiaHob', 7.814363212752834,
             7.060292660125423, 6.360590506183596, 6.360590506183596, 3808.5625],
            ['foodpreparation_FreestandingCookerFrontControl_FSCookerFCElectricCavityGasHob', 8.916543768477002,
             8.993946228425074, 6.094008586621882, 6.094008586621882, 6973.1],
            ['foodpreparation_FreestandingCookerFrontControl_FSCookerFCGasCavityGasHob', 10.224524884608574,
             9.335333755755153, 6.620494350149022, 6.620494350149022, 6924.0625],
            ['foodpreparation_FreestandingCookerFrontControl_FSCookerFCGasDualCavityGasHob', 9.190850296953336,
             9.642608435979511, 5.021280680055724, 5.021280680055724, 1454.113924050633],
            ['foodpreparation_FreestandingCookerFrontControl_all', 2.2988472755930043, 2.0973124799082163,
             1.9407504018908122, 1.9407504018908122, 52361.27848101266],
            ['foodpreparation_Hood_ChimneyDesignHood', 7.448012020672684, 5.499946467393445, 6.016145184789017,
             6.016145184789017, 5953.40625],
            ['foodpreparation_Hood_ChimneyStandardHood', 11.246552550473048, 11.643882353942969, 7.004686394164222,
             7.004686394164222, 9534.794871794871],
            ['foodpreparation_Hood_GroupHood', 8.296804550760013, 6.375110483884031, 7.036202402828096,
             7.036202402828096, 6512.461538461538],
            ['foodpreparation_Hood_PulloutHood', 7.622720428768925, 5.405259889997449, 5.772956001384817,
             5.772956001384817, 11323.95],
            ['foodpreparation_Hood_all', 5.573518272670847, 4.825022795943045, 4.404876894962496, 4.404876894962496,
             33150.794871794875],
            ['foodpreparation_all_all', 3.877093651216277, 3.5335772511893184, 2.608578943457868, 2.608578943457868,
             415670.5189873418],
            ['foodpreservation_BuiltinFoodPreservation_BuiltinBottomFreezer', 4.692222417815583, 3.586074049482151,
             3.574340748628828, 3.574340748628828, 42928.1875],
            ['foodpreservation_BuiltinFoodPreservation_BuiltinTopFreezer', 13.428100340278421, 11.09847101683598,
             8.97233058047244, 8.97233058047244, 4773.8],
            ['foodpreservation_BuiltinFoodPreservation_BuiltinUnderCounterFreezer', 9.301724362647265,
             7.630796686974586, 6.72413672519367, 6.72413672519367, 3709.65],
            ['foodpreservation_BuiltinFoodPreservation_BuiltinUnderCounterRefrigerator', 4.664356432981395,
             4.270150490862308, 3.2338209623263103, 3.2338209623263103, 16005.7625],
            ['foodpreservation_BuiltinFoodPreservation_BuiltinUprightFreezer', 7.6669422591349345, 7.89046865695296,
             4.522858663736791, 4.522858663736791, 2436.4],
            ['foodpreservation_BuiltinFoodPreservation_BuiltinUprightRefrigerator', 3.4011777903339824,
             3.280518912230649, 2.013005863038769, 2.013005863038769, 17110.775],
            ['foodpreservation_BuiltinFoodPreservation_all', 3.3283564841288578, 2.438826166030443, 2.829082930510188,
             2.829082930510188, 87012.5125],
            ['foodpreservation_ChestFreezer_LargeChestFreezer', 10.192787712755308, 9.7214226394657, 6.39739173536576,
             6.39739173536576, 8594.19642857143],
            ['foodpreservation_ChestFreezer_SmallChestFreezer', 12.37664623768403, 12.263334694525668,
             8.213324194322002, 8.213324194322002, 7225.096153846154],
            ['foodpreservation_ChestFreezer_all', 8.280714173910846, 8.335723852642417, 5.50629227896407,
             5.50629227896407, 15382.017857142857],
            ['foodpreservation_FreestandingBottomFreezer_SmallBottomFreezer', 3.534550450548674, 3.3474713921241426,
             3.149668399312251, 3.149668399312251, 20298.0],
            ['foodpreservation_FreestandingBottomFreezer_all', 3.4802840157916277, 2.8963481279180083,
             3.070407486888499, 3.070407486888499, 20413.73417721519],
            ['foodpreservation_FreestandingFreezer_FreestandingUprightFreezer', 4.829800731903401, 3.617379819854211,
             3.67151618241876, 3.67151618241876, 13097.075],
            ['foodpreservation_FreestandingFreezer_all', 4.906849509290349, 3.4720708175186012, 3.6825042120905125,
             3.6825042120905125, 13177.974683544304],
            ['foodpreservation_FreestandingRefrigeratorFreezer_FreestandingBottomFreezer', 9.844455978977502,
             9.05471083835109, 7.008121466345048, 7.008121466345048, 3807.0178571428573],
            ['foodpreservation_FreestandingRefrigeratorFreezer_FreestandingTopFreeze', 20.22813207823094,
             22.54438108493273, 9.147168807690477, 9.147168807690477, 2610.836363636364],
            ['foodpreservation_FreestandingRefrigeratorFreezer_all', 7.456412475667241, 5.819488298911262,
             5.873112906633879, 5.873112906633879, 6242.706896551724],
            ['foodpreservation_FreestandingRefrigerator_FreestandingUprightRefrigerator', 4.013683190365864,
             3.496877111656911, 2.8479685316311563, 2.8479685316311563, 18126.25],
            ['foodpreservation_FreestandingRefrigerator_all', 3.8042391083089546, 3.1669652277760987, 2.796754063162316,
             2.796754063162316, 18292.51282051282],
            ['foodpreservation_FreestandingTopFreezer_SmallTopFreezer', 5.8125040316974745, 4.178687620270356,
             5.140243957282114, 5.140243957282114, 11189.912280701754],
            ['foodpreservation_FreestandingTopFreezer_all', 5.860783162404989, 4.289136583090893, 4.61667949674561,
             4.61667949674561, 12150.551724137931],
            ['foodpreservation_FreestandingUnderCounter_FreestandingUnderCounterFreezer', 12.945894418271587,
             12.622055192575306, 8.485211198363167, 8.485211198363167, 5504.191780821918],
            ['foodpreservation_FreestandingUnderCounter_FreestandingUnderCounterRefrigerator', 12.349674163476971,
             10.18529550792751, 9.176334537984712, 9.176334537984712, 9609.236111111111],
            ['foodpreservation_FreestandingUnderCounter_all', 9.39161596074968, 7.81602122592701, 6.984394099941319,
             6.984394099941319, 13933.518987341773],
            ['foodpreservation_all_all', 3.219839054899099, 2.527789352098535, 2.7106901085332056, 2.7106901085332056,
             177525.43037974683]]
    elif which == 'SCR_LY':
        sunburst_data = [
            ['dishcare_all_all', 1.6382378966811009, 0.951791554899823, 1.4424089432695864, 1.4424089432695864,
             136818.88607594935],
            ['dishcare_bifullsizedishwasherstainless_all', 1.8367281834212072, 1.2253912100109259, 1.6116304654115186,
             1.6116304654115186, 92551.43037974683],
            ['dishcare_bislimlinedishwasher_all', 2.3697700986469554, 1.812418680713701, 1.8754788501492348,
             1.8754788501492348, 16784.45569620253],
            ['dishcare_fsfullsizedishwasherstainless_all', 1.5290840557409593, 1.2757127131003658, 1.0520902825124367,
             1.0520902825124367, 19695.98734177215],
            ['fabriccare_2460cmFLWashingMachineFS_all', 1.5618391566259173, 1.368661572006102, 0.9011473841859078,
             0.9011473841859078, 100611.64556962025],
            ['fabriccare_2460cmFrontLoadDryerFreestanding_60cmElectricStandardCondenseDryer', 3.0116651416295155,
             2.990300899519722, 1.766707118567728, 1.766707118567728, 12473.6625],
            ['fabriccare_2460cmFrontLoadDryerFreestanding_60cmElectricVentedDryer', 17.445566649386144,
             15.741728355672521, 8.719409724382931, 8.719409724382931, 3023.55],
            ['fabriccare_2460cmFrontLoadDryerFreestanding_60cmHeatPumpDryer', 5.125305617437318, 5.89807093128152,
             3.181044786312147, 3.181044786312147, 50738.9125],
            ['fabriccare_2460cmFrontLoadDryerFreestanding_all', 4.0172116653838925, 3.8236545827647954,
             2.488314437455876, 2.488314437455876, 66739.93670886075],
            ['fabriccare_BuiltinFrontLoadWashingMachine_BuiltinFrontLoadWashingMachine', 5.996512402158569,
             5.307344030195642, 3.3521205600902313, 3.3521205600902313, 6275.1],
            ['fabriccare_BuiltinFrontLoadWashingMachine_all', 5.079883404689424, 4.82393896909679, 2.6404003684173425,
             2.6404003684173425, 6933.2531645569625],
            ['fabriccare_BuiltinWasherDryerFrontLoad_BuiltinWasherDryerFrontLoad', 4.279402695539443, 4.281400508672858,
             1.8990645985279073, 1.8990645985279073, 4200.9625],
            ['fabriccare_BuiltinWasherDryerFrontLoad_all', 4.404200845880135, 4.319978058258153, 1.8266373202982937,
             1.8266373202982937, 4393.139240506329],
            ['fabriccare_Dryer_ElectricCondenseTumbleDryer', 9.40223492723692, 8.829110889684266, 4.106567588636279,
             4.106567588636279, 10877.973684210527],
            ['fabriccare_Dryer_all', 8.406032110443457, 8.410679393083258, 4.521092708041049, 4.521092708041049,
             11100.842105263158],
            ['fabriccare_FSFullsizeWashingMachineFrontLoad_WashingMachineFrontLoad', 2.7090928220316424,
             2.8342425039146533, 0.8538453055015326, 0.8538453055015326, 17202.8],
            ['fabriccare_FSFullsizeWashingMachineFrontLoad_all', 3.1390983400983568, 3.275093720427928,
             0.8238713402496752, 0.8238713402496752, 18562.62162162162],
            ['fabriccare_FrontLoadWasherDryerFreestanding_FrontLoadWasherDryerFreestanding', 2.0743771734415732,
             1.7618975282025322, 1.6319719093967207, 1.6319719093967207, 13649.775],
            ['fabriccare_FrontLoadWasherDryerFreestanding_WasherDryerFrontLoad', 11.062402888098239, 10.867402701931065,
             0.5104534361880643, 0.5104534361880643, 1254.6538461538462],
            ['fabriccare_FrontLoadWasherDryerFreestanding_all', 1.7961338882002416, 1.0647292030031157,
             1.7410794490207264, 1.7410794490207264, 14124.632911392406],
            ['fabriccare_HorizontalAxisTopLoadWashingMachine_HorizontalAxisTopLoadWashingMachine', 1.8236085423686275,
             1.6215849977109522, 1.2240062263331406, 1.2240062263331406, 28094.0],
            ['fabriccare_HorizontalAxisTopLoadWashingMachine_all', 2.066940173641102, 1.8324742610989118,
             1.3554721901918925, 1.3554721901918925, 28098.835443037973],
            ['fabriccare_SlimFrontLoadWashingMachineFS_SlimFrontLoadWashingMachineFS', 3.973212672046632,
             3.5338805821123067, 2.6713079527003853, 2.6713079527003853, 20713.7625],
            ['fabriccare_SlimFrontLoadWashingMachineFS_all', 4.263292616710673, 4.2139702705411475, 2.829248046814241,
             2.829248046814241, 21403.58227848101],
            ['fabriccare_all_all', 1.8315700591545097, 1.4381375627974557, 1.3587788559019265, 1.3587788559019265,
             260702.58227848102],
            ['foodpreparation_BuiltInOven_BuiltinCompactOven', 4.735946933417597, 4.909245825104646, 2.9996309411605684,
             2.9996309411605684, 12570.6125],
            ['foodpreparation_BuiltInOven_BuiltinDualCavityElectricOven', 5.271451296682059, 4.224473718353201,
             3.192492502880999, 3.192492502880999, 6318.9375],
            ['foodpreparation_BuiltInOven_BuiltinElectricOven', 2.776089462387389, 2.235625847799012,
             1.9552705801885213, 1.9552705801885213, 124340.275],
            ['foodpreparation_BuiltInOven_BuiltinSteamOven', 3.483803069158919, 3.0969897060538347, 1.927613737237927,
             1.927613737237927, 14665.125],
            ['foodpreparation_BuiltInOven_all', 2.700874360452326, 2.5989965945076063, 1.6575799470531178,
             1.6575799470531178, 160619.41772151898],
            ['foodpreparation_BuiltinHob_BuiltinGasHob', 8.097859264469326, 6.994472650589042, 5.9972624799148955,
             5.9972624799148955, 31577.8],
            ['foodpreparation_BuiltinHob_BuiltinInductionHob', 2.496450457561762, 1.7576865995706978, 2.148822998781516,
             2.148822998781516, 82288.7875],
            ['foodpreparation_BuiltinHob_BuiltinMixedHob', 17.466857801256243, 17.448910237723105, 3.731302578220495,
             3.731302578220495, 1731.1625],
            ['foodpreparation_BuiltinHob_BuiltinRadiantHob', 2.002310261374141, 1.6711657729198426, 1.3479659648493743,
             1.3479659648493743, 64138.1125],
            ['foodpreparation_BuiltinHob_all', 3.7644674641018874, 3.7959180569990347, 1.8008715227242769,
             1.8008715227242769, 183381.58227848102],
            ['foodpreparation_FreestandingCookerFrontControl_FSCookerFCElectCavityInductionHob', 3.5819204069061237,
             2.7314130214248284, 2.0016975894395572, 2.0016975894395572, 4823.5875],
            ['foodpreparation_FreestandingCookerFrontControl_FSCookerFCElectCavityRadiantHob', 2.694120223754534,
             2.261907516987169, 2.0452819937647946, 2.0452819937647946, 21690.4875],
            ['foodpreparation_FreestandingCookerFrontControl_FSCookerFCElectDualCavityInducHob', 7.553680840874497,
             7.778964579171052, 3.8229402070330076, 3.8229402070330076, 1239.4625],
            ['foodpreparation_FreestandingCookerFrontControl_FSCookerFCElectDualCavityRadiaHob', 5.154346616607134,
             4.369197791162236, 4.157388419568186, 4.157388419568186, 3808.5625],
            ['foodpreparation_FreestandingCookerFrontControl_FSCookerFCElectricCavityGasHob', 6.613792439715146,
             6.516438055926161, 3.7777524307644392, 3.7777524307644392, 6973.1],
            ['foodpreparation_FreestandingCookerFrontControl_FSCookerFCGasCavityGasHob', 7.452689352817623,
             8.397909001258295, 3.627718917118329, 3.627718917118329, 6924.0625],
            ['foodpreparation_FreestandingCookerFrontControl_FSCookerFCGasDualCavityGasHob', 7.02976223425682,
             6.969989325295553, 3.284957256365084, 3.284957256365084, 1454.113924050633],
            ['foodpreparation_FreestandingCookerFrontControl_all', 1.1925516612133473, 1.0167585757192863,
             0.8362591977656303, 0.8362591977656303, 52361.27848101266],
            ['foodpreparation_Hood_ChimneyDesignHood', 4.850962587125224, 4.360743927984913, 2.693576850271434,
             2.693576850271434, 5953.40625],
            ['foodpreparation_Hood_ChimneyStandardHood', 12.906441153149746, 12.97974728054827, 1.970447957009296,
             1.970447957009296, 9534.794871794871],
            ['foodpreparation_Hood_GroupHood', 9.063035511844138, 8.881284687695945, 3.210007911515145,
             3.210007911515145, 6512.461538461538],
            ['foodpreparation_Hood_PulloutHood', 6.13984730649852, 7.158602325422346, 2.3261910200954032,
             2.3261910200954032, 11323.95],
            ['foodpreparation_Hood_all', 3.829240355789471, 4.280008649972416, 2.2243116887713037, 2.2243116887713037,
             33150.794871794875],
            ['foodpreparation_all_all', 2.792376776581579, 2.658240890428274, 1.4014440795049756, 1.4014440795049756,
             415670.5189873418],
            ['foodpreservation_BuiltinFoodPreservation_BuiltinBottomFreezer', 2.3977709933017666, 1.9614595005308573,
             1.6954406115437959, 1.6954406115437959, 42928.1875],
            ['foodpreservation_BuiltinFoodPreservation_BuiltinTopFreezer', 12.036015465217137, 12.671031014268438,
             5.707795397169173, 5.707795397169173, 4773.8],
            ['foodpreservation_BuiltinFoodPreservation_BuiltinUnderCounterFreezer', 7.729032684624161,
             6.873216939019181, 2.5748021078547207, 2.5748021078547207, 3709.65],
            ['foodpreservation_BuiltinFoodPreservation_BuiltinUnderCounterRefrigerator', 3.9158652187416387,
             4.085040011013088, 2.13730744158873, 2.13730744158873, 16005.7625],
            ['foodpreservation_BuiltinFoodPreservation_BuiltinUprightFreezer', 7.4673889240297875, 7.433952764572218,
             2.310521206630559, 2.310521206630559, 2436.4],
            ['foodpreservation_BuiltinFoodPreservation_BuiltinUprightRefrigerator', 2.1209629636100384,
             2.2497362935710683, 1.1978404323549738, 1.1978404323549738, 17110.775],
            ['foodpreservation_BuiltinFoodPreservation_all', 1.8046785701561807, 1.432229933900205, 1.5268956789436037,
             1.5268956789436037, 87012.5125],
            ['foodpreservation_ChestFreezer_LargeChestFreezer', 5.172282374367009, 5.904555952838329, 2.952382990831245,
             2.952382990831245, 8594.19642857143],
            ['foodpreservation_ChestFreezer_SmallChestFreezer', 10.25231487869235, 10.474251833357535,
             1.4430446689334702, 1.4430446689334702, 7225.096153846154],
            ['foodpreservation_ChestFreezer_all', 2.934726482830283, 3.1610832832892983, 1.4656932627056154,
             1.4656932627056154, 15382.017857142857],
            ['foodpreservation_FreestandingBottomFreezer_SmallBottomFreezer', 1.9747278977891682, 0.8238388936056945,
             2.4234661210374147, 2.4234661210374147, 20298.0],
            ['foodpreservation_FreestandingBottomFreezer_all', 1.8539880201302121, 0.7523748428911776,
             2.3305536708466232, 2.3305536708466232, 20413.73417721519],
            ['foodpreservation_FreestandingFreezer_FreestandingUprightFreezer', 2.6974815375447787, 2.7279362144170136,
             1.5585422348975941, 1.5585422348975941, 13097.075],
            ['foodpreservation_FreestandingFreezer_all', 2.7003825872031926, 2.508120269118484, 1.501526214486694,
             1.501526214486694, 13177.974683544304],
            ['foodpreservation_FreestandingRefrigeratorFreezer_FreestandingBottomFreezer', 7.543085444639816,
             7.118937980594452, 4.565573294684679, 4.565573294684679, 3807.0178571428573],
            ['foodpreservation_FreestandingRefrigeratorFreezer_FreestandingTopFreeze', 20.65066089871388,
             22.318269720979167, 3.436991665635787, 3.436991665635787, 2610.836363636364],
            ['foodpreservation_FreestandingRefrigeratorFreezer_all', 5.519460838328597, 4.592339719600018,
             3.147065200194454, 3.147065200194454, 6242.706896551724],
            ['foodpreservation_FreestandingRefrigerator_FreestandingUprightRefrigerator', 1.6006140810035547,
             1.5110397814747816, 0.947578457488553, 0.947578457488553, 18126.25],
            ['foodpreservation_FreestandingRefrigerator_all', 1.6312438356500918, 1.511718055999347, 0.9884398688786505,
             0.9884398688786505, 18292.51282051282],
            ['foodpreservation_FreestandingTopFreezer_SmallTopFreezer', 2.019822432176484, 1.923722423929171,
             1.1539615678767376, 1.1539615678767376, 11189.912280701754],
            ['foodpreservation_FreestandingTopFreezer_all', 1.2515229647302915, 1.120671079177283, 0.8056791226692573,
             0.8056791226692573, 12150.551724137931],
            ['foodpreservation_FreestandingUnderCounter_FreestandingUnderCounterFreezer', 11.561837619298657,
             11.314586436965211, 3.9094461546805177, 3.9094461546805177, 5504.191780821918],
            ['foodpreservation_FreestandingUnderCounter_FreestandingUnderCounterRefrigerator', 9.513406524228843,
             8.078712201037579, 6.4501997970906295, 6.4501997970906295, 9609.236111111111],
            ['foodpreservation_FreestandingUnderCounter_all', 5.246452597418199, 4.333028403164306, 3.926070866829796,
             3.926070866829796, 13933.518987341773],
            ['foodpreservation_all_all', 1.6073927994583317, 1.3074412401897324, 1.1177158630752662, 1.1177158630752662,
             177525.43037974683]]
    else:
        sunburst_data = []

    return sunburst_data


def make_sample_size_study():
    sunburst_data = load_sunburst_data('SCR')
    df_sun = pd.DataFrame(sunburst_data, columns=['hierarchy', 'mape', 'mdape', 'mape_std', 'mdape_std', 'qty'])

    # Assuming df_sun is your DataFrame

    # Categorizing the hierarchy based on conditions
    def categorize_hierarchy(hierarchy):
        if hierarchy.count("_all") == 2:
            return "Product Line"
        elif hierarchy.count("_all") == 1:
            return "Product Group"
        else:
            return "Product Subgroup"

    # Creating a new column for category
    df_sun['category'] = df_sun['hierarchy'].apply(categorize_hierarchy)

    # Creating a column for log(qty)
    df_sun['log_qty'] = np.log(df_sun['qty'])

    # Creating the scatter plot using Plotly Express
    fig = px.scatter(df_sun, x='mape', y='log_qty', color='category', hover_data=['hierarchy'])
    fig.update_traces(marker=dict(size=12))
    # Customizing the layout and displaying the plot
    fig.update_layout(title='Scatter plot of MAPE vs log(Qty) with Hierarchy Categories',
                      xaxis_title='MAPE',
                      yaxis_title='log(<Qty>)',
                      xaxis=dict(title_font=dict(size=29, family='Arial', color='black'),
                                 # Adjust the font properties for x-axis label
                                 tickfont=dict(size=16, family='Arial', color='black')),
                      # Adjust the font properties for x-axis ticks
                      yaxis=dict(title_font=dict(size=20, family='Arial', color='black'),
                                 # Adjust the font properties for y-axis label
                                 tickfont=dict(size=16, family='Arial', color='black')),
                      # Adjust the font properties for y-axis ticks)
                      legend=dict(
                          x=0.80,  # Adjust the x-coordinate to position the legend
                          y=0.98,  # Adjust the y-coordinate to position the legend
                          traceorder='normal',
                          font=dict(family='Arial', size=12, color='black'),
                          bgcolor='rgba(255, 255, 255, 0.5)',
                          bordercolor='rgba(0, 0, 0, 0.5)',
                          borderwidth=2)

                      )
    fig.show()

    # Save the figure as an HTML file
    pio.write_html(fig, file=os.path.join('plotly', 'qty_segmentation_mape.html'), auto_open=True)


def next_month(yyyymm):
    year = int(yyyymm[:4])
    month = int(yyyymm[4:])

    # Calculate the next month and year
    if month < 12:
        next_month = month + 1
        next_year = year
    else:
        next_month = 1
        next_year = year + 1

    # Format the next year and month as "yyyymm"
    next_month_str = f"{next_year:04d}{next_month:02d}"

    return next_month_str


def read_my_sc_csv(my_csv_file):
    column_indices_to_read = [0, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29]
    column_names = ['date', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12', 'qty_sold']
    df = pd.read_csv(my_csv_file, sep=';', skiprows=12, usecols=column_indices_to_read, names=column_names)

    my_df = df.copy()
    for col in range(1, 13):  # Columns m1 through m12
        df[f'm{col}'] = df[f'm{col}'].shift(-col)

    # Fill NaN values with a specific integer value (e.g., 0)
    df = df.fillna(-1)

    for col in range(1, 13):  # Columns m1 through m12
        df[f'm{col}'] = df[f'm{col}'].astype(int)

    # Remove rows where column 'm12' has a value of -1
    df = df[df['m12'] != -1]

    df['date'] = df['date'].astype(str)

    return df[['date', 'qty_sold', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12']]


def reorder_df(df2):
    ''' Transform the original df where the rows reflect on the batch_id (e.g. 201501, 201502, ...) and the columns
    contain the qty_sold and the nr of service calls for the different warranty periods (e.g. m0, m1, ...m12), into a df
    where the columns will reflect on the batch_id and the firs row contains the qty_sold and the rest of the rows
    contain the service calls for the respective warranty periods.

    :param df2: dataframe with shape (nr_batches, 15)
                columns = ['period', 'qty_sold', 'm0', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7',
                                                                                     'm8', 'm9', 'm10', 'm11', 'm12']
    :return: dataframe with shape (14, nr_batches)
             df.columns = Index(['201501', '201502',...,'202207'], dtype='object', name='date')
             df.index = Index(['qty_sold', 'm0', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9',
                                           'm10', 'm11', 'm12'], dtype='object')
    '''
    df2 = df2.rename(columns={'period': 'date'})
    df2 = df2.sort_values(by='date')
    df2 = df2.T

    # renaming the columns by the yyyymm (which is the first row of the df), and removing that row from the df
    df2.columns = df2.iloc[0]
    df2 = df2[1:]
    df2.columns = df2.columns.astype(str)
    return df2


def sc_mi_mat(w_mat, y_mat):
    ''' Recursive reconstruction of the 13 number of service calls (mi with i=0,1,2,3,...,12) for each batch.
    Here we loop over the different batches and we call the sc_mi function that is dealing with the recursive
    reconstruction for one particular batch.

    w_mat is typicall an ndarray with shape (13,nr_batches) that represents the quantity of sold items that remain and
    which are able to give their first service call. This quantity is calculated based on the following formula:
    qty_remaining = qty_sold - np.cumsum(service_calls, axis=0) + service_calls

    y_mat = service_calls / qty_remaining
    y_mat array([0.00676984, 0.00724149, 0.00490521, 0.0042833 , 0.00397329,
       0.00378406, 0.00411328, 0.0038506 , 0.00370679, 0.00395746,
       0.00383787, 0.00409551, 0.00213264])

    output is the number of service calls, one column per batch, rows are the different mi's going from m0, m1, ..., m12
    example: >>> np.column_stack(sc_mat)
                array([[1917, 2002, 1948],
                       [1933, 1917, 1899],
                       [1168, 1178, 1244],
                       [ 880,  947,  952],
                       [ 769,  787,  815],
                       [ 647,  682,  817],
                       [ 590,  720,  796],
                       [ 620,  698,  745],
                       [ 621,  675,  730],
                       [ 609,  671,  644],
                       [ 643,  629,  723],
                       [ 626,  734,  752],
                       [ 354,  369,  359]])
    '''

    sc_mat = []

    # loop over number of batches given
    for j in range(w_mat.shape[1]):
        # all service calls from m0, m1, ..., m12
        sc_mat.append(sc_mi(w_mat[:, j], y_mat[:, j]))
    return np.column_stack(sc_mat)


def sc_mi(wi, yi, all_mi=None):
    ''' Recursive reconstruction of the 13 number of service calls (mi with i=0,1,2,3,...,12) for one particular batch.
    sc(m0) = w0.y0
    sc(m1) = (w1-sc(m0)).y1
    sc(m2) = (w2-sc(m1)).y2
    ...
    sc(m12) = (w12-sc(m11)).y12
    all the wi's are actually the same value being w0
    Doing this you get the real number of service calls received in that mparticular month of the warranty.
    '''
    if all_mi is None:
        all_mi = []

    if isinstance(wi, np.ndarray):
        wi = list(wi)

    if isinstance(yi, np.ndarray):
        yi = list(yi)

    if (len(wi) == len(yi)) and (len(wi) == 1):
        sc = int(wi[0] * yi[0])
        all_mi.append(sc)
        return all_mi
    else:
        wi_first = wi.pop(-1)
        yi_first = yi.pop(-1)
        if len(wi) > 0:
            previous_sc = sc_mi(wi, yi, all_mi)
            sc = int((wi_first - previous_sc[-1]) * yi_first)
            all_mi.append(sc)
            return all_mi
        else:
            return all_mi


def stripping_future_off(df_from_hive_with_future):
    # Strips the dataframe into 3 dataframes:
    # 1) df_sc: the service calls part (future months with nans are stripped off) From 201501 till 202310
    # 2) df_fs: the pure sales forcast (Only future months are considered) From 202311 till end of forecast
    # 3) df_as_fs: the fully available actual and forecasted sales(can be used to show the actual sales
    #    versus predicted sales) From start of the forecast (i.e. 201901) till end of the forecast

    # check when the qty_sold start to be nan, the first nan encountered is the first month of future predictions
    # When today is Nov 2023 (202311), the index should then point at 202310
    nan_indices = df_from_hive_with_future[df_from_hive_with_future['qty_sold'].isna()].index

    # Getting df_fs (forecasted sales)
    # when you do not get any nan_indices on the qty_sold, it means that you didnt get forecasted sales for the future
    # in this case something went wrong with the extraction from hive_metastore at
    # team_group_quality.predicted_sales_volume_enriched is not correctly done.
    if len(nan_indices) == 0:
        nan_index = df_from_hive_with_future.shape[0]
        df_fs = pd.DataFrame([], columns=['period', 'pred_qty_sold'])
    else:
        nan_index = nan_indices[0]

        # Collecting the part of the dataframe that lies in the future and keep only the period and the pred_qty_sold
        if 'qty_sold' in df_from_hive_with_future.columns:
            nan_indices = df_from_hive_with_future[df_from_hive_with_future['qty_sold'].isna()].index
            df_fs = df_from_hive_with_future.iloc[nan_indices][['period', 'pred_qty_sold']]
        else:
            print('Problems ahead: No column name "qty_sold" in df_from_hive_with_future dataframe')

    # Collecting the service call dataframe: df_sc (historic data, from the past)
    df_from_hive = df_from_hive_with_future.iloc[:nan_index].copy()
    int_cols = ['qty_sold'] + [f"m{i}" for i in range(13)]
    df_sc = df_from_hive.loc[:, int_cols].astype(int).copy()

    str_cols = ['period']
    df_sc['period'] = df_from_hive.loc[:, str_cols].astype(str)
    df_sc['pred_qty_sold'] = df_from_hive['pred_qty_sold']

    # Collecting the dataframe with actual sales and forecasted sales: df_as_fs
    df_as_fs = df_from_hive_with_future.dropna()
    df_as_fs = df_as_fs[['period', 'qty_sold', 'pred_qty_sold']]
    df_as_fs[['qty_sold', 'pred_qty_sold']] = df_as_fs[['qty_sold', 'pred_qty_sold']].astype(int)
    df_as_fs['period'] = df_as_fs['period'].astype(str)
    df_as_fs.set_index('period', inplace=True)

    return df_sc, df_fs, df_as_fs


def to_matrix(x):
    """A helper function to reshape a flattend vector to a matrix"""
    if isinstance(x, pd.Series):
        x = x.values
        # TODO: Replace 13 with number of m values
    return x.reshape(13, int(len(x) / 13))


def transform_data_for_scr_df(sca_series, scf_as_array, forecasted_qty_sold, scf_fs_array):
    ''' concatenating the information of the actual service calls and the forecasted service calls in order to be able to create a pandas dataframe later on.

    :param sca_series: the actual service calls for a particular batch (given by the naem of the series), here '201601'
    qty_sold    111920
    m0             821
    m1            1074
    m2             859
    m3             779
    m4             749
    m5             702
    m6             739
    m7             737
    m8             771
    m9             728
    m10            807
    m11            846
    m12            353
    Name: 201601, dtype: int64
    :param scf_array: array of forcasted service calls for this batch
     array([ 764,  914,  727,  651,  696,  693,  694,  811,  860,  868,  951,
       1021,  513])

    OUTPUT is one row with the information concatenated:
    ['201601', 111920, 821, 1074, 859, 779, 749, 702, 739, 737, 771, 728, 807, 846, 353, 764, 914, 727, 651, 696, 693, 694, 811, 860, 868, 951, 1021, 513]
    '''
    return [sca_series.name] + sca_series.values.tolist() + scf_as_array.tolist() + [
        forecasted_qty_sold] + scf_fs_array.tolist()

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_heatmap(df, site, key_out):
    plt.figure(figsize=(12, 10))
    cor = df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    file_out = os.path.join('output', 'site' + str(site) + key_out +'_correlation_all.png')
    plt.savefig(file_out)
    plt.close()

def create_scatter(df, site, key_out):
    pd.plotting.scatter_matrix(df, diagonal='kde')
    file_out = os.path.join('output', 'site' + str(site) + key_out + 'scatter.png')
    plt.savefig(file_out)
    plt.close()


def get_data_X_y(site, target_id):
    ####################################################################
    # site = '261_new'                                                   #
    # target_id = 0       # Give id of the meter (e.g. 0 for Total_TGBT) #

    out_dir = 'output'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    df_evs, df_meters, value_map = read_data(site)

    # removing all columns with _lag in the name
    lag_cols = [i for i, col in enumerate(df_evs.columns) if '_lag' in col]
    df_evs = df_evs.drop(df_evs.columns[lag_cols], axis=1)

    # removing columns with 0 variance
    df_evs = df_evs.loc[:, df_evs.var() != 0]

    # rename columns into a numeric value (column number)
    df_evs, ev_name_mapper = my_preprocessor(df_evs)
    df_meters, meters_name_mapper = my_preprocessor(df_meters)

    data = pd.merge(df_evs, df_meters.iloc[:, target_id], how='inner', left_index=True, right_index=True)
    target = data.iloc[:, -1].name
    data['is_weekend'] = np.where(data.index.dayofweek >= 5, 1, 0)

    # Split the data into training and testing sets
    train_data = data[:'2022-12-31']
    test_data = data['2023-01-01':]

    # Splitting the features and the target variable
    features = [col for col in data.columns if col != target]

    # Create DMatrix objects for training and testing
    X_train, y_train = train_data[features], train_data[target]
    X_test, y_test = test_data[features], test_data[target]
    return X_train, y_train, X_test, y_test, ev_name_mapper, meters_name_mapper

def make_visualisation(data, file_out):
    corr = data.corr()
    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right',
        size=5
    )
    plt.savefig(file_out)


def my_preprocessor(df):
    new_col_list = []
    col_name_mapper = {}
    for i, colname in enumerate(df.columns):
        new_col_list.append(str(i))
        col_name_mapper.update({i: colname})
    df.columns = new_col_list
    return df, col_name_mapper

def read_data(nr):
    file1 = str(nr)+'_EVS.csv'
    file2 = str(nr)+'_meters.csv'

    filename1 = os.path.join('data', file1)
    filename2 = os.path.join('data', file2)
    evs = pd.read_csv(filename1, index_col=0)
    evs.index = evs.index.astype('datetime64[ns]')
    meters = pd.read_csv(filename2, index_col=0)
    meters.index = meters.index.astype('datetime64[ns]')


    value_map = {}
    if 'ConditionText' in evs.columns:
        value_map = {value: num for num, value in enumerate(evs['ConditionText'].unique())}
        evs['ConditionTextNumeric'] = evs['ConditionText'].map(value_map)
        evs = evs.drop(columns='ConditionText')

    # select only the boolean columns
    bool_cols = evs.select_dtypes(include='bool').columns

    # convert boolean values to numeric 0 and 1
    evs_bool = evs[bool_cols].astype(int)

    # concatenate the boolean columns with the other numeric columns
    evs = pd.concat([evs.drop(columns=bool_cols), evs_bool], axis=1)

    evs = remove_NaN(evs)
    meters = remove_NaN(meters)
    return evs, meters, value_map

def remove_NaN(df):
    df = df.fillna(df.mean(axis=0))
    update_df = df.copy()
    for var in df.columns:
        update_df[var] = update_df[var].fillna(update_df[var].mean())
    return update_df


def create_box_figures(df, site, nr_col, key_out):
    print('\nCreating box-plots for site {} {}'.format(site, key_out))
    for i in range(0, len(df.columns), nr_col):
        print(i)
        file_out = os.path.join('output', 'site' + str(site) + key_out + str(i) + '-' + str(i + nr_col - 1) + '.png')
        make_box_plots(df.iloc[:, i:i + nr_col], file_out, i)


def make_box_plots(df, file_out, i):

    red_circle = dict(markerfacecolor='red', marker='o', markeredgecolor='white')
    fig, axs = plt.subplots(1, len(df.columns)+1, figsize=(20, 10))

    for j, ax in enumerate(axs.flat):
        if j < len(df.columns):
            ax.boxplot(df.iloc[:, j], flierprops=red_circle)
            ax.set_title(j+i, fontsize=20, fontweight='bold')
            ax.tick_params(axis='y', labelsize=14)
        else:
            for ii, val in enumerate(np.arange(1, 0, -1/len(df.columns))):
                ax.text(0., val, str(i+ii)+': '+df.columns[ii])
                #ax.get_xaxis().set_visible(False)
                #ax.get_yaxis().set_visible(False)
                ax.axis('off')

    plt.tight_layout()
    plt.savefig(file_out)
    plt.close()


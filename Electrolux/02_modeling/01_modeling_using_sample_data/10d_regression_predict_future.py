# %% [markdown]
# # Prediction into the future

# Idea: The most recent service call data (now is 2023-10) have the following structure:
# example data:
#       period  qty_sold   m0   m1   m2   m3  ...   m7   m8   m9  m10  m11  m12
#       202207    111070  504  646  344  338  ...  155  106  126  162  170   73
#       202208    114315  639  560  428  245  ...  104  127  141  178  181   83
#       202209    120222  518  663  332  239  ...  155  159  176  186  168    0
#       202210    120134  554  483  331  234  ...  177  196  194  191    0    0
#       202211    114564  465  450  286  244  ...  180  168  197    0    0    0
#       202212     86518  385  333  297  191  ...  166  185    0    0    0    0
#       202301     99894  299  415  214  208  ...  190    0    0    0    0    0
#       202302    104952  399  363  301  251  ...    0    0    0    0    0    0
#       202303    102638  280  395  275  267  ...    0    0    0    0    0    0
#       202304     83239  312  386  304  239  ...    0    0    0    0    0    0
#       202305     97315  417  464  326  282  ...    0    0    0    0    0    0
#       202306    100874  436  416  346    0  ...    0    0    0    0    0    0
#       202307     78091  401  463    0    0  ...    0    0    0    0    0    0
#       202308     82119  411    0    0    0  ...    0    0    0    0    0    0
#       202309     87413    0    0    0    0  ...    0    0    0    0    0    0
#
# You see that only the qty_sold for the previous month is available and no service calls are available (m0, m1, m2,...)
# This is normal because the month October is still ongoing and so the total number of service calls that will be made
# in the first month can only be given at the end of October. So to get m0 for a batch sold in 2023-09 one has to wait
# another month i.e. beginning of 2023-10. The get m1 for a batch sold in 2023-09 one has to wait until the beginning of
# 2023-11. And so on.
#
# Step 1: Substituting the Service Calls without actual values (i.e. 0)  by the regression prediction.
# Step 2: Calculate the SCR for the 2023-09, the actual value does exists because it is based on the last non zero
#              diagonal (411 -- (463, 346, 282) -- (..190, 185, 197, 191, 168)
# Step 3: We give the next month to predict i.e. 2023-10 together with the sales forecast for that month, and we give
#             training data from 2022-09 until 2023-09. The training data contains as much as possible actual service
#             service calls, where there are no actuals the predictions are used


# %%
import os
import warnings
import pickle
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from utils.util import adding_scrxQ_info_to_df
from utils.util import create_scr_figures_qtySold_study
from utils.util import extract_scr, get_mape_from_series
from utils.util import generate_exponential_weights, get_mapes, get_hierarchy_dirname, get_vol_diff_from_df, int_to_month
from utils.util import next_month, reorder_df, sc_mi_mat, sc_mi, stripping_future_off
from utils.util import to_matrix, transform_data_for_scr_df

# Filter out the UserWarning coming from the one hot encoding for the test data
warnings.filterwarnings("ignore", category=UserWarning, message="Found unknown categories in columns")
#warnings.filterwarnings("ignore", category=UserWarning, message="FixedFormatter should only be used together with FixedLocator")

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
            scr = extract_scr(df, i, 2, qty_sold_colid=1)
            scrf_as = extract_scr(df, i, 15, qty_sold_colid=1)
            scrf_fs = extract_scr(df, i, 29, qty_sold_colid=28)
            scr_df.append([df.iloc[i, 0], scr, scrf_as, scrf_fs])

    df_out = pd.DataFrame(scr_df, columns=['date', 'scr', 'scrf_as', 'scrf_fs'])

    #############
    merged_df = df.merge(df_out, left_on='date', right_on='date', how='left')
    merged_df.set_index('date', inplace=True)

    return merged_df

def get_scr_df(df: pd.DataFrame, scr_for_list) -> pd.DataFrame:
    ''' Extracts all the possible scr and forcasted scr from a given dataframe df of the form:
        df=
            date qty_sold sc_m1 sc_m2 sc_m3  ... scf_m10 scf_m11 scf_m12 sc_m13 scf_m13
        0   201705   223191  1251  1182   773  ...     401     386     373  202.0   362.0
        1   201706   239887  1394  1255   855  ...     420     404     390  239.0   378.0
        2   201707   228594  1453  1305   850  ...     417     402     388  222.0   376.0
        3   201708   219271  1388  1169   698  ...     382     368     355  183.0   344.0
        4   201709   223009  1379  1151   556  ...     386     373     361  197.0   350.0
        5   201710   221593  1216   996   618  ...     396     384     373  177.0   363.0
        ...
        [26 rows x 28 columns]

        the output will be a dataframe with date (index), scr and scrf (forecasted SCR)
                        scr      scrf
            date
            201805  0.027222  0.030259
            201806  0.029020  0.029366
            201807  0.030635  0.029407
            201808  0.033433  0.030431

    '''

    scr_df = []
    scr_df_columns = ['date']+['scr{}'.format(name[2]) for name in scr_for_list]
    # first 12 months are needed to calculate the SCR based on the history of the sales and service call forcasts
    if len(df) > 12:
        for i in range(12,len(df)):
            to_append = [df.iloc[i,0]]
            for k in range(len(scr_for_list)):
                to_append.append(extract_scr(df, i, df.columns.get_loc(scr_for_list[k])))

            scr_df.append(to_append)

    df_out = pd.DataFrame(scr_df, columns=scr_df_columns)
    df_out.set_index('date', inplace=True)

    return df_out

class DataPreprocesor:
    oh_encoder: OneHotEncoder

    def __init__(self):
        self.oh_encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, drop="first"
        )

    def get_train_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_features = self.get_feature_dataframe(df)
        self.oh_encoder.fit(
            df_features[["month_since_purchase", "month_of_year", "batch_id"]]
        )
        df_features_oh = self.one_hot_encode(df_features)
        return df_features_oh

    def get_test_data(self, df: pd.DataFrame, method={'exp decay': 1.0}, qts_scale_factor=1.0) -> pd.DataFrame:
        ''' Since the batch id of the test data has not been seen by the training data, the one hot
        encoding on this feature will be zeros everywhere. The function allows you to set the features  depending
        on the method chosen. One can choose between the following two methods:
        method = {'oskar': 3}
        the parameter in oskar's method is the number of batches considered (in this example = 3) and given them the
        weight 1/3. When using nr_reg_batches=12 oskars number can only be between 1 and 11
        or
        method = {'exp decay': 1.0}
        here the batch features will get exponential decay weights still summing up to one, the decay rate can be set.
        The lower the decay rate the longer the further away batches still contribute. For 11 batches, i.e. training on
        one full year of data (since one batch is ignored) the weights look like:
        [2.8e-05, 7.8e-05, 2.1e-04, 5.76e-04, 1.57e-03, 4.2e-03, 1.16e-02, 3.15e-02, 8.55e-02, 2.3e-01, 6.3e-01]

        '''

        df_features = self.get_feature_dataframe(df, qts_scale_factor=qts_scale_factor)

        df_features_oh = self.one_hot_encode(df_features)

        batch_id_columns = [column for column in self.oh_encoder.get_feature_names_out() if "batch_id" in column]

        # Batch id of TEST data don't exist in oh-encoder, because it is out of the range of the training data:
        # so to set it to some value there are multiple options:

        if list(method.keys())[0] == 'oskar':
            last_x_months = method['oskar']
            # Option 1: set to 1/3 for last three batch id's in train data
            if last_x_months > len(batch_id_columns):
                print("Weights for the batch id feature will not sum up to one!")
            for column in batch_id_columns[-last_x_months:]:
                df_features_oh[column] = 1 / last_x_months

        elif list(method.keys())[0] == 'exp decay':
            # Option 2: Use all the batches but give an exponentially decaying weight to further away batches, but with a
            # total sum over all the batches equal to 1
            exp_decay = generate_exponential_weights(len(batch_id_columns), decay_rate=method['exp decay'])
            for i, column in enumerate(sorted(batch_id_columns)):
                df_features_oh[column] = exp_decay[i]


        return df_features_oh

    def get_feature_dataframe(self, df: pd.DataFrame, qts_scale_factor = 1.0) -> pd.DataFrame:
        """
        :
        :param df:
        :param qts_scale_factor: Quantity Sold Scale Factor unly used to investigate what the effect is on SCR analysis
        :return:
        """
        n_m = 13

        # Get y data and weights
        qty_sold = (df.loc["qty_sold", :]*qts_scale_factor).astype(int)
        service_calls = df.loc[[f"m{i}" for i in range(n_m)], :]
        qty_remaining = qty_sold - np.cumsum(service_calls, axis=0) + service_calls

        purchase_month_of_year = [(int(x) % 100) - 1 for x in df.columns]
        month_since_purchase = np.array([int(x) for x in range(n_m)])
        service_call_month_of_year = (
            month_since_purchase.reshape(-1, 1) + purchase_month_of_year
        ) % 12

        # Create a dataframe with calendar month for each row (m0) and batch:
        # example: >>> df_month_of_year
                        #date 201601
                        #m0      jan
                        #m1      feb
                        #m2      mar
                        #m3      apr
                        #m4      may
                        #m5      jun
                        #m6      jul
                        #m7      aug
                        #m8      sep
                        #m9      oct
                        #m10     nov
                        #m11     dec
                        #m12     jan
        df_month_of_year = service_calls.copy()
        df_month_of_year.iloc[:, :] = service_call_month_of_year[:, :]
        df_month_of_year = df_month_of_year.applymap(int_to_month)

        # Create a dataframe with month since purchase. All columns are the same (m0 -> m12)
        # example: >>> df_month_since_purchase
        #      201601
        #   0      m0
        #   1      m1
        #   2      m2
        #   3      m3
        #   4      m4
        #   5      m5
        #   6      m6
        #   7      m7
        #   8      m8
        #   9      m9
        #   10    m10
        #   11    m11
        #   12    m12
        df_month_since_purchase = pd.DataFrame(
            data={column: service_calls.index for column in service_calls.columns}
        )


        # Create a dataframe with batch id for each row (m0 -> m13). All rows are the same-
        # Example: >>> df_batch_id
        #               201601
        #           0   201601
        #           1   201601
        #           2   201601
        #           3   201601
        #           4   201601
        #           5   201601
        #           6   201601
        #           7   201601
        #           8   201601
        #           9   201601
        #           10  201601
        #           11  201601
        #           12  201601
        df_batch_id = pd.DataFrame({column: [str(column)] * len(service_calls.index) for column in service_calls.columns})

        df_features = pd.DataFrame(
            {   "month_since_purchase": df_month_since_purchase.values.flatten(),
                "month_of_year": df_month_of_year.values.flatten(),
                "batch_id": df_batch_id.values.flatten(),
                "weight": qty_remaining.values.flatten(),
                "y": (service_calls / qty_remaining).values.flatten(),
            }
        )
        # An Example on how the df_features looks like (for 12 months traning period):
        #           month_since_purchase month_of_year batch_id  weight         y
        #       0                     m0           jan   201501  314597  0.004851
        #       1                     m0           feb   201502  324739  0.005281
        #       2                     m0           mar   201503  354097  0.004456
        #       3                     m0           apr   201504  339620  0.004096
        #       4                     m0           may   201505  323189  0.004545
        #       5                     m0           jun   201506  346602  0.004836
        #       6                     m0           jul   201507  328783  0.004401
        #       7                     m0           aug   201508  311892  0.004918
        #       8                     m0           sep   201509  368664  0.004991
        #       9                     m0           oct   201510  424491  0.004589
        #       10                    m0           nov   201511  410386  0.005388
        #       11                    m0           dec   201512  315425  0.005212
        #       12                    m1           feb   201501  313071  0.005254
        #       13                    m1           mar   201502  323024  0.004644
        #               .....
        #       23                    m1           jan   201512  313781  0.006135
        #       24                    m2           mar   201501  311426  0.003137
        #       25                    m2           apr   201502  321524  0.002799
        #               .....
        #       150                  m12           jul   201507  319302  0.000783
        #       151                  m12           aug   201508  301878  0.000997
        #       152                  m12           sep   201509  357196  0.000901
        #       153                  m12           oct   201510  412183  0.000990
        #       154                  m12           nov   201511  397358  0.001145
        #       155                  m12           dec   201512  304134  0.001075
        return df_features

    def one_hot_encode(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Add one hot encoded columns of dataframe

        Parameters
        ----------
        df_features : pd.DataFrame
            Dataframe with features

        Returns
        -------
        pd.DataFrame
            Dataframe with one hot encoded features
        """
        oh_data = self.oh_encoder.transform(
            df_features[["month_since_purchase", "month_of_year", "batch_id"]]
        )
        df_x = pd.DataFrame(oh_data)
        # Set index before concatenating
        df_x.index = df_features.index
        # Add names to columns
        df_x.columns = self.oh_encoder.get_feature_names_out()

        # Drop non one-hot-encoded columns
        df_features_oh = pd.concat([df_features, df_x], axis=1).drop(
            columns=[
                "batch_id",
                "month_since_purchase",
                "month_of_year",
            ]
        )
        return df_features_oh

class LogitRegression(LinearRegression):
    """Logistic regression with sklearn API

    Wrap linear regression after transforming target variable to logit

    https://stackoverflow.com/questions/44234682/how-to-use-sklearn-when-target-variable-is-a-proportion
    """

    def fit(self, x, p, sample_weight=None):
        """Fit model with features x to a target probability p"""
        p = np.asarray(p.astype(float))
        y = np.log(p / (1 - p))
        return super().fit(x, y, sample_weight)

    def predict(self, x):
        """Predict probability of p(y=1)"""
        y = super().predict(x)
        return 1 / (np.exp(-y) + 1)

def create_scr_with_sales_uncertainty(scr_dict, **kwargs):
    ''' This code shows the confidence level around the forecasted SCR. The
     confidence level comes from the +-x percent scaling of the forcasted sales volume. In order to be able to use this
     code a dictionary is expected as input that contains for each analysis (for e different scale factor applied on
     the sales volume e.g. qty_sold_scfs = [0.9, 1.0, 1.1]

    :param scr_dict: with keys that look like: dict_keys(['down', 'center', 'up'])
                     the value per key represents a dataframe with the following columns ("scu" stands for service
                     calls updated:
                     Index(['qty_sold', 'scu_m0', 'scu_m1', 'scu_m2', 'scu_m3', 'scu_m4', 'scu_m5',
                    'scu_m6', 'scu_m7', 'scu_m8', 'scu_m9', 'scu_m10', 'scu_m11', 'scu_m12'], dtype='object')
    :param kwargs: 'method': just used for creating the title of the figure and the filename of the saved figure
                   'hierarchy': used for creating the title of the figure and the filename of the saved figure but also
                                to set the ylim range for the SCR-plot (each prod line has a different scr range)
                   'qty_sold_scf': used for the title as well and to unpack the dataframes for the 3 dict-keys
                   'domain': is a flag that can be 1,2 or 3 and reflects on the 3 different prediction domains
                             when domain = 1 you do have actual sales and actual service calls (validation is possible)
                                         = 2 you do not have actual service calls but you do have the actual sales volume
                                             this actual sales volume is used to predict the service calls
                                         = 3 you do not have neiter the actual sales volume nor the actual service calls
                            only when domain=1 you have the information to show the volume differences on the second plot
    :return: saved figure
    '''

    # depending on the following flag a plot is created with just one or two subplots
    # the single subplot will only show the SCR forecast with the uncertainty band due to the Salse Forecast
    # the version with the 2 subplots will also show the volume difference, notice here that the volume difference is
    # taken as the difference between the actual SCR and the respective lower and upper values of the uncertainty band

    method = kwargs.pop('method','Regression - MarkovChain')
    hierarchy = kwargs.pop('hierarchy', 'line')
    qty_sold_scfs = kwargs.pop('qty_sold_scf',[1])
    remove_vol_plot = kwargs.pop('remove_vol_plot', True)

    percentage_error_val = round((qty_sold_scfs[1]-qty_sold_scfs[0])*100)

    if len(qty_sold_scfs) != 3:
        print('You need to give a array with length 3 to create the uncertainty band on the figure.')
        return

    xQ = kwargs.pop('xQ',0)
    title_addition = '- Sales Uncertainty ={}%'.format(percentage_error_val)

    xq_map = {0: ['scr','scrf_as', 'scrf_fs','0'],
              3: ['scr_rol_win03', 'scrf_as_rol_win03','scrf_fs_rol_win03', 'LQ'],
              12: ['scr_rol_win12', 'scrf_as_rol_win12', 'scrf_fs_rol_win12', 'LY']}
    vars_to_use = xq_map[xQ]

    scr_df = scr_dict['center']

    scr_df.index = pd.to_datetime(scr_df.index, format='%Y%m')
    scr_df.index = scr_df.index.strftime('%Y%m')
    print('    --> Create_scr_figures_qtySold_study Q{} --'.format(xq_map[xQ][-1]))
    if remove_vol_plot:
        fig, ax1 = plt.subplots( figsize=(15, 7))
        fig_name = os.path.join('plots','SCR','Regression',get_hierarchy_dirname(hierarchy), 'SCR_{}_{}_Q{}_scf_v1_{}.png'.format(
            method, hierarchy, xQ, percentage_error_val))
    else:
        fig, ax = plt.subplots(3,1, figsize=(19, 14), sharex=True)
        ax1 = ax[0]
        vol_diff_min_as = get_vol_diff_from_df(scr_df[vars_to_use[0]], scr_df[vars_to_use[1]])
        vol_diff_max_as = get_vol_diff_from_df(scr_df[vars_to_use[0]], scr_df[vars_to_use[1]])
        #mask = ~scr_df['vol_diff'].isna().values
        pos_vol_diff_as =  [val if val > 0 else 0 for val in vol_diff_min_as]
        neg_vol_diff_as =  [val if val < 0 else 0 for val in vol_diff_max_as]

        vol_diff_min_fs = get_vol_diff_from_df(scr_df[vars_to_use[0]], scr_df[vars_to_use[2]])
        vol_diff_max_fs = get_vol_diff_from_df(scr_df[vars_to_use[0]], scr_df[vars_to_use[2]])
        #mask = ~scr_df['vol_diff'].isna().values
        pos_vol_diff_fs =  [val if val > 0 else 0 for val in vol_diff_min_fs]
        neg_vol_diff_fs =  [val if val < 0 else 0 for val in vol_diff_max_fs]


    fig_name = os.path.join('plots','SCR','Regression',get_hierarchy_dirname(hierarchy),'SCR_{}_{}_Q{}_scf_v2_{}.png'.format(
            method, hierarchy, xQ, percentage_error_val))
    # Set up common elements for both cases
    ax1.plot(scr_df.index, scr_df[vars_to_use[0]], marker='o', color='k', label='Actual')
    ax1.plot(scr_df.index, scr_df[vars_to_use[2]], marker='o', color='g', label='SCRF FS')
    ax1.plot(scr_df.index, scr_df[vars_to_use[1]], marker='o', color='b', label='SCRF AS')
    # Fill the area between scr_min_df['scrf'] and scr_max_df['scrf'] with blue color and alpha value
    #ax1.fill_between(scr_df.index, scr_min_df[vars_to_use[1]], scr_max_df[vars_to_use[1]], color='blue', alpha=0.3)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Serv Call Ratio')
    ax1.grid()
    ax1.legend()
    ax1.set_xlim([scr_df['scrf'].first_valid_index(), scr_df['scrf'].last_valid_index()])

    if remove_vol_plot:
        ax1.set_title('{}: {} - Q{}\n {}'.format(
                hierarchy, method, xQ, title_addition)
            )
        ax1.set_xticklabels(scr_df.index, rotation=90)
    else:
        get_vol_diff_from_df(scr_df[vars_to_use[0]], scr_df[vars_to_use[1]])
        mape_as, _, mape_as_std, _ = get_mape_from_series(get_vol_diff_from_df(scr_df[vars_to_use[0]], scr_df[vars_to_use[1]]))
        mape_fs, _, mape_fs_std, _ = get_mape_from_series(get_vol_diff_from_df(scr_df[vars_to_use[0]], scr_df[vars_to_use[2]]))
        qty_sold_mean = scr_df['qty_sold'].mean()
        ax1.set_title('{}: {} - Q{}\n <q_sold>={:.0f} - MAPE AS={:.1f}+-{:.1f} - MAPE FS={:.1f}+-{:.1f}'.format(
                hierarchy, method, xQ, qty_sold_mean, mape_as, mape_as_std, mape_fs, mape_fs_std)
            )

        ax[1].bar(scr_df.index, pos_vol_diff_as,  color='green')
        ax[1].bar(scr_df.index, neg_vol_diff_as,  color='red')
        ax[1].set_ylabel('PE AS')
        ax[2].bar(scr_df.index, pos_vol_diff_fs,  color='green')
        ax[2].bar(scr_df.index, neg_vol_diff_fs,  color='red')
        ax[2].set_ylabel('PE AS')
        ax[1].set_xticklabels(scr_df.index, rotation=90)
        ax[1].grid()

    plt.tight_layout()
    plt.savefig(fig_name)

def predict_sc(df_train, df_test):
    # Training the model on the service call data for 12 batches
    log_reg = LogitRegression()
    log_reg.fit(
        df_train.drop(columns=["y", "weight"]),
        df_train["y"],
        sample_weight=df_train["weight"] / df_train["weight"].mean(),
    )

    # Prediction of the following month
    # calculate probabilities according to model
    p_pred = log_reg.predict(df_test.drop(columns=["weight", "y"]))
    p_pred = to_matrix(p_pred)

    # Weight gives quantity remaining after time step. Want to set to quantity
    # remaining at time 0
    qty_rem = to_matrix(df_test["weight"])
    for col in range(qty_rem.shape[1]):
        qty_rem[:, col] = qty_rem[0, col]

    # Total number of calls up to a time
    sum_p_calls_at_time = np.cumsum(p_pred, axis=0)
    # Number of products without a call at time t
    sum_remaining = np.multiply(qty_rem, (1 - sum_p_calls_at_time))

    # Roels method to reconstruct to SCs for the predictions, based on recursive function:
    pred_calls = sc_mi_mat(sum_remaining, p_pred)

    return pred_calls

def predict_future(df_region1_2, df_as_fs, df_sales_forecast):

    # This rate is used to set the exponential decay rate for the one hot encoding train_batches in the case of prediction
    decay_rate = 1
    test_oh_choice = {'exp decay': decay_rate}

    # Filling the actual service calls that are zero with its prediction values (using the actual sales)
    nr_reg_batches = 12
    nr_tail_batches = 2*nr_reg_batches+1
    next_pred_id = np.array([1,2,3,4,5,6,7,8,9,10,11,12])

    df_region1_2.reset_index(inplace=True)
    df = reorder_df(df_region1_2[-nr_tail_batches:])
    dp = DataPreprocesor()

    # prediction using actual sales
    train_data = df.iloc[0:14, 0:nr_reg_batches]
    test_data = df.iloc[0:14, nr_reg_batches+next_pred_id]
    df_train = dp.get_train_data(train_data)
    df_test = dp.get_test_data(test_data, method=test_oh_choice)
    pred_calls_as = predict_sc(df_train, df_test)

    # prediction using forecasted sales
    train_data = df.iloc[0:14, 0:nr_reg_batches]
    test_data = df.iloc[0:14, nr_reg_batches+next_pred_id]
    train_data.loc['qty_sold'] = df_region1_2[df_region1_2['date'].isin(test_data.columns)]['pred_qty_sold'].values
    df_train = dp.get_train_data(train_data)
    df_test = dp.get_test_data(test_data, method=test_oh_choice)
    pred_calls_fs = predict_sc(df_train, df_test)

    # Actual number of Service Calls at time m0,m1,m2,..,m12
    serv_calls = sc_mi_mat(to_matrix(df_test["weight"]),to_matrix(df_test["y"]))

    sc_list = []
    # Getting to the SCR stuff, Keeping the Actuals as they were (containing lower diagonal zeros,
    df_region1_2.set_index('date', inplace=True)
    for i, date in enumerate(test_data.columns):
        sc_list.append([date]+df_region1_2.loc[date][:14].values.tolist() +
                       pred_calls_as[:,i].tolist() + [int(df_region1_2.loc[date]['pred_qty_sold'])] +
                       pred_calls_fs[:,i].tolist())

    columns=['date', 'qty_sold'] + [f"sca_m{i}" for i in range(13)] + [f"scf_as_m{i}" for i in range(13)] + \
               ['pred_qty_sold'] + [f"scf_fs_m{i}" for i in range(13)]
    df_sc = pd.DataFrame(sc_list, columns=columns)


    # Step 3 Real predictions (SalesForecast list is required, and considered to be the sales forecast for the
    # respective future months that are not in the input dataframe
    for i,row in df_sales_forecast.iterrows():
        print(i)
        next_bacth_period = row['period']
        print('Full prediction for: {}'.format(next_bacth_period))
        # Training data (taking the last 13 rows of this growing dataframe)
        train_start_date = df_sc[-nr_reg_batches:]['date'].iloc[0]
        train_end_date = df_sc[-nr_reg_batches:]['date'].iloc[-1]
        train_data = df_sc[-nr_reg_batches:][['date', 'qty_sold']+[f"scf_fs_m{i}" for i in range(13)]].T.rename(index=lambda x: x.replace('scf_fs_', ''), inplace=False)
        print('Training data: {} - {}'.format(train_start_date, train_end_date))

        # Train data
        dp = DataPreprocesor()
        df_train = dp.get_train_data(train_data)

        # Test data
        my_values = [int(row['pred_qty_sold'])] + [0] * 13
        pred_data = pd.DataFrame(my_values, index=['qty_sold'] + [f"m{i}" for i in range(13)],
                                 columns=[next_bacth_period])
        df_test = dp.get_test_data(pred_data, method=test_oh_choice)
        pred_calls_fs = predict_sc(df_train, df_test)

        my_row_2add = [next_bacth_period] + [int(row['pred_qty_sold'])] + [int(row['pred_qty_sold'])] + list(pred_calls_fs.flatten())
        my_df_2add = pd.DataFrame([my_row_2add], columns=['date'] + ['qty_sold'] + ['pred_qty_sold'] + [f"scf_fs_m{i}" for i in range(13)] )
        df_sc = pd.concat([df_sc, my_df_2add], ignore_index=True)

    return df_sc

if __name__ == '__main__':

    # if the flag "combined_with_evaluation_results" is on, the results from the evaluation period (i.e. where we have the actual service calls)
    # will be added to the prediction period (where we do have either partial actuals for some of the waraanty months or no actuals at all)
    # and you will get the combined results on the 3 different time regions:
    # 1) period for which the actual SCRs and the forecasted SCRs exist (evaluation period, results are processed with
    #    script 10b_regression_MarkovChain), qty_sold and pred_qty_sold exist
    # 2) period for which the actual SCRs do not exist but the Actual sales do exist partially (for some of the waranty months m0, m1, ..m12). This is the
    #    Intermediate region. In this region we use the existing Actual sales to predict the rest of the SCs (m12, m11, m10, ..
    #    qty_sold and pred_qty_sold exist
    # 3) period for which neither the SCRs nor the Actual sales exist (True prediction region)
    #    Only pred_qty_sold exist, qty_sold does not exist since we are in future months
    #-------------------------------------------------------------------------------------------------------------------
    # This can be visualised as follows (status is given for a today date somewhere in Nov 2023), this means that the
    # qty_sold for 202310 is known but none of the sc_m0, sc_m1, sc_m12 are known:
    #
    #               m0 m1 m2    ...       m11 m12    qty_sold   pred_qty_sold
    #            ----------------------------------
    #  201501    -          Region 1              -   int           nan
    #  201502    -                                -   int           nan
    #     .      -                                -
    #     .      -            Actual              -
    #     .      -              SCs               -   int           int
    #  202207    -                                -   int           int
    #  202208    -                                -   int           int
    #  202209    -                                -   int           int
    #            ----------------------------------
    #  202210    -           Region 2          -  -   int           int
    #  202211    -                           -    -   int           int
    #  202212    -                         -      -   int           int
    #  202301    -      Actual           -        -   int           int
    #     .      -        SCs          -          -   int           int
    #     .      -                   -            -   int           int
    #     .      -                 -              -   int           int
    #  202305    -               -    Predicting  -   int           int
    #  202306    -            -       SCs         -   int           int
    #  202307    -        -                       -   int           int
    #  202308    -     -     Actual SCs           -   int           int
    #  202309    -  -          are zero here      -   int           int
    #  202310    -                                -   int           int
    #            ----------------------------------
    #            -       Region 3                 -
    #  202310    -       Actual SCs are nan here  -   nan           int
    #  202311    -                                -   nan           int
    #  202312    -           Full                 -   nan           int
    #  202401    -        Prediction              -   nan           int
    #  202402    -          Region                -   nan           int
    #  202403    -                                -   nan           int
    #            ----------------------------------


    # just running this hierarchy, instead of the full list
    hierarchy = 'FoodPreparation_all_all'

    # dataframe is read where we have multiple rows with the following columns:
    #   period (being the batch e.g. 201501),
    #   qty_sold (being float or int)
    #   m0, m1, ..., m12 (the service calls in the respective warranty period and
    #   pred_qty_sold which reflects on the forecasted sold volume.
    #
    # -the first number of rows will have nan for the pred_qty_sold since it is only generated after 201901
    # -from a given row number the number of service calls will become zero because those service calls are part
    #  of service calls which will be recored in the future.
    # -from a certain row number also the qty_sold will be nan because it will be reveiled only in the future

    # read the hive table (here we saved them as a csv file and read those)
    df_from_hive_with_future = pd.read_csv(os.path.join("data","full_sc_fs_tables","sc_fs_{}.csv".format(hierarchy)))
    df_from_hive_with_future['period'] = df_from_hive_with_future['period'].astype(str)

    # stripping the Region 1 and 2 from region 3. df_sc relates to region 1 and 2 and df_sales_forecast relates to region 3
    df_sc_from_hive, df_sales_forecast, df_as_fs = stripping_future_off(df_from_hive_with_future)

    # if you need the pred_qty_sold instead of qty_sold you need to adapt the code here
    scs_cols = ['period','qty_sold'] + [f"m{i}" for i in range(13)]
    df1 = df_sc_from_hive[scs_cols].copy()                              # df1 ranges over region 1 and 2 (but not 3)
                                                                        # so it starts at batch 201501 and ends at
                                                                        # batch 202310

    # When you want to show the 3 regions on the SCR graph
    with open(os.path.join('pickles','SCR_eval_period_{}.pkl'.format(hierarchy)), 'rb') as f:
        df2 = pickle.load(f)
    key_mapping = {'QtySoldScf1.0': 'center'}

    # Just renaming the keys of the dictionary so that the are named down, center, up instead of QtySoldScf0.9, -1.0
    # and QtySoldScf1.1 respectively (Use a dictionary comprehension to create a new dictionary with updated keys)
    dict_df_with_actuals = {key_mapping.get(k, k): v for k, v in df2.items()}
    df_region1_2 = dict_df_with_actuals['center']
    map_dict = {}
    for i in range(13):
        map_dict.update({f'sca_m{i}': f'm{i}'})
    df_region1_2 = df_region1_2.rename(columns=map_dict)

    df_sc = predict_future(df_region1_2, df_as_fs, df_sales_forecast)
    df_sc.set_index('date', inplace=True)




    # Merging the region 1 and 2 to region 3
    map_dict = {}
    for i in range(13):
        map_dict.update({f'm{i}': f'sca_m{i}'})
    df_region1_2 = df_region1_2.rename(columns=map_dict)



    df1_2 = df_region1_2[df_sc.columns]
    common_indices = df1_2.index.intersection(df_sc.index)
    df1_2.loc[common_indices] =df_sc.loc[common_indices]

    indices_not_in_index2 = df_sc.index.difference(df1_2.index)
    df1_2_3 = pd.concat([df1_2,df_sc.loc[indices_not_in_index2]], axis = 0)
    df1_2_3.reset_index(inplace=True)
    # TODO Need to update the adding_scr_info_to_df function for this case
    df1_2_3 = adding_scrxQ_info_to_df(adding_scr_info_to_df(df1_2_3))
    dict_df1_2_3={'QtySoldScf1.0': df1_2_3}
    create_scr_figures_qtySold_study(dict_df1_2_3, method='Future Pred (M. Chain)',
                                     hierarchy=hierarchy,
                                     remove_volume_plot=True,
                                     xQ=0)



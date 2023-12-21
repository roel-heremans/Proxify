##%% [markdown]
### Binary Classification Example
##Idea: Use a markov chain to model the probability of a service call given the
##number of months since the product was purchased, the month of the year and
##the date of purchase, conditional on no previous service call beeing made \
##\
##Why not use a hazard model instead? One reason why a hazard model may not be
##the best choice is that the basic hazard models miss important features of the
##data, such as seasonality and the tendancy to make calls just before the
##warranty expires. Also, we are not actually modelling the time to failure, but
##the time decision to make a service call when the customer thinks a failure
##has occured. A hazard model may be the most appropriate to model the actual
##time ot failure, but customers may wait for a couple of months after the
##failure to report the failure, or not report during summer, which could be why
##calls are made just before the warranty expires. \
##\
##In the markov chain, the probability to make a service call could be modeled
##by a number of different binary classification models. A good start would be
##to use the logistic regression model. \
##\
##We define the probability for a service call at t months after a purchase made
##at time T, during month M, conditional on now service call before time t, as
##$P(t| T, M)$. The parameters of the model are defined as :
##- $x_t$ : parameter for time since purchase [13 x 1]
##- $y_T$ : parameter for batch id [1 x number_of_batches]
##- $z_M$ : parameter for month of year [12 x 1] \
##And so the full parameter vector $\theta = [x_t, y_T, z_M]$. \
## The utility of making a service call at time t, given the time of purchase
##$T$ is defined as: \
##$U(t,T,M) = x_t + y_T + z_M$ and the probability is given by: \
##$P(t|T,M) = 1/(1+exp(-U(t,T,M)))$
##x = [1,2,3]
##y = [0.1,
##     0.2,]
##z = [0.01, 0.02, 0.03,...]
##Purchase during 202001 (t=0)
##P(m0 | T=202001) = 1/(1+exp(-(x[0] + y[0] + z[0])))
##P(m1 | T=202001) = 1/(1+exp(-(x[1] + y[0] + z[1])))
##P(m0 | T=202002) = 1/(1+exp(-(x[0] + y[1] + z[1])))
##P(m1 | T=202002) = 1/(1+exp(-(x[1] + y[1] + z[2])))
#### Calculate service calls per time period
##Let qty(T, t) be the number of purchases made at time T that have not had any
##service call up to time t. The number service calls made by at time t for
##purchases made at time T:
##$Q(t=0 |T ) = qty(T, t=0) \cdot P(t=0 | T )$
##\
##$qty(T, t=1) = qty(T, t=0) - Q(t=0 |T)$
##\
##And for any time $t$:
##\
##$Q(t |T) = qty(T, t-1) \cdot P(t | T ) $
##%%
import os
import sys
import warnings
import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

from utils.util import adding_scr_info_to_df, adding_scrxQ_info_to_df, check_on_zeros, create_necessary_paths
from utils.util import generate_exponential_weights, get_all_csv_files, get_mapes, get_mape_from_series, \
    get_vol_diff_from_df
from utils.util import int_to_month, reorder_df
from utils.util import sc_mi, sc_mi_mat, stripping_future_off, to_matrix, transform_data_for_scr_df
from utils.util import create_scr_figures_qtySold_study, create_scr_sunburst_figure_hierarchy

# Filter out the UserWarning coming from the one hot encoding for the test data
warnings.filterwarnings("ignore", category=UserWarning, message="Found unknown categories in columns")
warnings.filterwarnings("ignore", category=UserWarning,
                        message="FixedFormatter should only be used together with FixedLocator")


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

    def get_test_data(self, df: pd.DataFrame, method={'exp decay': 1.0}, qts_scale_factor=1) -> pd.DataFrame:
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

    def get_feature_dataframe(self, df: pd.DataFrame, qts_scale_factor=1) -> pd.DataFrame:
        """
        :
        :param df:
        :param qts_scale_factor: Quantity Sold Scale Factor unly used to investigate what the effect is on SCR analysis
        :return:
        """
        n_m = 13

        # Get y data and weights
        qty_sold = (df.loc["qty_sold", :] * qts_scale_factor).astype(int)
        service_calls = df.loc[[f"m{i}" for i in range(n_m)], :]
        qty_remaining = qty_sold - np.cumsum(service_calls, axis=0) + service_calls

        # in very rare cases the qty_remaining behaved strange. Meaning there was a zero and even negative SCs
        def replace_zero_negative(df):
            return df.mask(df <= 0, 1)

        # Assuming qty_remaining is your DataFrame
        # Replace zero and negative values with 1 using the function
        qty_remaining = replace_zero_negative(qty_remaining)

        purchase_month_of_year = [(int(x) % 100) - 1 for x in df.columns]
        month_since_purchase = np.array([int(x) for x in range(n_m)])
        service_call_month_of_year = (
                                             month_since_purchase.reshape(-1, 1) + purchase_month_of_year
                                     ) % 12

        # Create a dataframe with calendar month for each row (m0) and batch:
        # example: >>> df_month_of_year
        # date 201601
        # m0      jan
        # m1      feb
        # m2      mar
        # m3      apr
        # m4      may
        # m5      jun
        # m6      jul
        # m7      aug
        # m8      sep
        # m9      oct
        # m10     nov
        # m11     dec
        # m12     jan
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
        df_batch_id = pd.DataFrame(
            {column: [str(column)] * len(service_calls.index) for column in service_calls.columns})

        df_features = pd.DataFrame(
            {"month_since_purchase": df_month_since_purchase.values.flatten(),
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


def create_necessary_paths():
    to_create_paths = ['pickles', 'plots/CtrlPlots', 'plots/mape_summary', 'plots/SalesForecast', 'plots/SCR',
                       'plots/SCR/Regression/line', 'plots/SCR/Regression/group',
                       'plots/SCR/Regression/subgroup', 'plotly', 'tables']
    for my_path in to_create_paths:
        if not os.path.exists(os.path.join(os.getcwd(), my_path)):
            # Create the directory
            os.makedirs(os.path.join(os.getcwd(), my_path))


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


if __name__ == '__main__':

    # checking and creating paths if necessary
    create_necessary_paths()

    # if pickles exist you can use them instead of having to run the analysis again and again
    use_existing_pickles = False

    # These are the number of batches used in the regression model
    nr_reg_batches = 12

    # Nr of months in warranty period (m0,m1,m2,m3,m4,...m12)
    nr_mi = 13

    # if you want to see what the effect is of predicting not the next month (i.e. 0) but for instance 3 months later
    # than the next month (use 3). You can also run for instance [0,3,6,12] at once. You would do this to study the effect
    # on not predicting the next month but further months in the future. The create_regression_mape_plot function will
    # allow you to visualize the effect.
    next_pred_id = np.array([0])

    # This rate is used to set the exponential decay rate for the one hot encoding train_batches in the case of prediction
    decay_rate = 1
    test_oh_choice = {'exp decay': decay_rate}

    # This variable is used to get an estimate on the effect on the SCR MAPE when the sales_volume is predicted wrongly
    # by 10%, use volume_scf = 1.1. If you want to use the actual sales_volume use volume_scf=1
    volume_scfs = [1.0]

    hierarchy_files = sorted(
        get_all_csv_files(os.path.join('data', 'full_sc_fs_tables'), start_with='sc_fs_', end_with='.csv'))

    # just running this one, instead of the full list
    # hierarchy_files = ['foodpreparation_all_all', 'dishcare_all_all', 'fabriccare_all_all', 'foodpreservation_all_all']

    # this is the variable that collects the mapes of each product hierarchy so that the sunburst chart can be created
    sunburst_data = []
    sunburst_data_LQ = []
    sunburst_data_LY = []

    # looping over all the product hierarchies
    # hierarchy_files = ['FoodPreparation_Hood_all']

    #for hierarchy_i, hierarchy in enumerate(hierarchy_files):
    for hierarchy_i, hierarchy in enumerate(['FoodPreparation_all_all']):
        stop_flag = False
        print("{}/{}: {}".format(hierarchy_i + 1, len(hierarchy_files), hierarchy))

        # when running 3 times for volume_scfs = [0.9, 1.0, 1.1], this dict collects the results for each of the 3 runs,
        # and the info will be used to add the blue uncertainty band to the scr plots
        df_scr_qtySold_study = {}

        # looping over the scale factors for the sales volume [0.9, 1.0, 1.1]
        for scf_count, volume_scf in enumerate(volume_scfs):
            print("--> {}/{}: {}".format(scf_count + 1, len(volume_scfs), volume_scf))

            df_sc_fs = pd.read_csv(os.path.join("data", "full_sc_fs_tables", "sc_fs_{}.csv".format(hierarchy)))
            df_sc, df_sales_forecast, df_as_fs = stripping_future_off(df_sc_fs)
            df_sc_replaced_zeros, nr_interventions = check_on_zeros(df_sc)

            # nr_interventions are the count of zeros that were changed into ones.
            # although we want to see what the amount of interventions is, not taking into account the the lower triangle (containing zeros because those service calls
            # are from the future). We have a lower triangle of zeros, due to the fact that those service calls are not yet know since they will only be know in future
            # months. For instance when the current date is 15th of November 2023 (i.e. batch 202311), then the service calls for the batch 202310 will contain only
            # sc_m0 different from 0. sc_m1, sc_m2, ... sc_m12 will only be filled in the respective months 202312, 202401, ...202412, Hence the lower triangle with
            # zeros. Removing the amount of all those zeros is equal to (nr_mi+1)*(nr_mi)/2 = 91, for nr_mi = 13
            nr_interventions = int(nr_interventions - (nr_mi + 1) * (nr_mi) / 2)

            # if you need the pred_qty_sold instead of qty_sold you need to adapt the code here
            scs_cols = ['period', 'qty_sold'] + [f"m{i}" for i in range(13)]
            df_sc = df_sc_replaced_zeros[scs_cols].copy()

            # Probably not enough service calls for this product to deliver a worthy model/prediction
            # An additional study needs to happen to find out what the effect on the average mape is due to low data samples, or as a function of the
            # amount of zeros which were flipped into 1. For now we decide not to trust the model anymore when we flipped more than 10 zeros to ones.
            if nr_interventions > 100:
                print(
                    f'Be carefull with this dataset,\n{hierarchy} there seem to be more than {nr_interventions} service call values to be zero.')
                stop_flag = True
                break

            if df_sc.shape[0] < nr_reg_batches * 2:
                print(f'--> Aborting: Not enough batches to get the first value of SCR.')
                stop_flag = True
                break

            has_negative_values = (df_sc[[f"m{i}" for i in range(13)]] < 0).any().any()
            if has_negative_values:
                print(f'--> Aborting: There are Service Calls with negative values')
                stop_flag = True
                break

            df = reorder_df(df_sc)

            dp = DataPreprocesor()

            collist = df.columns[nr_reg_batches:]
            rowlist = ["next{}".format(i) for i in next_pred_id]
            res_df = pd.DataFrame([[None] * len(collist)] * len(rowlist), columns=collist, index=rowlist)

            df_for_scr = []  # this will become the dataframe to get the scr for this product hierarchy

            # This is the loop for the rolling window (Training data are based on 12 batches/months, prediction is the following batch/month)
            for i in range(df.shape[1] - nr_reg_batches - max(next_pred_id)):
                df_train = dp.get_train_data(df.iloc[:, i:i + nr_reg_batches])

                # We need to take once the pred_qty_sold from df_sales_forecast (when available, only available from 2019 onwards)
                # and once the actual sales and perform the service call predictions for both inputs
                test_data = df.iloc[:, i + nr_reg_batches + next_pred_id].copy()
                df_test_with_actual_sales = dp.get_test_data(test_data, method=test_oh_choice,
                                                             qts_scale_factor=volume_scf)
                pred_calls_as = predict_sc(df_train,
                                           df_test_with_actual_sales)  # predicted_calls_as (based on as=Actual Sales)
                pred_columns = df.columns[i + nr_reg_batches + next_pred_id].values

                # Check if forecasted sales exist for this batch if so overwriting the actual qty_sold with the
                # forecasted qty_sold and re-predicting but now with the forecasted sales
                are_all_periods_available = all(period in df_as_fs.index for period in pred_columns)
                if are_all_periods_available:
                    test_data.loc['qty_sold'] = df_as_fs.loc[pred_columns]['pred_qty_sold'].values
                    df_test_with_predicted_sales = dp.get_test_data(test_data, method=test_oh_choice,
                                                                    qts_scale_factor=volume_scf)
                    pred_calls_fs = predict_sc(df_train,
                                               df_test_with_predicted_sales)  # predicted_calls_fs (based on fs=Forecasted Sales)
                else:
                    # Before 201901 no forecasted sales exists, in this case we say that pred_calls_fs is the same as
                    # the predicted service calls based on the actual sales
                    pred_calls_fs = pred_calls_as

                # Roels method to reconstruct to SCs for the Actual number of Service Calls at time m0,m1,m2,..,m12
                serv_calls = sc_mi_mat(to_matrix(df_test_with_actual_sales["weight"]),
                                       to_matrix(df_test_with_actual_sales["y"]))

                mapes = get_mapes(serv_calls, pred_calls_as)
                df_pred = pd.DataFrame(columns=['Actual', 'Forecast_AS', 'Forecast_FS'],
                                       index='m' + pd.RangeIndex(start=0, stop=13, step=1, name='Month').astype(str))
                # the second index is the one that reflects on the number of batches asked to be predicted (defined in "next_pred_id" see at the beginning of the main
                # script), here set to zero because we keep the prediction of the next month wrt training
                df_pred['Actual'] = serv_calls[:, 0]
                df_pred['Forecast_AS'] = pred_calls_as[:, 0]
                df_pred['Forecast_FS'] = pred_calls_fs[:, 0]
                # create for each batch that got predicted, a control plot on the service call prediction quality, only when there is no scale applied on the
                # sales volume
                # if volume_scf == 1.0:
                #    title_txt = "Reg Fit Result: {}\nLearningSize={}, Predicting: {}".format(
                #        hierarchy,nr_reg_batches,pred_columns[0])
                #    create_regression_control_plot(df_pred, title=title_txt, hierarchy=hierarchy)

                # Collecting to the data batch date, qty_sold, actual SCs, forecasted SCs based on the actual sales, forecasted sales and SCs based on the
                # forecasted sales. Those data will serve to create a dataframe. This dataframe will be the input to add the SCR and SCR_forecasted
                df_for_scr.append(transform_data_for_scr_df(df.iloc[:, i + nr_reg_batches], pred_calls_as[:, 0],
                                                            test_data.loc['qty_sold'][0], pred_calls_fs[:, 0]))

                for i, col in enumerate(pred_columns):
                    res_df.at[rowlist[i], col] = mapes[i]

            columns = ['date', 'qty_sold'] + [f"sca_m{i}" for i in range(13)] + \
                      [f"scf_as_m{i}" for i in range(13)] + \
                      ['pred_qty_sold'] + [f"scf_fs_m{i}" for i in range(13)]

            # Ater processing all the batches, the dataframe, containing the batch date, qty_sold, sc_m0, ..,sc_m12, scf_m0, ... scf_m12,
            # is created from the collected list
            df_for_scr = pd.DataFrame(df_for_scr, columns=columns)

            # Adding the SCR and SCRF to the dataframe and at the same time adding the 3 month and 12 month rolling window
            df_scr = adding_scrxQ_info_to_df(adding_scr_info_to_df(df_for_scr))

            # Update of the dictionary where we have one key per volume scale factor ([0.9, 1.0, 1.1]) containing as value the
            # corresponding dataframe
            df_scr_qtySold_study.update({'QtySoldScf{}'.format(volume_scf): df_scr})

            # Uncommenting the following 4 lines, will show you the SCR actual versus Forecasted for the particular sales volume scale factor
            # there is actualy a better way to show the Actual SCR versus Forecasted SCR containing the uncertainty band due to the
            # sales volume uncertainty (see bellow at the call "create_scr_figures_qtySold_study")
            # create_scr_figures(df_scr,
            #                   method='Regression (M. Chain)',
            #                   hierarchy=hierarchy,
            #                   qty_sold_scf=volume_scf)

            # another plot that can be activated to show the sensitivity to the number of months you are predicting into the future,
            # with respect to the training batches. Uncomment the 2 lines below if you want to use this study.
            # title = "Regression: LearningBatchSize={}".format(nr_reg_batches)
            # create_regression_mape_plot(res_df, title=title, hierarchy=hierarchy)

        # save dict into pickle for future use ()
        if not stop_flag:
            pickle_name = os.path.join('pickles', 'SCR_eval_period_{}.pkl'.format(hierarchy))
            if use_existing_pickles:
                with open(pickle_name, 'rb') as f:
                    df_scr_qtySold_study = pickle.load(f)
            else:
                f = open(pickle_name, 'wb')
                pickle.dump(df_scr_qtySold_study, f)

            # Creation of the Actual SCR and Forecasted SCR as a function of the batch (year and month)
            create_scr_figures_qtySold_study(df_scr_qtySold_study,
                                             method='Regression (M. Chain)',
                                             hierarchy=hierarchy,
                                             qty_sold_scf=volume_scfs,
                                             xQ=0)
            create_scr_figures_qtySold_study(df_scr_qtySold_study,
                                             method='Regression (M. Chain)',
                                             hierarchy=hierarchy,
                                             qty_sold_scf=volume_scfs,
                                             xQ=3)
            create_scr_figures_qtySold_study(df_scr_qtySold_study,
                                             method='Regression (M. Chain)',
                                             hierarchy=hierarchy,
                                             qty_sold_scf=volume_scfs,
                                             xQ=12)

            # Collecting the data for the sunburst graph for the SCR, -LQ and -LY
            # move the 'scrf_as' into 'scrf_fs' If you want to see the sunburtst graphs with the forecasted sales
            # instead of the actual sales
            sunburst_data.append([hierarchy] +
                                 list(get_mape_from_series(
                                     get_vol_diff_from_df(
                                         df_scr_qtySold_study['QtySoldScf1.0']['scr'],
                                         df_scr_qtySold_study['QtySoldScf1.0']['scrf_fs']))) +
                                 [df_scr_qtySold_study['QtySoldScf1.0']['qty_sold'].mean()]
                                 )
            sunburst_data_LQ.append([hierarchy] +
                                    list(get_mape_from_series(
                                        get_vol_diff_from_df(
                                            df_scr_qtySold_study['QtySoldScf1.0']['scr_rol_win03'],
                                            df_scr_qtySold_study['QtySoldScf1.0']['scrf_fs_rol_win03']))) +
                                    [df_scr_qtySold_study['QtySoldScf1.0']['qty_sold'].mean()]
                                    )
            sunburst_data_LY.append([hierarchy] +
                                    list(get_mape_from_series(
                                        get_vol_diff_from_df(
                                            df_scr_qtySold_study['QtySoldScf1.0']['scr_rol_win12'],
                                            df_scr_qtySold_study['QtySoldScf1.0']['scrf_fs_rol_win12']))) +
                                    [df_scr_qtySold_study['QtySoldScf1.0']['qty_sold'].mean()]
                                    )

    pickle_name = os.path.join('pickles', 'SunBurst.pkl')
    f = open(pickle_name, 'wb')
    pickle.dump(sunburst_data, f)

    pickle_name = os.path.join('pickles', 'SunBurst_LQ.pkl')
    f = open(pickle_name, 'wb')
    pickle.dump(sunburst_data_LQ, f)

    pickle_name = os.path.join('pickles', 'SunBurst_LY.pkl')
    f = open(pickle_name, 'wb')
    pickle.dump(sunburst_data_LY, f)

    # Generating the sunburst graphs for the SCR, -LQ and -LY (you can play with the colormap)
    create_scr_sunburst_figure_hierarchy(sunburst_data, 'SCR', 'piyg')  # 'rdbu'
    create_scr_sunburst_figure_hierarchy(sunburst_data_LQ, 'SCR_LQ', 'piyg')
    create_scr_sunburst_figure_hierarchy(sunburst_data_LY, 'SCR_LY', 'piyg')





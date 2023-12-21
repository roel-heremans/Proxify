# Databricks notebook source
# MAGIC %pip install -U scikit-learn==1.3.1

# COMMAND ----------

# MAGIC %md
# MAGIC Due to the fact that I did not have the right permissions to run my python script "10b_regression_MarkovChain.py" with: 
# MAGIC %run "./10b_regression_MarkovChain.py" 
# MAGIC in databricks, I copied the code in this notebook and run it this way.
# MAGIC
# MAGIC When i ran:
# MAGIC  > ```%run "./10b_regression_MarkovChain.py" ```
# MAGIC
# MAGIC I received:
# MAGIC > Notebook not found: Repos/roel.heremans@electrolux.com/dev_quality_service_call_forecasting/notebooks/02_modeling/01_modeling_using_sample_data/10b_regression_MarkovChain.py. Notebooks can be specified via a relative path (./Notebook or ../folder/Notebook) or via an absolute path (/Abs/Path/to/Notebook). Make sure you are specifying the path correctly.
# MAGIC
# MAGIC Also when i tried:
# MAGIC >```%fs ls "Repos/roel.heremans@electrolux.com/dev_quality_service_call_forecasting/notebooks/02_modeling/01_modeling_using_sample_data/" ```
# MAGIC
# MAGIC I got: 
# MAGIC >  AccessControlException: Permission denied: user [roel.heremans] does not have [read] privilege on [dbfs:/Repos/roel.heremans@electrolux.com/dev_quality_service_call_forecasting/notebooks/02_modeling/01_modeling_using_sample_data]
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Binary Classification Example
# MAGIC
# MAGIC Idea: Use a markov chain to model the probability of a service call given the
# MAGIC number of months since the product was purchased, the month of the year and
# MAGIC the date of purchase, conditional on no previous service call beeing made
# MAGIC
# MAGIC Why not use a hazard model instead? One reason why a hazard model may not be
# MAGIC the best choice is that the basic hazard models miss important features of the
# MAGIC data, such as seasonality and the tendancy to make calls just before the
# MAGIC warranty expires. Also, we are not actually modelling the time to failure, but
# MAGIC the time decision to make a service call when the customer thinks a failure
# MAGIC has occured. A hazard model may be the most appropriate to model the actual
# MAGIC time ot failure, but customers may wait for a couple of months after the
# MAGIC failure to report the failure, or not report during summer, which could be why
# MAGIC calls are made just before the warranty expires. 
# MAGIC
# MAGIC In the markov chain, the probability to make a service call could be modeled
# MAGIC by a number of different binary classification models. A good start would be
# MAGIC to use the logistic regression model. 
# MAGIC
# MAGIC We define the probability for a service call at t months after a purchase made
# MAGIC at time T, during month M, conditional on now service call before time t, as
# MAGIC $P(t| T, M)$. The parameters of the model are defined as :
# MAGIC - $x_t$ : parameter for time since purchase [13 x 1]
# MAGIC - $y_T$ : parameter for batch id [1 x number_of_batches]
# MAGIC - $z_M$ : parameter for month of year [12 x 1] \
# MAGIC and so the full parameter vector $\theta = [x_t, y_T, z_M]$. \
# MAGIC The utility of making a service call at time t, given the time of purchase
# MAGIC $T$ is defined as: \
# MAGIC $U(t,T,M) = x_t + y_T + z_M$ and the probability is given by: \
# MAGIC $P(t|T,M) = 1/(1+exp(-U(t,T,M)))$
# MAGIC
# MAGIC  x = [1,2,3]
# MAGIC  y = [0.1,
# MAGIC       0.2,]
# MAGIC  z = [0.01, 0.02, 0.03,...]
# MAGIC  Purchase during 202001 (t=0)
# MAGIC  P(m0 | T=202001) = 1/(1+exp(-(x[0] + y[0] + z[0])))
# MAGIC  P(m1 | T=202001) = 1/(1+exp(-(x[1] + y[0] + z[1])))
# MAGIC  P(m0 | T=202002) = 1/(1+exp(-(x[0] + y[1] + z[1])))
# MAGIC  P(m1 | T=202002) = 1/(1+exp(-(x[1] + y[1] + z[2])))
# MAGIC
# MAGIC
# MAGIC ## Calculate service calls per time period
# MAGIC Let qty(T, t) be the number of purchases made at time T that have not had any
# MAGIC service call up to time t. The number service calls made by at time t for
# MAGIC purchases made at time T:
# MAGIC
# MAGIC $Q(t=0 |T ) = qty(T, t=0) \cdot P(t=0 | T )$
# MAGIC
# MAGIC $qty(T, t=1) = qty(T, t=0) - Q(t=0 |T)$
# MAGIC
# MAGIC And for any time $t$:
# MAGIC
# MAGIC $Q(t |T) = qty(T, t-1) \cdot P(t | T ) $
# MAGIC
# MAGIC %%

# COMMAND ----------

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
from utils.util import generate_exponential_weights, get_all_csv_files, get_mapes, get_mape_from_series, get_vol_diff_from_df
from utils.util import int_to_month, reorder_df
from utils.util import sc_mi, sc_mi_mat, stripping_future_off, to_matrix, transform_data_for_scr_df
from utils.util import create_scr_figures_qtySold_study, create_scr_sunburst_figure_hierarchy

# Filter out the UserWarning coming from the one hot encoding for the test data
warnings.filterwarnings("ignore", category=UserWarning, message="Found unknown categories in columns")
warnings.filterwarnings("ignore", category=UserWarning, message="FixedFormatter should only be used together with FixedLocator")
warnings.filterwarnings('ignore', category=RuntimeWarning)

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

    def get_feature_dataframe(self, df: pd.DataFrame, qts_scale_factor = 1) -> pd.DataFrame:
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

# COMMAND ----------

if __name__ == '__main__':
    
    # checking and creating paths if necessary
    create_necessary_paths()
    
    # if pickles exist you can use them instead of having to run the analysis again and again
    use_existing_pickles = False

    # These are the number of batches used in the regression model
    nr_reg_batches = 12

    #Nr of months in warranty period (m0,m1,m2,m3,m4,...m12)
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
    volume_scfs = [0.9, 1.0, 1.1]

    hierarchy_files = sorted(get_all_csv_files(os.path.join('data', 'full_sc_fs_tables'), start_with='sc_fs_', end_with='.csv'))

    # just running this one, instead of the full list
    # hierarchy_files = ['foodpreparation_all_all', 'dishcare_all_all', 'fabriccare_all_all', 'foodpreservation_all_all']
    #hierarchy_files = ['FabricCare_2460cmFrontLoadDryerFreestanding_60cmElectricVentedDryer']

    # this is the variable that collects the mapes of each product hierarchy so that the sunburst chart can be created
    sunburst_data = []
    sunburst_data_LQ = []
    sunburst_data_LY = []

    # looping over all the product hierarchies
    for hierarchy_i, hierarchy in enumerate(hierarchy_files):
    #for hierarchy_i, hierarchy in enumerate(['FoodPreparation_FreestandingCookerFrontControl_FSCookerFCElectCavityInductionHob']):
        stop_flag = False
        print("{}/{}: {}".format(hierarchy_i+1, len(hierarchy_files), hierarchy))

        # when running 3 times for volume_scfs = [0.9, 1.0, 1.1], this dict collects the results for each of the 3 runs, 
        # and the info will be used to add the blue uncertainty band to the scr plots
        df_scr_qtySold_study = {}

        # looping over the scale factors for the sales volume [0.9, 1.0, 1.1]
        for scf_count, volume_scf in enumerate(volume_scfs):
            print("--> {}/{}: {}".format(scf_count+1, len(volume_scfs), volume_scf))


            df_sc_fs = pd.read_csv(os.path.join("data", "full_sc_fs_tables", "sc_fs_{}.csv".format(hierarchy)))
            df_sc, df_sales_forecast = stripping_future_off(df_sc_fs)
            df_sc_replaced_zeros, nr_interventions = check_on_zeros(df_sc)

            # nr_interventions are the count of zeros that were changed into ones.
            # although we want to see what the amount of interventions is, not taking into account the the lower triangle (containing zeros because those service calls
            # are from the future). We have a lower triangle of zeros, due to the fact that those service calls are not yet know since they will only be know in future 
            # months. For instance when the current date is 15th of November 2023 (i.e. batch 202311), then the service calls for the batch 202310 will contain only 
            # sc_m0 different from 0. sc_m1, sc_m2, ... sc_m12 will only be filled in the respective months 202312, 202401, ...202412, Hence the lower triangle with
            # zeros. Removing the amount of all those zeros is equal to (nr_mi+1)*(nr_mi)/2 = 91, for nr_mi = 13
            nr_interventions = int(nr_interventions - (nr_mi+1)*(nr_mi)/2)

            # if you need the pred_qty_sold instead of qty_sold you need to adapt the code here
            scs_cols = ['period','qty_sold'] + [f"m{i}" for i in range(13)]
            df_sc = df_sc_replaced_zeros[scs_cols].copy()

            # Probably not enough service calls for this product to deliver a worthy model/prediction
            # An additional study needs to happen to find out what the effect on the average mape is due to low data samples, or as a function of the 
            # amount of zeros which were flipped into 1. For now we decide not to trust the model anymore when we flipped more than 10 zeros to ones.
            if nr_interventions > 10:
                print(f'Be carefull with this dataset,\n{hierarchy} there seem to be more than {nr_interventions} service call values to be zero.')
                stop_flag = True
                break

            if df_sc.shape[0] < nr_reg_batches*2:
                print(f'--> Aborting: Not enough batches to get the first value of SCR. Only {df_sc.shape[0]}-batches available')
                stop_flag = True
                break

            has_negative_values = (df_sc[ [f"m{i}" for i in range(13)]] < 0).any().any()
            if has_negative_values:
                print(f'--> Aborting: There are ServiceCalls in the full data matrix with negative values.')
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
                df_train = dp.get_train_data(df.iloc[:, i:i+nr_reg_batches])
                df_test = dp.get_test_data(df.iloc[:, i+nr_reg_batches+next_pred_id], method=test_oh_choice, qts_scale_factor=volume_scf)
                #print("Learning on: {}".format(df.columns[i:i+nr_reg_batches].values))
                #print("Predicting on: {}".format(df.columns[i+nr_reg_batches+next_pred_id].values))
                pred_columns = df.columns[i+nr_reg_batches+next_pred_id].values

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
                
                #Roels method to reconstruct to SCs for the predictions, based on recursive function:
                pred_calls =sc_mi_mat(sum_remaining, p_pred)

                # Roels method to reconstruct to SCs for the Actual number of Service Calls at time m0,m1,m2,..,m12
                serv_calls = sc_mi_mat(to_matrix(df_test["weight"]),to_matrix(df_test["y"]))

                mapes = get_mapes(serv_calls, pred_calls)
                df_pred = pd.DataFrame(columns=['Actual','Forecast'], index='m' + pd.RangeIndex(start=0, stop=13, step=1, name='Month').astype(str))
                df_pred['Actual'] = serv_calls[:,0]
                df_pred['Forecast'] = pred_calls[:,0]

                # create for each batch that got predicted, a control plot on the service call prediction quality, only when there is no scale applied on the 
                # sales volume
                #if volume_scf == 1.0:
                #    title_txt = "Reg Fit Result: {}\nLearningSize={}, Predicting: {}".format(
                #        hierarchy,nr_reg_batches,pred_columns[0])
                #    create_regression_control_plot(df_pred, title=title_txt, hierarchy=hierarchy)

                # Collecting to the data batch date, qty_sold, actual SCs, forecasted SCs, in order to create a dataframe later on.
                # This dataframe will be the input to add the SCR and SCR_forecasted
                df_for_scr.append(transform_data_for_scr_df(df.iloc[:, i+nr_reg_batches], pred_calls[:,0]))

                for i, col in enumerate(pred_columns):
                    res_df.at[rowlist[i], col] = mapes[i]

                columns=['date', 'qty_sold',
                             'sca_m0','sca_m1', 'sca_m2', 'sca_m3', 'sca_m4', 'sca_m5',
                             'sca_m6', 'sca_m7', 'sca_m8', 'sca_m9', 'sca_m10', 'sca_m11', 'sca_m12',
                             'scf_m0', 'scf_m1', 'scf_m2', 'scf_m3', 'scf_m4', 'scf_m5',
                             'scf_m6', 'scf_m7', 'scf_m8', 'scf_m9', 'scf_m10', 'scf_m11', 'scf_m12']
            
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
            #create_scr_figures(df_scr,
            #                   method='Regression (M. Chain)',
            #                   hierarchy=hierarchy,
            #                   qty_sold_scf=volume_scf)

            # another plot that can be activated to show the sensitivity to the number of months you are predicting into the future,
            # with respect to the training batches. Uncomment the 2 lines below if you want to use this study.
            #title = "Regression: LearningBatchSize={}".format(nr_reg_batches)
            #create_regression_mape_plot(res_df, title=title, hierarchy=hierarchy)
        
        # save dict into pickle for future use ()
        if not stop_flag:
            pickle_name = os.path.join('pickles','SCR_eval_period_{}.pkl'.format(hierarchy))
            if use_existing_pickles:
                with open(pickle_name, 'rb') as f:
                    df_scr_qtySold_study = pickle.load(f)
            else:
                f = open(pickle_name, 'wb')
                pickle.dump(df_scr_qtySold_study,f)

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
            sunburst_data.append([hierarchy] +
                             list(get_mape_from_series(
                                 get_vol_diff_from_df(
                                     df_scr_qtySold_study['QtySoldScf1.0']['scr'],
                                     df_scr_qtySold_study['QtySoldScf1.0']['scrf'] ))) +
                             [df_scr_qtySold_study['QtySoldScf1.0']['qty_sold'].mean()]
                             )
            sunburst_data_LQ.append([hierarchy] +
                                list(get_mape_from_series(
                                    get_vol_diff_from_df(
                                        df_scr_qtySold_study['QtySoldScf1.0']['scr_rol_win03'],
                                        df_scr_qtySold_study['QtySoldScf1.0']['scrf_rol_win03'] ))) +
                                [df_scr_qtySold_study['QtySoldScf1.0']['qty_sold'].mean()]
                                )
            sunburst_data_LY.append([hierarchy] +
                                list(get_mape_from_series(
                                    get_vol_diff_from_df(
                                        df_scr_qtySold_study['QtySoldScf1.0']['scr_rol_win12'],
                                        df_scr_qtySold_study['QtySoldScf1.0']['scrf_rol_win12'] ))) +
                                [df_scr_qtySold_study['QtySoldScf1.0']['qty_sold'].mean()]
                                )
    
    # Generating the sunburst graphs for the SCR, -LQ and -LY (you can play with the colormap)
    create_scr_sunburst_figure_hierarchy(sunburst_data,'SCR', 'piyg') #'rdbu'
    create_scr_sunburst_figure_hierarchy(sunburst_data_LQ,'SCR_LQ', 'piyg')
    create_scr_sunburst_figure_hierarchy(sunburst_data_LY,'SCR_LY', 'piyg')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Displaying the sunburst_data, 
# MAGIC can be useful after processing on new input data, to update the sunburst_data in the function "load_sunburst_data" in utils.util
# MAGIC The sunburst data:
# MAGIC ++++++++++++++++++
# MAGIC [['DishCare_BIFullsizeDishWasherStainless_BIFullsizeDishWasherStainless', 6.782751478868598, 4.463117797682699, 7.6662526108314255, 7.6662526108314255, 90229.93617021276], ['DishCare_BIFullsizeDishWasherStainless_all', 6.782751478868598, 4.463117797682699, 7.6662526108314255, 7.6662526108314255, 90229.93617021276], ['DishCare_BISlimlineDishWasher_BISlimlineDishWasher', 6.983839532802256, 5.356711368853457, 6.538186152930451, 6.538186152930451, 16873.59574468085], ['DishCare_BISlimlineDishWasher_all', 6.983839532802256, 5.356711368853457, 6.538186152930451, 6.538186152930451, 16873.59574468085], ['DishCare_FSFullsizeDishWasherStainless_FSFullsizeDishWasherStainless', 6.613433646881363, 4.109109109109106, 6.96064431285243, 6.96064431285243, 18875.94680851064], ['DishCare_FSFullsizeDishWasherStainless_all', 6.613433646881363, 4.109109109109106, 6.96064431285243, 6.96064431285243, 18875.94680851064], ['DishCare_FSSlimlineDishWasher_FSSlimlineDishWasher', 13.734465160336473, 12.004909983633386, 8.242385493497348, 8.242385493497348, 2668.1170212765956], ['DishCare_FSSlimlineDishWasher_all', 13.734465160336473, 12.004909983633386, 8.242385493497348, 8.242385493497348, 2668.1170212765956], ['DishCare_all_all', 6.96390121847371, 4.092165571822953, 8.15645428533983, 8.15645428533983, 132950.585106383], ['FabricCare_2460cmFLWashingMachineFS_2460cmFrontLoadWashingMachineFS', 6.364855721383529, 3.560212528202805, 8.018357875170544, 8.018357875170544, 98085.5], ['FabricCare_2460cmFLWashingMachineFS_all', 6.364855721383529, 3.560212528202805, 8.018357875170544, 8.018357875170544, 98085.5], ['FabricCare_2460cmFrontLoadDryerFreestanding_60cmElectricStandardCondenseDryer', 7.7302185959722145, 6.267051373203918, 5.482344678465313, 5.482344678465313, 11502.234042553191], ['FabricCare_2460cmFrontLoadDryerFreestanding_60cmHeatPumpDryer', 8.233204425630662, 6.130840931292745, 6.319274693388767, 6.319274693388767, 49729.47872340425], ['FabricCare_2460cmFrontLoadDryerFreestanding_all', 7.438194842208678, 6.084960866066864, 5.952209354053895, 5.952209354053895, 64134.15957446808], ['FabricCare_BuiltinFrontLoadWashingMachine_BuiltinFrontLoadWashingMachine', 9.226005196703914, 6.440092165898623, 7.9337251032453056, 7.9337251032453056, 6058.31914893617], ['FabricCare_BuiltinFrontLoadWashingMachine_all', 9.20526223655548, 6.814108830237867, 7.6719459156181875, 7.6719459156181875, 6587.063829787234], ['FabricCare_BuiltinWasherDryerFrontLoad_all', 11.27516402867757, 9.214609692050615, 9.216205560907117, 9.216205560907117, 4333.13829787234], ['FabricCare_FrontLoadWasherDryerFreestanding_FrontLoadWasherDryerFreestanding', 6.901305589406063, 4.609696068429273, 6.729883391084125, 6.729883391084125, 13388.106382978724], ['FabricCare_FrontLoadWasherDryerFreestanding_all', 6.5483285183498765, 3.789202611974618, 6.88543269729046, 6.88543269729046, 13754.68085106383], ['FabricCare_HorizontalAxisTopLoadWashingMachine_HorizontalAxisTopLoadWashingMachine', 7.043225417249437, 4.190080759945577, 7.656015414992349, 7.656015414992349, 27954.957446808512], ['FabricCare_HorizontalAxisTopLoadWashingMachine_all', 7.056676772187923, 4.190080759945577, 7.652123061531801, 7.652123061531801, 27954.957446808512], ['FabricCare_SlimFrontLoadWashingMachineFS_SlimFrontLoadWashingMachineFS', 10.899212102996552, 6.93209105869845, 11.342060020159483, 11.342060020159483, 20143.255319148935], ['FabricCare_SlimFrontLoadWashingMachineFS_all', 10.850259887772227, 6.865861188291099, 11.390732000968857, 11.390732000968857, 20659.81914893617], ['FabricCare_all_all', 6.702940520680092, 4.105882844555511, 8.68825711087652, 8.68825711087652, 251291.74468085106], ['FoodPreparation_BuiltInOven_BuiltinCompactOven', 8.270589823979469, 6.339637509850267, 7.220129190572085, 7.220129190572085, 12728.574468085106], ['FoodPreparation_BuiltInOven_BuiltinDualCavityElectricOven', 11.736431449600994, 10.296354447769989, 9.4878169331486, 9.4878169331486, 6149.021276595745], ['FoodPreparation_BuiltInOven_BuiltinElectricOven', 7.879483385871187, 5.94398466588566, 8.476283249030146, 8.476283249030146, 120818.68085106384], ['FoodPreparation_BuiltInOven_BuiltinSteamOven', 8.474593194066406, 5.839238845144347, 8.041537207830459, 8.041537207830459, 14833.893617021276], ['FoodPreparation_BuiltInOven_all', 7.883420790461908, 5.523633071586879, 9.267677943488371, 9.267677943488371, 156664.52127659574], ['FoodPreparation_BuiltinHob_BuiltinGasHob', 11.935061634473874, 8.593677049289662, 10.250776595641094, 10.250776595641094, 30839.872340425532], ['FoodPreparation_BuiltinHob_BuiltinInductionHob', 7.324610700905166, 5.153501584180056, 7.714057080887877, 7.714057080887877, 83573.27659574468], ['FoodPreparation_BuiltinHob_BuiltinRadiantHob', 7.9138412375444664, 5.691903612020692, 6.847426812400359, 6.847426812400359, 59882.765957446805], ['FoodPreparation_BuiltinHob_all', 8.276658866197284, 5.458546438945327, 8.488905329615054, 8.488905329615054, 179485.05319148937], ['FoodPreparation_FreestandingCookerFrontControl_FSCookerFCElectCavityRadiantHob', 7.674774346480641, 6.489328173970902, 6.563266872140828, 6.563266872140828, 21047.936170212764], ['FoodPreparation_FreestandingCookerFrontControl_FSCookerFCElectDualCavityRadiaHob', 12.611828598070623, 11.237254901960785, 10.438626429760808, 10.438626429760808, 3459.436170212766], ['FoodPreparation_FreestandingCookerFrontControl_FSCookerFCGasDualCavityGasHob', 13.679128777239312, 12.276295133437987, 9.938648032474406, 9.938648032474406, 1286.7340425531916], ['FoodPreparation_FreestandingCookerFrontControl_all', 6.001625837693976, 4.0353578914542645, 6.548380018524122, 6.548380018524122, 49119.97872340425], ['FoodPreparation_Hood_PulloutHood', 13.511098319378432, 10.14092553799276, 10.182160210917116, 10.182160210917116, 12679.611111111111], ['FoodPreparation_all_all', 7.869443760675242, 4.855538469987367, 9.825388690876215, 9.825388690876215, 408226.8191489362], ['FoodPreservation_BuiltinFoodPreservation_BuiltinBottomFreezer', 8.453453425454544, 5.3828416276174735, 8.560939235377033, 8.560939235377033, 42434.574468085106], ['FoodPreservation_BuiltinFoodPreservation_BuiltinUnderCounterRefrigerator', 8.834725168577572, 5.896841495218779, 8.588044885383699, 8.588044885383699, 15372.617021276596], ['FoodPreservation_BuiltinFoodPreservation_BuiltinUprightRefrigerator', 6.929602645969796, 4.640596063568628, 7.187656366636466, 7.187656366636466, 16882.31914893617], ['FoodPreservation_BuiltinFoodPreservation_all', 7.317292727411509, 3.810625328707774, 9.24783814787652, 9.24783814787652, 85101.37234042553], ['FoodPreservation_FreestandingBottomFreezer_SmallBottomFreezer', 6.044241477875582, 4.0508511096746425, 5.350174157117557, 5.350174157117557, 18689.85106382979], ['FoodPreservation_FreestandingBottomFreezer_all', 5.996757309561954, 3.8986371236046056, 5.3685114304013455, 5.3685114304013455, 18692.85106382979], ['FoodPreservation_FreestandingFreezer_FreestandingUprightFreezer', 8.465233540335397, 7.006680514231185, 7.175699935788224, 7.175699935788224, 12053.478723404256], ['FoodPreservation_FreestandingFreezer_all', 8.373497142928496, 6.956958480457808, 7.142608713790465, 7.142608713790465, 12084.22340425532], ['FoodPreservation_FreestandingRefrigerator_FreestandingUprightRefrigerator', 7.882454446461743, 6.191661182077164, 7.423295004700402, 7.423295004700402, 16639.31914893617], ['FoodPreservation_FreestandingRefrigerator_all', 7.889053688294903, 6.2259912835625055, 7.417404878945311, 7.417404878945311, 16655.64893617021], ['FoodPreservation_all_all', 7.246149279090088, 4.1374909567026625, 9.359313565119196, 9.359313565119196, 165027.0744680851]]

# COMMAND ----------

print('New sunburst data - Nr of product hierarchies considered in SunBurst: {}'.format(len(sunburst_data)))

# COMMAND ----------

print("The sunburst data:")
print("++++++++++++++++++")
print(sunburst_data)

# COMMAND ----------



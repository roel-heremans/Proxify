import os


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pickle
from utils import  create_box_figures, create_heatmap, create_scatter, my_preprocessor, read_data

my_selection = [[0, 1, 6, 7, 8], [20, 22, 23, 24, 25], [26, 27, 28, 29, 30, 31],
                [52, 53, 54, 55, 56],[62, 63, 64, 65, 66],[82, 83, 84, 85, 86],
                [87, 88, 89, 90, 91],[97, 98, 99, 100, 101],[102, 103, 104, 105, 106],
                [107, 108, 109, 110, 111],[112, 113, 114, 115, 116],[117, 118, 119, 120, 121],
                [122, 123, 124, 125, 126],[137, 138, 139, 140, 141],[142, 143, 144, 145, 146],
                [147, 148, 149, 150, 151],[157, 158, 159, 160, 161],[162, 163, 164, 165, 166],
                [167, 168, 169, 170, 171],[172, 173, 174, 175, 176],[177, 178, 179, 180, 181],
                [182, 183, 184, 185, 186],[197, 198, 199, 200, 201],[202, 203, 204]]

if __name__ == "__main__":
    out_dir ='output'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ####################################################################
    site = '261_new'                                                   #
    nr_col_ev = 10      # number of plots on one page for EV           #
    nr_col_meter = 10   # number of plots on one page for meters       #
    target_id_list = [0]  # Give id of the meter     #
                                            # (e.g. 0 for Total_TGBT)  #
    plots_on = False     # True for getting data exploratory plots      #
    ####################################################################

    df_evs, df_meters, value_map = read_data(site)

    # removing all columns with _lag in the name
    lag_cols = [i for i, col in enumerate(df_evs.columns) if '_lag' in col]
    df_evs = df_evs.drop(df_evs.columns[lag_cols], axis=1)

    # removing columns with 0 variance
    df_evs = df_evs.loc[:, df_evs.var() != 0]

    # Making box plots of all the EV variables
    if plots_on:
        key_out = '_EVO_'
        create_box_figures(df_evs, site, nr_col_ev, key_out)

    # Making box plots of all the meters variables
    if plots_on:
        key_out = '_meters_'
        create_box_figures(df_meters, site, nr_col_meter, key_out)

    # rename columns into a numeric value (column number)
    df_evs, ev_name_mapper = my_preprocessor(df_evs)
    df_meters, meters_name_mapper = my_preprocessor(df_meters)

    for target_id in target_id_list:
        print('\nStart XGBoost training')
        print('************************')
        print('Site: {}'.format(site))
        print('Target: {}'.format(meters_name_mapper[target_id]))


        # removing columns which are highly correlated
        #corr_matrix = df_evs.corr().abs()
        #upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        #to_drop_unique = set()
        #for column in upper.columns:
        #    to_drop = [i for i, val in enumerate(upper[column]) if val > 0.8]
        #    if to_drop:
        #        print("{}: {}".format(column, to_drop))
        #        for val in to_drop:
        #            to_drop_unique.add(val)
        #df_evs = df_evs.drop(df_evs.columns[list(to_drop_unique)], axis=1)

        # dropping all vars related to the date-time
        if plots_on:
            evs_to_drop = [7,8,9,10,11,12,13,14,15,16,18,19,20,21]
            evs_to_drop = []
            df_evs.drop(df_evs.iloc[:, evs_to_drop], inplace=True, axis=1)
            create_heatmap(df_evs, site, '_EVO_')
            create_scatter(df_evs, site, '_EVO_')

        if plots_on:
            meters_to_drop = range(7, df_meters.shape[1])
            df_meters.drop(df_meters.iloc[:, meters_to_drop], inplace=True, axis=1)
            create_heatmap(df_meters, site, '_meters_')
            create_scatter(df_meters, site, '_meters_')

        data = pd.merge(df_evs, df_meters.iloc[:, target_id], how='inner', left_index=True, right_index=True)
        target = data.iloc[:, -1].name
        data['is_weekend'] = np.where(data.index.dayofweek >= 5, 1, 0)


        # Split the data into training and testing sets
        train_data = data[:'2022-12-31']
        test_data = data['2023-01-01':]

        # Splitting the features and the target variable
        features = [col for col in data.columns if col != target]

        # Create DMatrix objects for training and testing
        train_dmatrix = xgb.DMatrix(train_data[features], train_data[target])
        test_dmatrix = xgb.DMatrix(test_data[features], test_data[target])

        # Define the hyperparameters for the XGBoost model
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }

        # Train the XGBoost model using cross-validation
        cv_results = xgb.cv(
            params=params,
            dtrain=train_dmatrix,
            num_boost_round=1000,
            early_stopping_rounds=10,
            nfold=5,
            verbose_eval=10,
            metrics=['rmse']
        )

        # Get the optimal number of boosting rounds based on cross-validation
        num_boost_rounds = cv_results['test-rmse-mean'].idxmin()

        # Train the final XGBoost model with the optimal number of boosting rounds
        model = xgb.train(
            params=params,
            dtrain=train_dmatrix,
            num_boost_round=num_boost_rounds,
            evals=[(test_dmatrix, 'Test')],
            verbose_eval=10
        )

        # Make predictions on the test set
        y_pred = model.predict(test_dmatrix)

        # Calculate the root mean squared error
        rmse = mean_squared_error(test_data[target], y_pred, squared=False)
        print('Root Mean Squared Error:', rmse)

        # Save the trained model to a file using pickle
        key_out = '_XGB_'+str(target_id)
        model_out = os.path.join('output', 'site' + str(site) + key_out + '.pkl')

        objects_to_dump = {
        'model': model,
        'X_train': train_data[features],
        'y_train': train_data[target],
        'X_test': test_data[features],
        'y_test': test_data[target]
        }

        with open(model_out, 'wb') as file:
            pickle.dump(objects_to_dump, file)

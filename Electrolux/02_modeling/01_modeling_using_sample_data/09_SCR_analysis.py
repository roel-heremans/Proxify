import os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D


def lstm_train(input_array, **kwargs):
    ''' This is a first approach to estimate the SCR based on previous scr values. An LSTM model is used in order to do
     the prediction. This is a very temporary attempt and is currently not worked on for the moment
     The keyword argument "look_back" tells us how much historic values of the SCR are considered in order to forecast
     the next one. This function return the model that can be used to do the prediction on new data.
    '''

    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    # Generate a small time series dataset (example data)
    time_series = input_array

    # Define the number of previous time steps to use for prediction
    look_back = kwargs.pop('look_back', 3)

    # Create sequences of input data and their corresponding targets
    X, y = [], []
    for i in range(len(time_series) - look_back):
        X.append(time_series[i:i + look_back])
        y.append(time_series[i + look_back])

    X = np.array(X)
    y = np.array(y)

    # Reshape the input data for LSTM (samples, time steps, features)
    X = X.reshape(X.shape[0], look_back, 1)

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(X, y, epochs=100, batch_size=1, verbose=2)

    return model


def lstm_predict(model, last_sequence, **kwargs):
    '''Getting the model and a sequence of input data, perform the next month prediction.'''

    # Make a prediction for the next value in the time series
    look_back = kwargs.pop('look_back', 3)
    last_sequence = last_sequence.reshape(1, look_back, 1)
    next_value = model.predict(last_sequence)

    return next_value[0][0]


def create_scr_figures(result_dict: dict) -> dict:
    ''' create figures where the scr actual is compared with the forecasted once, as well as vollume difference
        plots.'''
    scr_dict = {}
    for hierarchy in ['line', 'group', 'subgroup']:
        for method in ['weibull', 'gamma', 'mm2w', 'lfp', 'tb','avg', 'naive']:
            print('{}_{}'.format(hierarchy,method))
            df = result_dict['{}_{}_in_out_df'.format(method,hierarchy)]
            scr_df = get_scr_df(df)

            fig, ax = plt.subplots(2,1, figsize=(15,5), sharex=True)
            ax[0].plot(scr_df['scr'], marker='o', color='k', label='Actual')
            ax[0].plot(scr_df['scrf'], marker='o', color='b', label='Predicted')
            ax[0].set_xlabel('Month')
            ax[0].set_ylabel('Serv Call Ratio')
            ax[0].grid()
            ax[0].legend()

            vol_diff = (scr_df['scr'] - scr_df['scrf']) / scr_df['scr'] * 100
            mape = np.mean(abs(vol_diff))
            mdape = np.median(abs(vol_diff))
            mape_std = np.std(abs(vol_diff))
            scr_dict.update({'{}_{}'.format(method, hierarchy): [mape, mdape, mape_std]})
            ax[0].set_title('Product {}: {}    --       MAPE={:.2f}+-{:.2f}'.format(hierarchy, method, mape, mape_std))

            pos_vol_diff =  [val if val > 0 else 0 for val in vol_diff]
            neg_vol_diff =  [val if val < 0 else 0 for val in vol_diff]
            ax[1].bar(range(len(vol_diff)),pos_vol_diff,  color='green')
            ax[1].bar(range(len(vol_diff)),neg_vol_diff,  color='red')
            ax[1].set_ylabel('(Act-Pre)/Act [%]')
            ax[1].grid()
            plt.tight_layout()
            fig_name = os.path.join('plots','SCR_{}_{}.png'.format(method, hierarchy))
            plt.savefig(fig_name)
            plt.close()
    return scr_dict


def create_summary_figure(scr_dict: dict):
    ''' the final results in percentage error for the SCR calculation based on the prediction using a particular model
    is shown

    INPUT dict:
    {'weibull_line': [23.547081957148485, 21.326500524826585, 18.580083111964548],
     'gamma_line': [19.356897899037914, 17.258903441844367, 15.111518661588617],
     'mm2w_line': [1873.0016983130736, 1845.9426641100004, 313.71389997127017],
     'lfp_line': [19.237440183744678, 16.228334834295676, 13.883831251874104],
     'tb_line': [13.819310708118493, 13.016638462446153, 11.360926133013729],
     'avg_line': [17.067218955384792, 16.587388137376877, 8.614866440549662],
     'naive_line': [11.447229929672858, 11.682270852776245, 8.35370034009824],
     'weibull_group': [28.01441217802734, 24.543754513081005, 20.233572747647],
     'gamma_group': [23.35920476330504, 17.30258538517963, 18.57078294849129],
     ...
     'naive_subgroup': [12.522825223209376, 8.8365748645238, 11.781948018562124]}

     OUTPUT:
     summary figure showing for both methods (IC and IRC) the percentage error on the SCR for the different models.
    '''


    # mapping between model and [color, horizontal shift for the plot]
    models_cmap = {'weibull': ['b',-3], 'gamma': ['r', -2], 'lfp': ['g', -1], 'mm2w': ['m', 0], 'tb': ['c',1], 'naive': ['y', 2], 'avg': ['k', 3]}

    table_dict = {}

    for hierarchy in ['line', 'group', 'subgroup']:
        fig, ax = plt.subplots()
        # Plot data for each method
        for method in ['mape','mdape']:
            i = 0
            for model in models_cmap.keys():
                mape, mdape, stdev = scr_dict['{}_{}'.format(model, hierarchy)]
                trans = Affine2D().translate(+0.1*models_cmap[model][1], 0.0) + ax.transData
                label = '{}'.format(model)
                if method == 'mape':
                    mean = mape
                else:
                    mean = mdape

                ax.errorbar(method, mean, yerr=stdev, marker='o', color=models_cmap[model][0], label=label, transform=trans)
                i+=1
        # Set labels and title
        ax.set_xlabel('Methods')
        ax.set_ylabel('Percentage Error [%]')
        ax.set_title('Product {} - {}'.format(hierarchy.replace("_",""),'SCR'))
        # Create the legend with unique items
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = []
        unique_handles = []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
        # Display the plot with the modified legend
        ax.legend(unique_handles, unique_labels,loc='upper left', bbox_to_anchor=(1, 1))
        # Display the plot
        plt.tight_layout()
        plt.grid()
        plt.ylim([0,80])
        fig_name = 'Summary_{}_{}.png'.format(hierarchy.replace("_",""),"scr")
        plt.savefig(os.path.join('plots',fig_name))


def extract_scr(df: pd.DataFrame, row_id: int, col_id: int,**kwargs) -> float:
    '''Given an input dataframe with at least 13 rows and at least 12 columns, this function extracts the SCR from it.
       Notice for an SCR calculation one needs 12 months sc data prior the actual date for which the scr is calculated.
       The service call data for the first month until the 12th month need to be in consecutive column ids and the
       col_id indicates where the first month sc value can be found in the df.
       df looks like:
                       date qty_sold sc_m1 sc_m2 sc_m3  ... sc_m8 sc_m9 sc_m10 sc_m11 sc_m12
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
       col_id = 2 since there are 2 columns before column 'sc_m1'.

       SCR for month i is calculated as follows:
       (nominator_part1 + nominator_part2 + nominator_part3) / (1/4 * denominator_part1 + 1/12 * denominator_part2 +
        1/16 * denominator_part3)
        where
        nominator_part1 = nr of service calls coming from a sale batch of month i-1 reported in month i
        nominator_part2 = nr of service calls coming from sales batches of months i-2, i-3 and i-4 reported in month i
        nominator_part3 = nr of service calls coming from sales batches of months i-5, i-6 .. i-12 reported in month i
        and where
        denominator_part1 = sales_volume from month i-1
        denominator_part2 = sum of the sales volumes from months i-2, i-3 and i-4
        denominator_part3 = sum of the sales volumes from months i-5, i-6 until i-12

        To get the scr value for date=201805 call the function as follows (the default qty_sold_colid=1 is taken):
        >> extract_scr(df, 12, 2)

    '''

    qty_sold_colid = kwargs.pop('qty_sold_colid',1)
    debug = kwargs.pop('debug', 0)

    scr = (sum([df.iloc[row_id-j,col_id-1+j] for j in range(1,2)])  + \
           sum([df.iloc[row_id-j,col_id-1+j] for j in range(2,5)]) + \
           sum([df.iloc[row_id-j,col_id-1+j] for j in range(5,13)]) \
           ) / \
          ( 0.25 * sum([df.iloc[row_id-j,qty_sold_colid] for j in range(1,2)]) + \
            1/12 * sum([df.iloc[row_id-j,qty_sold_colid] for j in range(2,5)]) + \
            1/16 * sum([df.iloc[row_id-j,qty_sold_colid] for j in range(5,13)]) \
            )
    if debug:
        print('First factor: sales={}, sc={}'.format([df.iloc[row_id-j,col_id-1+j] for j in range(1,2)],
                                                       [df.iloc[row_id-j,qty_sold_colid] for j in range(1,2)]))
        print('Second factor: sales={}, sc={}'.format([df.iloc[row_id-j,col_id-1+j] for j in range(2,5)],
                                                       [df.iloc[row_id-j,qty_sold_colid] for j in range(2,5)]))
        print('Third factor: sales={}, sc={}'.format([df.iloc[row_id-j,col_id-1+j] for j in range(5,13)],
                                                       [df.iloc[row_id-j,qty_sold_colid] for j in range(5,13)]))

    return scr


def get_scr_df(df: pd.DataFrame) -> pd.DataFrame:
    ''' Extracts all the possible scr and forcasted scr from an input dataframe df which is of the form:
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

    # first 12 months are needed to calculate the SCR based on the history of the sales and service call forcasts
    if len(df) > 12:
        for i in range(12,len(df)):
            scr  = extract_scr(df, i, 2)
            scrf = extract_scr(df, i,14)
            scr_df.append([df.iloc[i,0], scr, scrf])

    df_out = pd.DataFrame(scr_df, columns=['date', 'scr', 'scrf'])
    df_out.set_index('date', inplace=True)

    return df_out


def perform_lstm_forecast(df,**kwargs):
    predict_n = kwargs.pop('predict_n', 3)
    look_back = kwargs.pop('look_back', 4)
    hierarchy = kwargs.pop('hierarchy', 'Not specified')
    lstm_model = lstm_train(df.iloc[:-predict_n]['scr'].values, look_back=look_back)

    for i in range(predict_n):
        input = df.iloc[-(look_back + predict_n) + i:-predict_n + i]['scr'].values
        prediction = lstm_predict(lstm_model, input, look_back=look_back)
        actual = df.iloc[-predict_n + i]['scr']
        df.loc[df.iloc[df.shape[0] - predict_n + i].name, 'prediction'] = prediction
    df['percentage_error'] = (df['scr'] - df['prediction']) / df['scr'] *100
    ax = df[['scr', 'prediction']].plot(marker='o', figsize=(10, 5))
    ax.legend(['SCR Actual', 'SCR LSTM prediction'])
    ax.set_title('Product {}: MAPE={:.2f}+-{:.2f}   -   Look back={}'.format(hierarchy,
                                                                             np.mean(abs(df['percentage_error'])),
                                                                             np.std(abs(df['percentage_error'])),
                                                                             look_back))
    return ax

def create_lst_output_figures(result_dict):
    for hierarchy in ['line','group','subgroup']:
        for model in ['weibull']:
            df = result_dict['{}_{}_in_out_df'.format(model, hierarchy)]
            scr_df = get_scr_df(df)

            ax = perform_lstm_forecast(scr_df, predict_n=3, look_back=4, hierarchy=hierarchy)
            fig_name = 'SCR_LSTMpred_{}.png'.format(hierarchy)
            plt.savefig(os.path.join('plots',fig_name))


if __name__ == '__main__':

    to_create_paths = ['plots']
    for my_path in to_create_paths:
        if not os.path.exists(my_path):
            # Create the directory
            os.makedirs(my_path)

    with open(os.path.join('pickles','all_results.pkl'), 'rb') as f:
        result_dict = pickle.load(f)

    # perform the LSTM analysis to predict the SCR and make figures of the results
    create_lst_output_figures(result_dict)


    scr_dict = create_scr_figures(result_dict)
    create_summary_figure(scr_dict)


import os
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import surpyval as surv

line = ['Europe_product_line_food_preservation.csv','Europe_product_line_fabric_care.csv']
group = ['europe_prod_group_freestanding_refrigerator.csv']
subgroup = ['europe_prod_subgroup_7219.csv']
file_mapping = {'line': line,'group': group,'subgroup': subgroup}


def get_xcn(df, batch_id, **kwargs):
    n_drop = kwargs.pop('n_drop', 2)
    n_know = kwargs.pop('n_know', 12)

    cutoff = len(df.iloc[batch_id, n_drop:n_drop+n_know])

    n = list(df.iloc[batch_id, n_drop:n_drop+n_know])
    n.append(df.iloc[batch_id,1]-sum(df.iloc[batch_id, n_drop:n_drop+n_know]))

    x = [[i, i + 1] for i in np.arange(0, cutoff)]
    x.append(cutoff)

    c = [2 for i in np.arange(0, cutoff)]
    c.append(1) # adding right censor indicator

    return x, c, n, cutoff


def get_avg_predictions(df, batch_id, warranty_period, n_drop):
    if batch_id >=12:
        train_series = df.iloc[batch_id - warranty_period:batch_id, 1:].transpose().mean(axis=1)
        predictions = train_series[1:] / train_series[0] * df.iloc[batch_id, n_drop - 1]
    else:
        predictions = [np.nan for i in range(12)]

    return predictions


def get_naive_predictions(df, batch_id, warranty_period, n_drop,**kwargs):

    # if now argument is given the actual qty_sold will be used for the normalization of the naive model
    qty_sold_key = kwargs.pop('qty_sold_key', 'qty_sold')

    if batch_id >= 12:
        predictions = df.iloc[batch_id - warranty_period, n_drop:] / \
                      df.iloc[batch_id - warranty_period][qty_sold_key] * \
                      df.iloc[batch_id][qty_sold_key]
    else:
        predictions = [np.nan for i in range(12)]

    return predictions


def predict(model, cutoff, scaling, **kwargs):

    n_pred_sc  = np.round(np.ravel(
            [(model.ff(i+1) - model.ff(i))*scaling for i in range(0,cutoff+1)]
        )).astype(int)
    print('Forecast: {}'.format(n_pred_sc))
    return n_pred_sc


def create_sc_accuracy_figure(**kwargs):

    example_ape = {'know_4': [105.33333333333333, 76.05985037406484, 111.21495327102804],
                   'know_5': [89.7590361445783, 101.70068027210884, 47.07446808510639],
                   'know_6': [78.1456953642384, 44.95677233429395, 30.666666666666664],
                   'know_7': [43.333333333333336, 21.390374331550802, 29.022988505747126],
                   'know_8': [23.076923076923077, 17.12707182320442, 11.200000000000001],
                   'know_9': [12.18836565096953, 2.67639902676399, 6.720430107526881],
                   'know_10': [0.2590673575129534, 2.3746701846965697, 8.547008547008547],
                   'know_11': [9.685230024213075, 4.810126582278481, 2.2284122562674096],
                   'date': ['201609', '201610', '201611']}
    ape_know = kwargs.pop('ape_know', example_ape)
    know_min = kwargs.pop('know_min', 4)
    method = kwargs.pop('method', 'weibull')
    hierarchy = kwargs.pop('hierarchy', 'line')


    nrows = len([key for key in ape_know.keys() if 'know_' in key])

    if nrows == 0:
        print("Nothing to plot. Exit")
        return

    # Create a figure and gridspec layout
    fig = plt.figure(figsize=(16, 12))  # Adjust the figure size as needed
    gs = gridspec.GridSpec(nrows, 5, width_ratios=[4, 4, 4, 4, 1])
    date = ape_know['date']
    my_means = []
    my_stds = []

    for i in range(nrows):
        my_mean = np.nanmean(ape_know['know_{}'.format(know_min + i)])
        my_std = np.nanstd(ape_know['know_{}'.format(know_min + i)])
        ax1 = plt.subplot(gs[i, :4])
        if i == 0:
            ax1.set_title('{} - {}:\n Use x SCAs --> forecast (x+1)th month'.format(hierarchy,method))
        ax1.plot(date, ape_know['know_{}'.format(know_min + i)], marker='o',
                 label='know {:02} ({:.0f},{:.0f})'.format(know_min + i, my_mean, my_std))
        if method in ['weibull', 'gamma']:
            ax1.set_ylim([0, 200])
        else:
            ax1.set_ylim([0, 100])
        ax1.set_ylabel('APE')
        ax1.legend(loc='upper left')

        ax2 = plt.subplot(gs[i, 4], sharey=ax1)
        ax2.errorbar(0.5, my_mean, yerr=my_std, marker='o', color='k')
        my_means.append(my_mean)
        my_stds.append(my_std)


    plt.subplots_adjust(hspace=0, wspace=0)
    ax1.set_xticklabels(date, rotation=90)
    ax2.set_xticklabels([])
    plt.savefig(os.path.join('plots','SCF_Accuracy_{}_{}.png'.format(method, hierarchy)))
    plt.close()

    fig, ax = plt.subplots(1,1, figsize=(10,8))
    ax.errorbar([x+know_min for x in range(nrows)], my_means, yerr=my_stds, marker='o', color='k')
    ax.set_xlabel('x months')
    ax.set_ylabel('MAPE')
    ax.set_ylim([0,100])
    ax.set_title('{} - {}:\n Use x SCAs --> forecast (x+1)th month'.format(method, hierarchy))
    plt.grid()
    plt.savefig(os.path.join('plots','SCF_Accuracy_{}_{}_overall.png'.format(method, hierarchy)))
    plt.close()

def find_all_files_with(my_path, contains, dontcontain):

    # Specify the directory path where you want to search for files


    # Specify the keyword you want to search for in file names


    # List all files in the directory
    all_files = os.listdir(my_path)

    # Filter files that contain the keyword in their names
    matching_files = [file[9:-4] for file in all_files if contains in file and dontcontain not in file]

    return matching_files



if __name__ == '__main__':

    # To meassure how long the script runs:
    start_time = time.time()

    warranty_period = 12
    n_drop = 2
    dic_exp = {}
    qty_sold_key = 'qty_sold'

    #all_files = find_all_files_with('data', 'fabriccare','_all')
    #all_files = find_all_files_with('data', 'foodpreparation','_all')
    all_files = find_all_files_with('data', 'foodpreservation','_all')
    all_files = ['foodpreservation_FreestandingUnderCounter_FreestandingUnderCounterRefrigerator']
    for hierarchy in all_files:
        pickle_name = os.path.join('pickles','scaf_{}.pkl'.format(hierarchy))
        if os.path.exists(pickle_name):
            print('Already processed. {}'.format(pickle_name))
            break

        df = pd.read_csv(os.path.join('data','sc_table_{}.csv'.format(hierarchy)))
        df = df.rename(columns={'period': 'date'})
        df = df.sort_values(by='date')

        for method in ['weibull', 'gamma', 'lfp', 'naive', 'avg',]:


            # creation of df that will be exported into a pickle
            in_out_df = pd.DataFrame(
                columns=['date', qty_sold_key,
                         'sca_m0','sca_m1', 'sca_m2', 'sca_m3', 'sca_m4', 'sca_m5',
                         'sca_m6', 'sca_m7', 'sca_m8', 'sca_m9', 'sca_m10', 'sca_m11', 'sca_m12',
                         'scf_m0', 'scf_m1', 'scf_m2', 'scf_m3', 'scf_m4', 'scf_m5',
                         'scf_m6', 'scf_m7', 'scf_m8', 'scf_m9', 'scf_m10', 'scf_m11', 'scf_m12'])

            if method in ['naive', 'avg']:
                ape_know = {}
                for batch_id in range(len(df)):
                    date = df.iloc[batch_id]['date']
                    qty_sold = df.iloc[batch_id][qty_sold_key]
                    if method == 'avg':
                        predictions = get_avg_predictions(df, batch_id, warranty_period, n_drop)
                    if method == 'naive':
                        predictions = get_naive_predictions(df, batch_id, warranty_period, n_drop, qty_sold_key='qty_sold')

                    # Create a dictionary mapping column names to values
                    update_dict = {'date': date, qty_sold_key: qty_sold}
                    update_dict.update({f'sca_m{i}': val for i, val in enumerate(df.iloc[batch_id, n_drop:])})
                    update_dict.update({f'scf_m{i}': val for i, val in enumerate(predictions)})
                    # Update the DataFrame with the values from the dictionary
                    in_out_df.loc[batch_id, list(update_dict.keys())] = list(update_dict.values())
            else:
                # assume that sc is know for "know" number of months, learn a model and predict the know+1 month sc
                ape_know = {}
                know_min = 4

                for know in range(know_min, warranty_period+1):
                    print('\nAnalysing with know={}\n*******************************************'.format(know))
                    apes = []
                    for batch_id in range(len(df)):
                        date = df.iloc[batch_id]['date']
                        qty_sold = df.iloc[batch_id][qty_sold_key]
                        print('Batch{:02}: {}\nInput:    {}'.format(batch_id, df.iloc[batch_id]['date'],
                                                         df.iloc[batch_id,n_drop:].values))
                        if method in ['gamma', 'weibull', 'lfp']:
                            x,c,n, cutoff = get_xcn(df, batch_id, n_know=know, n_drop=n_drop)
                            print('Actua in: {}'.format(n))
                            if method == 'weibull':
                                model  = surv.Weibull.fit(x=x, c=c, n=n, how='MSE', offset=False)
                            elif method == 'gamma':
                                model = surv.Gamma.fit(x=x, c=c, n=n, how='MSE', offset=False)
                            elif method == 'lfp':
                                model = surv.Weibull.fit(x=x, c=c, n=n, lfp=True)
                            predictions = predict(model, cutoff, sum(n))
                            prediction = predictions[-1]
                            # Create a dictionary mapping column names to values
                            update_dict = {'date': date, qty_sold_key: qty_sold}
                            update_dict.update({f'sca_m{i}': val for i, val in enumerate(df.iloc[batch_id,n_drop:])})
                            update_dict.update({f'scf_m{know}': prediction})
                            # add also the first four forecasts using from the naive method
                            if know == warranty_period:
                                # use the naive model predictions for the first 4 terms of the SCR
                                predictions_naive = get_naive_predictions(df, batch_id, warranty_period, n_drop)
                                if isinstance(predictions_naive, pd.Series):
                                    update_dict.update({f'scf_m{i}': val for i, val in enumerate(predictions_naive[:4])})
                                else:
                                    update_dict.update({f'scf_m{i}': val for i, val in enumerate(predictions[:4])})
                        # Update the DataFrame with the values from the dictionary
                        in_out_df.loc[batch_id, list(update_dict.keys())] = list(update_dict.values())
                        actual = df.iloc[batch_id,n_drop+know]
                        ape = abs(actual-prediction)/actual*100
                        print('actual={}, prediction={}, ape={:.2f}'.format(actual, prediction,ape))
                        apes.append(ape)
                    ape_know.update({'know_{}'.format(know): apes})
                ape_know.update({'date': list(df['date'])})

            # adding ape_know for the naive and avg method
            if ape_know == {}:
                ape_know.update({'date': in_out_df['date']})
                for i in range(4,12):
                    key = f'know_{i}'
                    nominator = np.abs(in_out_df[f'sca_m{i}'] - in_out_df[f'scf_m{i}']).astype(float)
                    denominator = in_out_df[f'sca_m{i}'].astype(float)
                    value = np.where(denominator != 0, (nominator / denominator) * 100, 0)
                    ape_know.update({key: value})
            dic_exp.update({'{}_{}_ape_know'.format(method,hierarchy): ape_know})
            dic_exp.update({'{}_{}_in_out_df'.format(method,hierarchy): in_out_df})
                # save dict into pickle

        pickle_name = os.path.join('pickles','scaf_{}.pkl'.format(hierarchy))
        f = open(pickle_name, 'wb')
        pickle.dump(dic_exp,f)
    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Script execution time: {elapsed_time} seconds")

    #for hierarchy in ['line', 'group', 'subgroup']:
    #    for method in  ['gamma', 'lfp', 'avg', 'naive', 'weibull']:
    #        print('{}_{}'.format(hierarchy, method))
    #        ape_know = dic_exp['{}_{}_ape_know'.format(method,hierarchy)]
    #        create_sc_accuracy_figure( ape_know=ape_know, know_min=4, method=method, hierarchy=hierarchy)







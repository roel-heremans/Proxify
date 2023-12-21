import os
import pandas as pd
import numpy as np
from surpyval.regression import WeibullPH, ExponentialPH, CoxPH
import surpyval as surv
import matplotlib.pyplot as plt

# This code was part of the trial to make surpyval work for the regression case. I did not manage the full process and got abandoned due to the Markov Chain regression.
# See python script "10b_regression_MarkovChain.py" and notebook "10b_run_script" which runs in databricks. 

def regression_example():
    from surpyval.datasets import Tires
    from surpyval import CoxPH

    ''' Tires.data: shape = (34, 9)
        Columns: ['Tire age', 'Wedge gauge', 'Interbelt gauge', 'EB2B', 'Peel force',
                  'Carbon black (%)', 'Wedge gauge×peel force', 'Survival', 'Censoring']
        Rows: 0-33
        
        Tire age  Wedge gauge  ...  Survival  Censoring
    0       1.22         0.81  ...      1.02          1
    1       1.19         0.69  ...      1.05          0
    2       0.93         0.77  ...      1.22          1
    3       0.85         0.80  ...      1.17          0
    4       0.85         0.85  ...      1.09          1
    5       0.91         0.89  ...      1.09          0
    6       0.93         0.98  ...      1.17          0
    7       1.10         0.76  ...      1.10          1
    '''

    x = Tires.data['Survival']
    c = Tires.data['Censoring']
    Z = Tires.data[['Tire age', 'Wedge gauge', 'Interbelt gauge', 'EB2B', 'Peel force',
        'Carbon black (%)', 'Wedge gauge×peel force']]
    model = CoxPH.fit(x=x, Z=Z, c=c)

    Z = Tires.data[['Wedge gauge', 'Interbelt gauge', 'Peel force', 'Wedge gauge×peel force']]
    model_CoxPH = CoxPH.fit(x=x, Z=Z, c=c)

    model_WeibullPH = WeibullPH.fit(x=x, Z=Z, c=c)
    print(model_CoxPH)
    print(model_CoxPH.p_values)

    return model_CoxPH, model_WeibullPH

def get_xcn(Z):

    # since Z contains the qty_sold we need to subtract 1
    cutoff = Z.shape[1]-1
    Z_n = []

    # c and x do not change
    c = [2 for i in np.arange(0, cutoff)]
    c.append(1)

    x = [[i, i + 1] for i in np.arange(0, cutoff)]
    x.append(cutoff)

    for reg_i in range(Z.shape[0]):

        n = list(Z.iloc[reg_i,1:])
        n.append(Z.iloc[reg_i,0]-sum(n))

        Z_n.append(n)


    return x, c, Z_n, cutoff


def weibull_cdf(x, **kwargs):

    alpha = kwargs.pop('shape', 1)
    beta = kwargs.pop('scale', 1)

    cdf_weibull = 1 - np.exp(-(np.array(x)/alpha)**beta)

    return cdf_weibull

def predict(model, cutoff):

    cdf_weibull = weibull_cdf(range(cutoff+1), shape=model.alpha, scale=model.beta)
    scaling = sum(model.data['n'])
    n_pred_sc  = np.round(np.ravel([(cdf_weibull[i+1] - cdf_weibull[i])*scaling for i in range(0,cutoff)])).astype(int)
    print('Forecast: {}'.format(n_pred_sc))
    return n_pred_sc

def predict_regression(model, cutoff):

    alpha = model.params[0]
    beta = model.params[1]
    beta_i = model.params[2:]

    cdf_weibull = weibull_cdf(range(cutoff+1), shape=alpha, scale=beta)
    scaling = sum(model.data['n'])
    n_pred_sc  = np.round(np.ravel(
            [(cdf_weibull[i+1] - cdf_weibull[i])*scaling for i in range(0,cutoff)]
        )).astype(int)
    print('Forecast: {}'.format(n_pred_sc))
    return n_pred_sc

def transform_xcn_to_xc_only(**kwargs):
    '''For the regression we need to input the x and c without the n, hence this transformation'''
    x = kwargs.pop('x', [[0,1],[1,2], 3,  2, 5] )
    c = kwargs.pop('c', [2    ,2    , 0, -1, 1] )
    n = kwargs.pop('n', [4    ,2    , 3,  5, 6] )

    if len(x) == len(n) == len(c):
        xcn_equal_lenghts = 1
    else:
        print("Wrong input. Variables x and n need to have the same length")

    xonly = []
    conly = []
    for i, ni in enumerate(n):
        ci = c[i]
        xi = x[i]
        for j in range(ni):
            xonly.append(xi)
            conly.append(ci)

    return xonly, conly


def organize_data(hierarchy):
    from utils.util import load_service_call_data

    ''' extract train and test data for hierarchy in ['line', 'group', 'subgroup']
     and concat both and rearrange so that it can be consumed for the regression analysis.
     the new dataframe looks like:
                  201705  201706  201707  201708  ...  201903  201904  201905  201906
        qty_sold  223191  239887  228594  219271  ...  172637  175147  198479  196969
        m1          1251    1394    1453    1388  ...     695     747     855    1038
        m2          1182    1255    1305    1169  ...     675     696     818     985
        m3           773     855     850     698  ...     395     540     570     654
        m4           602     658     590     471  ...     384     417     466     480
        m5           539     496     398     412  ...     352     380     403     313
        m6           415     383     366     359  ...     322     360     287     247
        m7           322     350     324     247  ...     266     243     258     247
        m8           335     313     289     272  ...     246     197     238     231
        m9           291     270     321     309  ...     187     187     201     162
        m10          259     274     319     346  ...     167     177     169     147
        m11          274     339     357     408  ...     159     146     110     153
        m12          344     381     448     377  ...     144     113     160     227
        m13          202     239     222     183  ...      37      71      98     144
     '''

    train_data, test_data = load_service_call_data(hierarchy)

    # transpose the train and test data and merge them in one table
    train_data.set_index('date', inplace=True)
    t_train_df = train_data.T.rename_axis(None, axis=1)

    test_data.set_index('date', inplace=True)
    t_test_df = test_data.T.rename_axis(None, axis=1)

    df = pd.concat([t_train_df, t_test_df], axis=1)

    return df

def plot_reggression_direct_comparison(date, alpha_dir, beta_dir, alpha_reg, beta_reg, nr_covariates):

    formatted_dates = [str(date_i) for date_i in date]
    formatted_dates = [f"{date[:4]}{date[4:]}" for date in formatted_dates]
    x = np.arange(0, len(formatted_dates), 5)  # Set ticks every 5 data points

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(formatted_dates, alpha_dir, label='Direct')
    ax1.plot(formatted_dates, alpha_reg, label='Regression')
    ax1.set_xticks(x)
    ax1.set_xticklabels(formatted_dates[::5], rotation=90)

    ax2.plot(formatted_dates, beta_dir, label='Direct')
    ax2.plot(formatted_dates, beta_reg, label='Regression')
    ax2.set_xticks(x)
    ax2.set_xticklabels(formatted_dates[::5], rotation=90)

    ax1.set_title('Weibull: alpha')
    ax1.set_xlabel('Batch id')
    ax1.legend()
    ax1.grid()

    ax2.set_title('Weibull: beta')
    ax2.set_xlabel('Batch id')
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.savefig(os.path.join('plots','Parameters_regression_direct_foodpreservation_all_all_{}.png'.format(nr_covariates)))


if __name__ == '__main__':
    #from surpyval.regression import WeibullPH, ExponentialPH, CoxPH
#
    #df = organize_data('line')
    #df['censoring'] = 0
##
    ##df = df.T
##
    #x = df.iloc[1:13,12]
    #Z = df.iloc[1:13,0:12]
    #c = df.iloc[1:13]['censoring']
##
    ###x= df.iloc[12,1:13]
    ###Z= df.iloc[0:12, 1:13]
    ###c= df.iloc[-1,1:13]
##
    ### Weibull Proportional Hazard model
    #weibull_ph_model = WeibullPH.fit(x=x, Z=Z, c=c)
    ##print(weibull_ph_model)
#


    # Cox Proportional Hazard model
    #cox_ph_model = CoxPH.fit(x=x, Z=Z, c=c)

    # Exponential Proportional Hazard model
    #exponential_ph_model = ExponentialPH.fit(x=x, Z=Z, c=c)
    # print(exponential_ph_model)

    #model_CoxPH, model_WeibullPH = regression_example()
    #transform_xcn_to_xconly()
    #regression_example()

    hierarchy = 'foodpreparation_all_all'

    df2 = pd.read_csv(os.path.join('data','sc_table_{}.csv'.format(hierarchy)))
    df2 = df2.rename(columns={'period': 'date'})
    df2 = df2.sort_values(by='date')
    df2 = df2.T

    # renaming the columns by the yyyymm (which is the first row of the df), and removing that row from the df
    df2.columns = df2.iloc[0]
    df2 = df2[1:]


    offset = 0         # start at 201705
    reg_on = 80          # number of columns on which to do the regression
    sc_range = (1,14)   # range for the service calls m0, m1, ..., m12
    nr_covariates = 12

    # adding for each column the number of right censored service calls (= qty_sold -  sum( sc_i))
    #df2.loc['r_censor'] =df2.loc['qty_sold']-df2.iloc[sc_range[0]:sc_range[1], :].sum(axis=0)
    d2d = df2.iloc[:, offset:offset+reg_on]
    x, c, n, cutoff = get_xcn(d2d.T)

    x2 = x                  # df2.iloc[sc_range[0]:sc_range[1], offset+reg_on]
    Z2 = np.array(n).T        # df2.iloc[sc_range[0]:sc_range[1], offset:offset+reg_on]
    c2 = c

    remove_right_censored = 1
    if remove_right_censored:
        x2 = x[:-1]
        Z2 = Z2[:-1,:]
        c2 = c2[:-1]

    alpha_reg = []
    beta_reg = []
    alpha_dir = []
    beta_dir = []
    date = []

    for s in range(Z2.shape[1]-(nr_covariates+1)):
        print(s)
        model_reg = WeibullPH.fit(x=x2, Z=Z2[:,s:s+nr_covariates], c=c2, n=Z2[:,s+nr_covariates+1])
        model_ic  = surv.Weibull.fit(x=x2, c=c2, n=Z2[:,s+nr_covariates+1].flatten(), how='MSE', offset=False)
        alpha_dir.append(model_ic.params[0])
        beta_dir.append(model_ic.params[1])
        alpha_reg.append(model_reg.params[0])
        beta_reg.append(model_reg.params[1])
        date.append(d2d.columns[s+nr_covariates+1])
    plot_reggression_direct_comparison(date, alpha_dir, beta_dir, alpha_reg, beta_reg, nr_covariates)
    model_reg.model.ff(range(0,13), Z2[:,s:s+nr_covariates], *model_reg.params)

    Z1 = Z2[:,0:1].flatten()
    x_only, c_only = transform_xcn_to_xc_only(x=x2,c=c2,n=Z1)

    model_ic  = surv.Weibull.fit(x=x2, c=c2, n=Z1, how='MSE', offset=False)
    print(model_ic)
    model_ic  = surv.Weibull.fit(x=x_only, c=c_only, how='MSE', offset=False)
    print(model_ic)

    weibull_ph_model2.model.ff(range(0,13), Z2, *weibull_ph_model2.params)
    #model_ic  = surv.Weibull.fit(x=x[:-1], c=c[:-1], n=n[:-1], how='MSE', offset=False)
    #n_pred = predict(model_ic, cutoff)
    predict_regression(model_ic, cutoff)


    a=1



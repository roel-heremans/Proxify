# Databricks notebook source
# MAGIC %md
# MAGIC ## Required package
# MAGIC - Surpyval
# MAGIC - mlflow
# MAGIC

# COMMAND ----------

# MAGIC %pip install surpyval
# MAGIC %pip install mlflow

# COMMAND ----------

import os, pickle, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import surpyval as surv

import mlflow
import mlflow.sklearn

# COMMAND ----------

# MAGIC %md
# MAGIC ## Required Utils, created for this analysis, are getting imported:
# MAGIC - load_service_call_data

# COMMAND ----------

def load_service_call_data(*args):
# Product LINE
    if args[0] == 'line':
        food_preservation_train_data = pd.DataFrame(
            columns=['date','qty_sold','m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12','m13'])
        food_preservation_train_data['date']= ['201705','201706','201707','201708','201709','201710','201711','201712',
                                               '201801','201802','201803','201804','201805']
        food_preservation_train_data['qty_sold']=[ 223191,239887,228594,219271,223009,221593,218133,158218,172558,
                                                   162798,179055,179193,216338]
        food_preservation_train_data['m1']= [1251,1394,1453,1388,1379,1216,1024,910,894,951,841,910,963]
        food_preservation_train_data['m2']= [1182,1255,1305,1169,1151,996,975,817,777,791,795,868,1030]
        food_preservation_train_data['m3']= [773,855,850,698,556,618,602,508,462,556,563,610,696]
        food_preservation_train_data['m4']= [ 602,658,590,471,501,447,400,388,365,444,458,521,544]
        food_preservation_train_data['m5']= [ 539,496,398,412,366,368,355,316,419,432,407,425,506]
        food_preservation_train_data['m6']= [ 415,383,366,359,335,331,393,405,339,413,378,387,364]
        food_preservation_train_data['m7']= [ 322,350,324,247,294,364,386,376,346,326,374,312,272]
        food_preservation_train_data['m8']= [ 335,313,289,272,323,406,371,380,360,364,330,232,274]
        food_preservation_train_data['m9']= [ 291,270,321,309,341,376,384,335,323,283,194,228,257]
        food_preservation_train_data['m10']= [ 259,274,319,346,381,416,325,351,243,213,197,199,206]
        food_preservation_train_data['m11']= [ 274,339,357,408,434,416,377,278,214,222,222,212,211]
        food_preservation_train_data['m12']= [ 344,381,448,377,383,398,330,237,232,193,209,182,267]
        food_preservation_train_data['m13']= [ 202,239,222,183,197,177,109,135,87,113,90,118,149]

        food_preservation_test_data = pd.DataFrame(
            columns=['date','qty_sold','m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12','m13'])
        food_preservation_test_data['date'] = ['201806','201807','201808','201809','201810','201811','201812','201901',
                                               '201902','201903','201904','201905','201906']
        food_preservation_test_data['qty_sold'] = [240976,227943,237674,209602,224913,214440,145946,179610,164794,
                                                   172637,175147,198479,196969]
        food_preservation_test_data['m1'] = [1155,1379,1317,1280,1061,864,860,754,731,695,747,855,1038]
        food_preservation_test_data['m2'] = [1174,1114,1066,923,784,791,692,640,620,675,696,818,985]
        food_preservation_test_data['m3'] = [753,708,613,486,578,464,391,357,428,395,540,570,654]
        food_preservation_test_data['m4'] = [544,477,331,370,456,333,257,333,300,384,417,466,480]
        food_preservation_test_data['m5'] = [404,322,306,285,293,244,292,235,355,352,380,403,313]
        food_preservation_test_data['m6'] = [285,317,267,281,253,282,269,275,309,322,360,287,247]
        food_preservation_test_data['m7'] = [283,303,242,209,313,278,284,266,306,266,243,258,247]
        food_preservation_test_data['m8'] = [274,254,203,280,291,289,287,250,281,246,197,238,231]
        food_preservation_test_data['m9'] = [278,208,222,257,321,337,288,264,201,187,187,201,162]
        food_preservation_test_data['m10'] = [221,256,267,276,345,306,261,203,192,167,177,169,147]
        food_preservation_test_data['m11'] = [246,257,311,336,339,304,219,183,184,159,146,110,153]
        food_preservation_test_data['m12'] = [266,370,333,327,312,245,171,186,190,144,113,160,227]
        food_preservation_test_data['m13'] = [195,167,155,176,146,123,107,72,65,37,71,98,144]
    elif args[0] == 'group':
        food_preservation_train_data = pd.DataFrame(
            columns=['date', 'qty_sold', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12',
                     'm13'])
        food_preservation_train_data['date']= ['201707','201708','201709','201710','201711','201712','201801','201802','201803','201804','201805','201806','201807']
        food_preservation_train_data['qty_sold']=[23729,25981,20530,20879,20653,18397,17832,15323,16650,18989,23677,25173,26484]
        food_preservation_train_data['m1']= [234,204,181,156,147,107,142,113,97,136,127,185,223]
        food_preservation_train_data['m2']= [146,132,139,122,122,105,90,88,87,92,121,160,131]
        food_preservation_train_data['m3']= [111,70,63,70,62,54,60,62,47,53,91,81,90]
        food_preservation_train_data['m4']= [53,56,47,64,39,46,31,50,50,66,64,58,52]
        food_preservation_train_data['m5']= [48,63,56,47,42,36,41,46,69,48,52,59,29]
        food_preservation_train_data['m6']= [53,48,32,39,48,41,38,40,41,56,44,23,29]
        food_preservation_train_data['m7']= [43,35,38,47,33,37,36,42,39,30,30,23,28]
        food_preservation_train_data['m8']= [31,40,29,43,51,53,50,42,39,22,36,21,30]
        food_preservation_train_data['m9']= [46,32,36,40,55,40,36,25,31,26,27,36,32]
        food_preservation_train_data['m10']= [34,28,48,36,29,46,29,26,26,20,25,21,34]
        food_preservation_train_data['m11']= [36,44,53,38,40,27,28,17,28,24,18,29,29]
        food_preservation_train_data['m12']= [57,40,42,51,39,25,25,20,22,13,28,34,33]
        food_preservation_train_data['m13']= [28,23,23,17,9,16,9,9,17,15,18,19,9]

        food_preservation_test_data = pd.DataFrame(
            columns=['date','qty_sold','m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12','m13'])
        food_preservation_test_data['date'] = ['201806','201807','201808','201809','201810','201811','201812','201901',
                                               '201902','201903','201904','201905','201906']
        food_preservation_test_data['qty_sold'] = [25173,26484,27886,22984,24367,22148,17384,19826,15599,17373,17698,
                                                   21296,21159]
        food_preservation_test_data['m1'] = [185,223,214,176,132,106,132,106,89,83,94,115,145]
        food_preservation_test_data['m2'] = [160,131,128,100,94,102,68,76,62,72,82,95,137]
        food_preservation_test_data['m3'] = [81,90,61,63,53,47,44,39,34,26,58,79,89]
        food_preservation_test_data['m4'] = [58,52,43,42,83,41,30,35,28,29,33,50,44]
        food_preservation_test_data['m5'] = [59,29,35,24,23,20,27,19,38,40,42,47,31]
        food_preservation_test_data['m6'] = [23,29,23,23,14,30,36,24,41,32,36,28,26]
        food_preservation_test_data['m7'] = [23,28,25,24,40,23,38,31,23,23,24,23,19]
        food_preservation_test_data['m8'] = [21,30,22,17,23,23,23,26,25,28,16,28,22]
        food_preservation_test_data['m9'] = [36,32,22,24,24,42,38,26,30,17,20,16,16]
        food_preservation_test_data['m10'] = [21,34,28,24,31,36,30,22,16,13,23,17,13]
        food_preservation_test_data['m11'] = [29,29,31,42,28,36,24,22,15,15,18,20,12]
        food_preservation_test_data['m12'] = [34,33,41,35,25,38,14,25,24,20,8,14,23]
        food_preservation_test_data['m13'] = [19,20,20,14,14,16,14,10,4,5,7,14,19]
    elif args[0] == 'subgroup':
        food_preservation_train_data = pd.DataFrame(columns=['date','qty_sold','m1','m2','m3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12', 'm13'])
        food_preservation_train_data['date']= ['201708','201709','201710','201711','201712','201801','201802','201803','201804','201805','201806','201807','201808']
        food_preservation_train_data['qty_sold']=[25981,20530,20879,20653,18397,17832,15309,16625,18970,23637,25147,26435,27846]
        food_preservation_train_data['m1']= [203,181,156,147,107,142,113,97,136,126,184,222,213]
        food_preservation_train_data['m2']= [132,139,122,122,105,90,86,87,92,120,159,131,128]
        food_preservation_train_data['m3']= [70,63,70,62,54,60,62,47,53,90,81,90,61]
        food_preservation_train_data['m4']= [55,47,64,39,46,31,50,50,66,64,58,52,43]
        food_preservation_train_data['m5']= [63,56,47,42,36,41,46,69,48,52,59,28,35]
        food_preservation_train_data['m6']= [48,31,39,48,41,38,39,41,56,44,23,29,23]
        food_preservation_train_data['m7']= [35,38,47,33,37,36,42,39,30,29,23,28,24]
        food_preservation_train_data['m8']= [40,28,43,51,52,50,42,39,22,36,21,30,22]
        food_preservation_train_data['m9']= [32,36,40,55,40,36,25,31,26,27,36,32,22]
        food_preservation_train_data['m10']= [28,48,36,29,46,29,26,26,20,25,21,34,28]
        food_preservation_train_data['m11']= [44,53,38,40,27,28,17,27,24,18,29,28,31]
        food_preservation_train_data['m12']= [40,42,51,39,25,25,20,22,13,28,34,33,41]
        food_preservation_train_data['m13']= [22,23,17,9,16,9,9,17,15,18,19,20,20]

        food_preservation_test_data = pd.DataFrame(
            columns=['date','qty_sold','m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12','m13'])
        food_preservation_test_data['date'] = ['201806','201807','201808','201809','201810','201811','201812','201901',
                                               '201902','201903','201904','201905','201906']
        food_preservation_test_data['qty_sold'] = [25147,26435,27846,22954,24323,22121,17354,19793,15566,17334,17680,21264,21129]
        food_preservation_test_data['m1'] = [184,222,213,176,132,106,132,106,89,83,93,114,144]
        food_preservation_test_data['m2'] = [159,131,128,99,94,102,67,76,62,72,82,95,137]
        food_preservation_test_data['m3'] = [81,90,61,63,53,47,44,39,34,26,58,79,89]
        food_preservation_test_data['m4'] = [58,52,43,42,83,41,30,35,28,28,33,50,44]
        food_preservation_test_data['m5'] = [59,28,35,24,23,20,27,19,38,40,42,47,31]
        food_preservation_test_data['m6'] = [23,29,23,23,13,30,36,24,41,31,35,28,26]
        food_preservation_test_data['m7'] = [23,28,24,24,40,23,38,31,23,23,24,23,19]
        food_preservation_test_data['m8'] = [21,30,22,17,23,23,23,26,25,28,16,28,22]
        food_preservation_test_data['m9'] = [36,32,22,24,24,42,36,26,30,17,20,16,16]
        food_preservation_test_data['m10'] = [21,34,28,24,31,35,30,22,16,13,23,17,13]
        food_preservation_test_data['m11'] = [29,28,31,42,28,36,23,22,15,15,18,20,12]
        food_preservation_test_data['m12'] = [34,33,41,35,25,38,14,25,24,20,8,14,23]
        food_preservation_test_data['m13'] = [19,20,20,14,14,16,14,10,4,5,7,14,19]
    else:
        print("choose on of the following options: line, group or subgroup")
    return food_preservation_train_data, food_preservation_test_data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Flags and display settings and name-mappers
# MAGIC

# COMMAND ----------

# Flag that sets: Running with/without mlflow logging (1/0)
mlflow_flag = 1

# When displaying a float it will use the precision specified in float_formatter (here 2 decimal numbers)
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

map_name = {'line':1, 'group':2, 'subgroup':3}

# COMMAND ----------

def calculate_interval_n_right_censored_data(df, batch_index) -> pd.DataFrame:
    '''
    Returns x,c,n and cutoff (variables needed to do the surpyval analysis) for a given row in a service call dataframe.

    Assumption on the input service call dataframe, df:
    Each row in the dataframe represents the number of service calls received at the end
    of a certain month (interval censored). The (length of a row -2) == the number of
    months considered over which the service calls are observed. The minus 2 is needed
    because a row also contains in the first column the date and in the second column
    the quantity of the batch sold.
    First column: is the date of the batch
    Second column: the quantity sold for that batch_index
    Third column: m1, quantity of service calls after the end of the first month from the time of the sold batch
    Fourth column: m2, quantity of service calls after the first month and before the end of the second month
    Fifteenth column: m13, ...
    example:
    INPUT:
          date  qty_sold    m1    m2   m3   m4  ...   m8   m9  m10  m11  m12  m13
    0   201705    223191  1251  1182  773  602  ...  335  291  259  274  344  202
    1   201706    239887  1394  1255  855  658  ...  313  270  274  339  381  239
    ...
    OUTPUT:
    x, list of failure times
    c, list of censoring flags (-1 = left, 0 = failure/event, 1 = right or 2 = interval censoring)
       for interval censoring x must have the left and right value of the interval
    n, the count array associated with x
    x: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], 13]
    c: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
    n: [1251, 1182, 773, 602, 539, 415, 322, 335, 291, 259, 274, 344, 202, 216402]
    '''

    # number of initial columns to be dropped because they represent different information than qty of service calls
    n_drop = 2
    n=list(df.iloc[batch_index, n_drop:])

    # adding the right censored value (i.e. qty without service call)
    n.append(df.iloc[batch_index,1]-sum(df.iloc[batch_index, n_drop:]))

    cutoff = len(df.iloc[batch_index, n_drop:])

    x = [[i, i+1] for i in np.arange(0, cutoff)]
    x.append(cutoff)  # adding cutoff time

    c = [2 for i in np.arange(0, cutoff)]
    c.append(1) # adding right censor indicator

    return x, c, n, cutoff



# COMMAND ----------

def visualize_fit(sc_input, cutoff, fitted_model, batch_id, **kwargs):
    ''' This function can be used to show a single figure with the actual service calls versus the predicted service
    calls. But it can also be used to generate one of the subplots in a bigger figure like for instance the plots called
    FitQuality_{...}.png. To use the latter, you have to specify the axis in which the figure need to be shown. When the
    axes is not given as an argument of the visualize_fit function, a new standalone figure will be created.
    If 'N_batch' is given as a argument to the function, one assumes that the interval and right censored method is used.
    When 'N_batch' is not given as an argument to the function, one assumes that the interval censored method is used,
    without the right censoring datapoint.

    Example:
        INPUT:
        sc_input: service calls for batch_id expressing the service calls after one month, after 2 months, ...
                  [1251, 1182, 773, 602, 539, 415, 322, 335, 291, 259, 274, 344, 202, 216402]
        cutoff = 13
        fitted_model, the output from surpyval, e.g.:
                Parametric SurPyval Model
                =========================
                Distribution        : Weibull
                Fitted by           : MSE
                Parameters          :
                     alpha: 4.767167533164514
                      beta: 1.0365673754443328
        batch_index: row in the bigger service call dataframe for which the service calls are considered
        model: 'weibull', 'gamma', 'mm2w', 'lfp' or 'tb'
    '''

    if 'axes' in kwargs:
        ax = kwargs['axes']
    else:
        fig, ax = plt.subplots()

    txt = kwargs.pop('put_text', '')
    model = kwargs.pop('model', '')

    # since each model, has a different output format for the figure title, we are collecting the text per model
    if model == 'gamma':
        ax_txt = "(shape, scale) = ({:.2f}, {:.2f}\n(MAPE, MdAPE) = ({})\nBatchId = {}".format(fitted_model.alpha,1/fitted_model.beta,txt,batch_id)
    elif model == 'weibull':
        ax_txt = "(shape, scale) = ({:.2f}, {:.2f}\n(MAPE, MdAPE) = ({})\nBatchId = {}".format(fitted_model.beta,fitted_model.alpha,txt,batch_id)
    elif model == 'mm2w':
        ax_txt ='(MAPE,MdAPE)=({})\nBatchId={}'.format(txt, batch_id)
    elif model == 'lfp':
        ax_txt ='(MAPE,MdAPE)=({})\nBatchId={}'.format(txt, batch_id)
    elif model == 'tb':
        ax_txt = '(MAPE,MdAPE)=({})\nBatchId={}'.format(txt, batch_id)
    else:
        print('Choose one of models: weibull, gamma, mm2w, lfp, tb')
        return

    # If this function is used with the keyword argument 'N_batch' then we are considering the ICR method instead of IC.
    # In this case we have one more bin, and we are using the number of sold items in that batch as the normalization
    # factor
    if 'N_batch' in kwargs:
        xvals = np.arange(1, cutoff+2)
        bincounts = sc_input
        scaling = kwargs['N_batch']
        #sets the max of the y-axis to the maximum in the service call vector
        max_y_lim = math.ceil(max(sc_input[:-1]) / 100) * 100
        ax.set_ylim([0,max_y_lim])
        ax_txt = 'Est SC IRC:\n' + ax_txt
    else:
        xvals = np.arange(1, cutoff+1)
        bincounts = sc_input[:-1]
        scaling = fitted_model.scaling
        ax_txt = 'Est SC IC:\n' + ax_txt

    ax.errorbar(xvals, bincounts,  marker='.', color='k',linestyle="None",label='Act SC')
    ax.errorbar(xvals, bincounts,  marker='+', color='k',linestyle="None")

    xnn= np.array(range(0,13))
    pdff_dots = np.ravel([(fitted_model.ff(i+1) - fitted_model.ff(i))*scaling for i in xnn])

    ax.plot(xnn+1, pdff_dots, color='r',marker='.',linestyle="None",label=ax_txt)
    ax.plot(xnn+1, pdff_dots, color='r',marker='+',linestyle="None")
    ax.set_ylabel('# Serv. Calls')
    ax.set_xlabel('Month ID')
    ax.legend()
    ax.grid()
    return ax



# COMMAND ----------

def visualize_ic_irc_error_matrix(mape_irc, mape_ic,  **kwargs):
    mlflow_flag = kwargs.pop('mlflow_flag', 0)
    metric = kwargs.pop('metric', 'MAPE')
    fig_name = kwargs.pop('fig_name', 'fig_name_placeholder.png')

    plt.plot(np.array(range(len(mape_irc)))+1,mape_irc,marker='o',label='Interval and Right censored')
    plt.plot(np.array(range(len(mape_irc)))+1,mape_ic,marker='o',label='Interval censored')
    plt.legend()
    plt.xlabel('Batch_id')
    plt.ylabel('{} [%]'.format(metric))
    plt.grid()
    plt.ylim([0,55])

    plt.savefig(fig_name)
    if mlflow_flag:
        mlflow.log_artifact(fig_name)
    plt.close()

# COMMAND ----------

def visualize_shape_and_scale_track(shape_irc, scale_irc, shape_ic, scale_ic, **kwargs):

    method = kwargs.pop('method', 'weibull')
    fig_name = kwargs.pop('fig_name', 'fig_name_placeholder.png')
    mlflow_flag = kwargs.pop('mlflow_flag', 0)

    if method in ['weibull', 'gamma']:
        fig, ax = plt.subplots(2, 2,figsize=(10,6))
        ax[0,0].plot(np.array(range(len(shape_irc)))+1, shape_irc, marker='o')
        ax[0,1].plot(np.array(range(len(scale_irc)))+1, scale_irc, marker='o')
        ax[1,0].plot(np.array(range(len(shape_ic)))+1, shape_ic, marker='o')
        ax[1,1].plot(np.array(range(len(scale_ic)))+1, scale_ic, marker='o')

        ax[1,0].set_xlabel('Batch_id')
        ax[1,1].set_xlabel('Batch_id')
        ax[0,0].set_ylabel('IRC')
        ax[0,0].set_title('{}: Shape'.format(method))
        ax[0,1].set_title('{}: Scale'.format(method))
        ax[1,0].set_ylabel('IC')
        ax[0,0].grid()
        ax[0,1].grid()
        ax[1,0].grid()
        ax[1,1].grid()

        plt.savefig(fig_name)
        if mlflow_flag:
            mlflow.log_artifact(fig_name)
        plt.close()

# COMMAND ----------

def visualize_ttfi(vol_diff, **kwargs):
    ''' This function can be used to show a single figure with the percentage error between the actual and the
    predicted number of servoce calls as a function of the number of months in the warranty period.
    But it can also be used to generate one of the subplots in a bigger figure like for instance the plots called
    FitQuality_{...}.png. To use the latter, you have to specify the axis in which the figure need to be shown. When the
    axes is not given as an argument of the visualize_ttfi function, a new standalone figure will be created.

    Example:
        INPUT:
        vol_diff:   percentage error between the actual and the predicted number of service calls as a function of the
                    number of months in the warranty period, e.g.
                    array([2.48, 11.42, -11.90, -17.44, -6.68, -12.29, -17.08, 9.25, 16.15,
                          23.94, 42.34, 63.37, 50.00])

    '''
    fig_path = 'plots'
    pos_vol_diff =  [val if val > 0 else 0 for val in vol_diff]
    neg_vol_diff =  [val if val < 0 else 0 for val in vol_diff]

    if 'save_name' in kwargs:
        fig_name = kwargs['save_name']
        fig, ax = plt.subplots()

    if 'axes' in kwargs:
        ax = kwargs['axes']

    ax.bar(range(1,len(vol_diff)+1),pos_vol_diff,  color='green')
    ax.bar(range(1,len(vol_diff)+1),neg_vol_diff,  color='red')
    method = kwargs.pop('method','parametric')
    ax.set_ylim(-80, 80)
    ax.set_ylabel('Volume diff (Act-Est)/Act [%]')
    ax.set_xlabel('Month ID')


    if 'save_name' in kwargs:
        plt.savefig(os.path.join(fig_path, fig_name), bbox_inches='tight')
        plt.close()
    return



# COMMAND ----------

def get_absolute_percentage_errors(actual_values, predicted_values, **kwargs):
    '''
    Return the mean or the median absolute percentage error.
    :param actual_values:
    :param predicted_values:
    :param kwargs:
    :return: either the mean or the median of the absolute percentage error
    '''
    pes = (actual_values - predicted_values) / actual_values * 100
    apes = abs(pes)

    # Calculate median absolute percentage error
    if kwargs['how'] == 'mean':
        return np.mean(apes), pes
    elif kwargs['how'] == 'median':
        return np.median(apes), pes
    else:
        print('Choose mean or median for the how value')
        return []



# COMMAND ----------

def get_sc_predictions(fitted_model, cutoff):
    '''This function returns the predicted number of service calls in a range from 0 till cutoff value.'''

    n_pred_sc = np.round(np.ravel(
            [(fitted_model.ff(i+1) - fitted_model.ff(i))*fitted_model.scaling for i in range(cutoff)]
        )).astype(int)
    return n_pred_sc

# COMMAND ----------

def get_sc_results(train_data, hierarchy, **kwargs):
    '''
    This function will generate the different plots as described in the description of the main function.
    As well as the pickle with the mape, and mdape values and the dataframe with actuals (sc_m) and predictions (scf_m).
    In the df the columns are ['date', 'qty_sold', 'sc_m1', 'sc_m2', 'sc_m3', 'sc_m4', 'sc_m5', 'sc_m6',
    'sc_m7', 'sc_m8', 'sc_m9', 'sc_m10', 'sc_m11', 'sc_m12', 'scf_m1', 'scf_m2', 'scf_m3', 'scf_m4', 'scf_m5',
    'scf_m6', 'scf_m7', 'scf_m8', 'scf_m9', 'scf_m10', 'scf_m11', 'scf_m12'].
     If the mlflow_flag is set to 1 it will aslo save the data, models, figures, etc..

     INPUT:
     'train_data':
           date  qty_sold    m1    m2   m3   m4  ...   m8   m9  m10  m11  m12  m13
     0   201705    223191  1251  1182  773  602  ...  335  291  259  274  344  202
     1   201706    239887  1394  1255  855  658  ...  313  270  274  339  381  239
     2   201707    228594  1453  1305  850  590  ...  289  321  319  357  448  222

     'hierarchy' can be one of the followings: {'line', 'group', 'subgroup'}

     'model' optional keyword argument which can be one of the following: {'weibull', 'gamma', 'mm2w', 'lfp', 'tb'}
             when no model keyword is given the default is taken which corresponds to 'weibull'
     OUTPUT:
     mape_irc = list of floats containing the values of the mean absolute percentage error using the IRC method
     mape_ic  = same but for IC method
     mdape_irc = list of floats containing the values of the median absolute percentage error using the IRC method
     mdape_ic = same but for IC method
     in_out_df = dataframe containing the actual service call data and the forcasted once.

     example:
     mape_irc = [29.6, 30.2, 31.3, 30.1, 26.0, 22.9, 28.2, 21.6, 30.6, 30.2, 37.1, 37.3]
     in_out_df =
              date qty_sold sc_m1 sc_m2 sc_m3  ... scf_m10 scf_m11 scf_m12 sc_m13 scf_m13
        0   201705   223191  1251  1182   773  ...     401     386     373  202.0   362.0
        1   201706   239887  1394  1255   855  ...     420     404     390  239.0   378.0
        2   201707   228594  1453  1305   850  ...     417     402     388  222.0   376.0
     '''

    model = kwargs.pop('model','weibull')
    fig_path = 'plots'

    # We will save per 4 sales batches per png and will call them
    # their name with an additional specifier a,b,c,...
    rows_per_pagesave = 4
    save_sub_plot_id = 'a'

    N_idx = train_data.shape[0]
    # For each batch the mape, mdape, shape and scale are stored for both methods irc and ic
    mape_irc = []
    mape_ic = []
    mdape_irc = []
    mdape_ic = []

    in_out_df = []
    #pd.DataFrame(columns=['date', 'qty_sold', 'sc_m1', 'sc_m2', 'sc_m3', 'sc_m4', 'sc_m5', 'sc_m6',
    #                                  'sc_m7', 'sc_m8', 'sc_m9', 'sc_m10', 'sc_m11', 'sc_m12',
    #                                  'scf_m1', 'scf_m2', 'scf_m3', 'scf_m4', 'scf_m5', 'scf_m6',
    #                                  'scf_m7', 'scf_m8', 'scf_m9', 'scf_m10', 'scf_m11', 'scf_m12'])

    if model in ['weibull', 'gamma']:
        shape_irc = []
        scale_irc = []
        shape_ic = []
        scale_ic = []

    Nb = len(train_data['qty_sold'])
    for batch_index, N in enumerate(train_data['qty_sold']):
        print('Batch: {}'.format(batch_index))
        if (batch_index % rows_per_pagesave) == 0:
            fig, ax = plt.subplots(rows_per_pagesave, 4, figsize=(35,70*rows_per_pagesave/Nb))

        x_train,c_train,n_train,cutoff = calculate_interval_n_right_censored_data(train_data, batch_index)

        if 'test_data' in kwargs:
            test_data = kwargs['test_data']
            x_test,c_test,n_test,_ = calculate_interval_n_right_censored_data(test_data, batch_index)
        else:
            n_test = n_train

        #fitting the respective models e.g. weibull, gamma, lfp, turnbull, ...
        if model == 'weibull':
            model_irc = surv.Weibull.fit(x=x_train, c=c_train, n=n_train, how='MSE', offset=False)
            model_ic  = surv.Weibull.fit(x=x_train[:-1], c=c_train[:-1], n=n_train[:-1], how='MSE', offset=False)
        elif model == 'gamma':
            model_irc = surv.Gamma.fit(x=x_train,      c=c_train,      n=n_train, how='MSE', offset=False)
            model_ic  = surv.Gamma.fit(x=x_train[:-1], c=c_train[:-1], n=n_train[:-1], how='MSE', offset=False)
        elif model == 'lfp':
            model_irc = surv.Weibull.fit(x=x_train,      c=c_train,      n=n_train, lfp=True)
            model_ic  = surv.Weibull.fit(x=x_train[:-1], c=c_train[:-1], n=n_train[:-1], lfp=True)
        elif model == 'mm2w':
            model_irc = surv.MixtureModel(x=x_train,      c=c_train,      n=n_train, dist=surv.Weibull, m=2)
            model_ic  = surv.MixtureModel(x=x_train[:-1], c=c_train[:-1], n=n_train[:-1], dist=surv.Weibull, m=2)
        elif model == 'tb':
            model_irc = surv.Turnbull.fit(x=x_train,      c=c_train,      n=n_train)
            model_ic  = surv.Turnbull.fit(x=x_train[:-1], c=c_train[:-1], n=n_train[:-1])

        model_irc.scaling = sum(n_train)
        model_ic.scaling = sum(n_train[:-1])

        # making predictions for ic and irc method
        n_pred_irc_sc = get_sc_predictions(model_irc, cutoff)
        n_pred_ic_sc  = get_sc_predictions(model_ic, cutoff)

        if 'test_data' in kwargs:
            date = test_data.iloc[batch_index]['date']
            qty_sold =  test_data.iloc[batch_index]['qty_sold']
        else:
            date = train_data.iloc[batch_index]['date']
            qty_sold =  train_data.iloc[batch_index]['qty_sold']

        row_data = { 'date': date, 'qty_sold': qty_sold,
                     **{f'sc_m{i}': val for i, val in enumerate(n_test[:-1], start=1)},
                     **{f'scf_m{i}': val for i, val in enumerate(n_pred_irc_sc, start=1)}
                     }
        in_out_df.append(row_data)

        # getting the MAPE for both methods
        err1_irc, vol_diff_irc = get_absolute_percentage_errors(np.array(n_test[:-1]), np.array(n_pred_irc_sc),how='mean')
        err1_ic,  vol_diff_ic  = get_absolute_percentage_errors(np.array(n_test[:-1]), np.array(n_pred_ic_sc), how='mean')
        err2_irc, _ = get_absolute_percentage_errors(np.array(n_test[:-1]), np.array(n_pred_irc_sc),how='median')
        err2_ic, _  = get_absolute_percentage_errors(np.array(n_test[:-1]), np.array(n_pred_ic_sc), how='median')


        # saving the fit result plots
        visualize_fit(n_test, cutoff, model_ic, batch_index, model=model, axes=ax[batch_index % rows_per_pagesave,0], put_text='{:.3f},{:.3f}'.format(err1_ic,err2_ic))
        visualize_ttfi(vol_diff_ic,axes=ax[batch_index % rows_per_pagesave,1],model=model)

        #fig_name = 'irc_batch{:02d}_mape{:.2f}_.png'.format(batch_index, err_irc)
        visualize_fit(n_test, cutoff, model_irc, batch_index, model=model, N_batch=N, axes=ax[batch_index % rows_per_pagesave,2], put_text='{:.3f},{:.3f}'.format(err1_irc,err2_irc))
        #fig_name = 'irc_batch{:02d}_ttf.png'.format(batch_index)
        visualize_ttfi(vol_diff_irc,axes=ax[batch_index % rows_per_pagesave,3],model=model)

        mape_irc.append(err1_irc)
        mape_ic.append(err1_ic)
        mdape_irc.append(err2_irc)
        mdape_ic.append(err2_ic)
        if model in ['weibull', 'gamma']:
            shape_irc.append(model_irc.beta)
            scale_irc.append(model_irc.alpha)
            shape_ic.append(model_ic.beta)
            scale_ic.append(model_ic.alpha)

        # figure showing 4xrows_per_pagesave subplots: (interval censored data and fit) with (volume diff) and  (right censored data
        # and fit) with (volume diff) as a function of each month of the reliability period
        if ((batch_index % rows_per_pagesave) == rows_per_pagesave-1) or (batch_index == Nb-1):
            fig_name = os.path.join(fig_path,'FitQuality_{}{:02d}{}_{}.png'.format(model,  map_name[hierarchy], save_sub_plot_id, hierarchy))
            fig.savefig(fig_name)
            plt.close()
            save_sub_plot_id = chr(ord(save_sub_plot_id)+1)

        if mlflow_flag:
            mlflow.sklearn.log_model(model_irc, 'weibull_irc_model_batch{}'.format(batch_index))
            mlflow.log_metric('MAPE_{}'.format(batch_index), err1_irc)
            mlflow.log_metric('MdAPE_{}'.format(batch_index), err2_irc)
            if ((batch_index % rows_per_pagesave) == rows_per_pagesave-1) or (batch_index == Nb-1):
                mlflow.log_artifact(fig_name)

    # MAPE: Comparison both methods (irc & ic)
    fig_name = os.path.join(fig_path,'MAPE_{}{:02d}_{}.png'.format(model, map_name[hierarchy],hierarchy))
    visualize_ic_irc_error_matrix(mape_irc, mape_ic, fig_name=fig_name, metric='MAPE', mlflow_flag=mlflow_flag)

    # MdAPE: Comparison both methods (irc & ic)
    fig_name = os.path.join(fig_path,'MdAPE_{}{:02d}_{}.png'.format(model, map_name[hierarchy],hierarchy))
    visualize_ic_irc_error_matrix(mdape_irc, mdape_ic, fig_name=fig_name, metric='MdAPE', mlflow_flag=mlflow_flag)

    # Tracking the variation on the shape and scale for both methods; (weibull and gamma)
    if model in ['weibull', 'gamma']:
        fig_name = os.path.join(fig_path,'Params_{}{:02d}_{}.png'.format(model, map_name[hierarchy],hierarchy))
        visualize_shape_and_scale_track(shape_irc, scale_irc, shape_ic, scale_ic, **kwargs)

    return mape_irc[:-1], mape_ic[:-1], mdape_irc[:-1], mdape_ic[:-1], pd.DataFrame(in_out_df)

# COMMAND ----------

def get_sc_naive_results(train_data, hierarchy, **kwargs):
    ''' Same function as the get_sc_results but doing it for the naive and avg model.
     We decided to create a separate function due to the fact that the data needs to be treated differently
     compared to the other models. Here the model will be either the average of the previous 12 months or the service
     calls of the same month of the previous year.
     The train_series for a given sold_batch yyyymm, takes the 12 months before, show as the following dataframe:

            qty_sold    m1    m2   m3   m4   m5   m6   m7   m8   m9  m10  m11  m12  m13
        0     223191  1251  1182  773  602  539  415  322  335  291  259  274  344  202
        1     239887  1394  1255  855  658  496  383  350  313  270  274  339  381  239
        2     228594  1453  1305  850  590  398  366  324  289  321  319  357  448  222
        3     219271  1388  1169  698  471  412  359  247  272  309  346  408  377  183
        4     223009  1379  1151  556  501  366  335  294  323  341  381  434  383  197
        5     221593  1216   996  618  447  368  331  364  406  376  416  416  398  177
        6     218133  1024   975  602  400  355  393  386  371  384  325  377  330  109
        7     158218   910   817  508  388  316  405  376  380  335  351  278  237  135
        8     172558   894   777  462  365  419  339  346  360  323  243  214  232   87
        9     162798   951   791  556  444  432  413  326  364  283  213  222  193  113
        10    179055   841   795  563  458  407  378  374  330  194  197  222  209   90
        11    179193   910   868  610  521  425  387  312  232  228  199  212  182  118

    for the avg model it takes the average over the rows for the naive model it takes just one row
    (corresponding to the same month but from the previous year).
    The average returns the following series:
        qty_sold        202125.000000
            m1            1134.250000
            m2            1006.750000
            m3             637.583333
            m4             487.083333
            m5             411.083333
            m6             375.333333
            m7             335.083333
            m8             331.250000
            m9             304.583333
            m10            293.583333
            m11            312.750000
            m12            309.500000
            m13            156.000000
        the naive returns the first row indexed 0 of the input df, which is:
                qty_sold    223191
                m1            1251
                m2            1182
                m3             773
                m4             602
                m5             539
                m6             415
                m7             322
                m8             335
                m9             291
                m10            259
                m11            274
                m12            344
                m13            202
     '''

    model = kwargs.pop('model','avg')   # (model = {'avg', 'naive'})
    training_window = 12
    nr_cross_val = 12
    rows_per_pagesave = 4
    save_sub_plot_id = 'a'
    mape  = []
    mdape = []

    in_out_df = []
    #pd.DataFrame(columns=['date', 'qty_sold', 'sc_m1', 'sc_m2', 'sc_m3', 'sc_m4', 'sc_m5', 'sc_m6',
    #                                  'sc_m7', 'sc_m8', 'sc_m9', 'sc_m10', 'sc_m11', 'sc_m12',
    #                                  'scf_m1', 'scf_m2', 'scf_m3', 'scf_m4', 'scf_m5', 'scf_m6',
    #                                  'scf_m7', 'scf_m8', 'scf_m9', 'scf_m10', 'scf_m11', 'scf_m12'])

    if 'test_data' in kwargs:
        test_data = kwargs['test_data']
        df = pd.concat([train_data, test_data], ignore_index=True)
    else:
        df = pd.concat([train_data, train_data], ignore_index=True)


    # average model:

    for batch_id in range(nr_cross_val):
        print('Model: {}, Batch Id: {}'.format(model, batch_id))
        if (batch_id % rows_per_pagesave) == 0:
            fig, ax = plt.subplots(rows_per_pagesave, 2, figsize=(35,70*rows_per_pagesave/nr_cross_val))

        if model == 'avg':
            train_series = df.iloc[batch_id:batch_id+training_window,1:].transpose().mean(axis=1)
        if model == 'naive':
            train_series = df.iloc[batch_id,1:].transpose()

        test_series = df.iloc[batch_id+training_window+1,1:]

        # saving the service call prediction and the actual in df
        date = df.iloc[batch_id + training_window + 1]['date']
        qty_sold = df.iloc[batch_id + training_window + 1]['qty_sold']

        row_data = {'date': date, 'qty_sold': qty_sold,
                    **{f'sc_m{i}': val for i, val in enumerate(train_series[1:-1], start=1)},
                    **{f'scf_m{i}': val for i, val in enumerate(test_series[1:-1], start=1)}
                    }
        in_out_df.append(row_data)

        train_series = train_series[1:]/train_series[0]
        test_series = test_series[1:]/test_series[0]
        vol_diff = (test_series - train_series) / test_series * 100

        positive_volume_diff = [val if val >= 0 else 0 for val in vol_diff]
        negative_volume_diff = [val if val < 0 else 0 for val in vol_diff]
        mape1, _ = get_absolute_percentage_errors(test_series, train_series,how='mean')
        mdape1, _ = get_absolute_percentage_errors(test_series, train_series,how='median')
        mape.append(mape1)
        mdape.append(mdape1)

        ax[batch_id % rows_per_pagesave, 0].plot(test_series.index, test_series, marker='.', color='k',linestyle="None",label="Actual Normalized SC")
        ax[batch_id % rows_per_pagesave, 0].plot(test_series.index, test_series, marker='+', color='k',linestyle="None")
        ax[batch_id % rows_per_pagesave, 0].plot(test_series.index, train_series,
                   color='r',
                   label="Pred Normalized SC\n(MAPE, MdAPE)=({:.2f},{:.2f})\nBatchId: {}".format(mape1,mdape1,batch_id))
        ax[batch_id % rows_per_pagesave, 0].set_xlabel("ttfi")
        ax[batch_id % rows_per_pagesave, 0].set_ylabel("Normalized <service calls>")
        ax[batch_id % rows_per_pagesave, 0].legend()
        ax[batch_id % rows_per_pagesave, 0].grid()

        ax[batch_id % rows_per_pagesave, 1].bar(test_series.index, positive_volume_diff, color='green')
        ax[batch_id % rows_per_pagesave, 1].bar(test_series.index, negative_volume_diff, color='red')
        ax[batch_id % rows_per_pagesave, 1].set_ylim(-max(abs(vol_diff))*1.05, max(abs(vol_diff))*1.05)
        ax[batch_id % rows_per_pagesave, 1].set_xlabel('Volume_diff_%')
        ax[batch_id % rows_per_pagesave, 1].set_ylabel('ttfi')
        ax[batch_id % rows_per_pagesave, 1].grid()

        if ((batch_id % rows_per_pagesave) == rows_per_pagesave-1) or (batch_id == nr_cross_val-1):
            fig_name = os.path.join('plots','FitQuality_{}{:02d}{}_{}.png'.format(model,  map_name[hierarchy], save_sub_plot_id, hierarchy))
            fig.savefig(fig_name)
            if mlflow_flag:
                mlflow.log_artifact(fig_name)
            plt.close()
            save_sub_plot_id = chr(ord(save_sub_plot_id)+1)

    return mape, mdape, pd.DataFrame(in_out_df)



# COMMAND ----------

def main():
    '''This function will generate different control plots concerning the service call analysis which are stored in the
        plots directory. It will also create one summary table for the MAPE and MdAPE results for the different models for
        the different product hierarchy (Line, Group, SubGroup).
        In total 7 menthods are run and compared: 'weibull', 'gamma','lfp' (Limited Failure Population),
        'mm2w' (mixture model based on 2 weibull distributions), 'tb' (turnbull), and 2 naive models
        'naive' (1 month comparison) and 'avg' (12 month average comparison). AT this stage limited data is used from
        '201705' until '201906'.

        Here is a list of figures that will be generated:
        1) FitQuality_<model><01/02/03><a/b/c/d>_<line/group/subgroup>.png
        Those plots show for 4 consecutive sales batches (one per row), 4 subplots (most left one actual versus predicted
        service calls as a function of the months in the warranty period, one more to the right is the corresponding volume
        difference plot, reflecting the percentage error as a function of the months in the warranty period. Those two do
        not use the right censored data point and is called the IC method [stands for Interval Censored], the two most right
        plots correspond to the ICR method [standing for the Interval Censored Plus Right censored data point]).
        01 is used for line, 02 for group and 03 for subgroup
        2) MAPE_<model><01/02/03>_<line/group/subgroup>.png
        Shows the Mean Absolute Percentage Error for both method (IC and ICR) as a function of the sales_batch
        3) MdAPE_<model><01/02/03>_<line/group/subgroup>.png
        Same as 2) but now for the Median Absolute Percentage Error
        4) Params_<model><01/02/03>_<line/group/subgroup>
        Some more informative plots on the values of the model parameters of the surpyval fit

        A pickle is saved containing the values in a dictionary of the mape, mdape for method={ic,icr}, for
        model={'weibull', 'gamma', 'mm2w', 'lfp', 'tb', 'naive', 'avg'}, for hierarchy={line, group, subgroup}
        In the same pickle an dataframe is stored for each hierarchy, which contain the service calls actual (for 12 months
        in the warranty period) and forecasted (again for 12 months in the warranty period) as a function of the sales batch
        (per row in the df). Those dataframes will be used to calculate the Service Call Ratio's

        OUTPUT:
        all_results.pkl is a dictionary that contains the following keys constructed as follows:
        <model>_<hierarchy>_<method>_<error_metric> where
            model in {weibull, gamma, mm2w, lfp, tb, naive, avg},
            hierarchy in {line, group, subgroup}
            method in {ic, irc}  (does not exist for the naive and the avg method)
            error_metric in {mape, mdape}
            Example:
            all_results['weibull_line_ic_mape'] =
            [29.6, 30.2, 31.3, 30.1, 26.0, 22.9, 28.2, 21.6, 30.6, 30.2, 37.1, 37.3]

        <model>_<hierarchy>_in_out_df where
            model in {weibull, gamma, mm2w, lfp, tb, naive, avg},
            hierarchy in {line, group, subgroup}
            Example:
                all_results['weibull_line_ic_in_out_df'] =
                      date qty_sold sc_m1 sc_m2 sc_m3  ... scf_m10 scf_m11 scf_m12 sc_m13 scf_m13
                0   201705   223191  1251  1182   773  ...     401     386     373  202.0   362.0
                1   201706   239887  1394  1255   855  ...     420     404     390  239.0   378.0
                2   201707   228594  1453  1305   850  ...     417     402     388  222.0   376.0
            '''

    to_create_paths = ['pickles', 'plots', 'tables']
    for my_path in to_create_paths:
        if not os.path.exists(my_path):
            # Create the directory
            os.makedirs(my_path)

    result_dict = {}
    for model in ['weibull', 'gamma', 'lfp', 'mm2w', 'tb', 'avg', 'naive']:
        if mlflow_flag:
            for hierarchy in ['line', 'group', 'subgroup']:
                with mlflow.start_run(run_name='{}_{}'.format(model, hierarchy)):
                    print('Load data: Product {}'.format(hierarchy))
                    train_data, test_data = load_service_call_data(hierarchy)

                    train_data.to_csv('train_data.csv')
                    test_data.to_csv('test_data.csv')
                    mlflow.log_artifact('train_data.csv')
                    mlflow.log_artifact('test_data.csv')
                    print('Entering {} analysis'.format(model))
                    if model in ['weibull', 'gamma', 'lfp', 'tb', 'mm2w']:
                        mape_irc, mape_ic, mdape_irc, mdape_ic, in_out_df = get_sc_results(train_data, hierarchy, model=model, test_data=test_data)

                        result_dict.update({'{}_{}_{}_{}'.format(model, hierarchy, 'irc', 'mape'): mape_irc,
                                            '{}_{}_{}_{}'.format(model, hierarchy, 'ic', 'mape'): mape_ic,
                                            '{}_{}_{}_{}'.format(model, hierarchy, 'irc', 'mdape'): mdape_irc,
                                            '{}_{}_{}_{}'.format(model, hierarchy, 'ic', 'mdape'): mdape_ic,
                                            '{}_{}_{}'.format(model, hierarchy, 'in_out_df'): in_out_df})
                        in_out_df.to_csv('sc_act_pred_{}_{}.csv'.format(model, hierarchy))
                        mlflow.log_param('mape_irc_{}_{}'.format(model, hierarchy), mape_irc)
                        mlflow.log_param('mape_ic_{}_{}'.format(model, hierarchy), mape_ic)
                        mlflow.log_param('mdape_irc_{}_{}'.format(model, hierarchy), mdape_irc)
                        mlflow.log_param('mdape_ic_{}_{}'.format(model, hierarchy), mdape_ic)
                        mlflow.log_artifact('sc_act_pred_{}_{}.csv'.format(model, hierarchy))
                    elif model in ['avg', 'naive']:
                        mape, mdape, in_out_df = get_sc_naive_results(train_data, hierarchy, model=model, test_data=test_data)
                        result_dict.update({'{}_{}_{}'.format(model, hierarchy, 'mape'): mape,
                                            '{}_{}_{}'.format(model, hierarchy, 'mdape'): mdape,
                                            '{}_{}_{}'.format(model, hierarchy, 'in_out_df'): in_out_df})
                        in_out_df.to_csv('sc_act_pred_{}_{}.csv'.format(model, hierarchy))
                        mlflow.log_param('mape_{}_{}'.format(model, hierarchy), mape)
                        mlflow.log_param('mdape_{}_{}'.format(model, hierarchy), mdape)
                        mlflow.log_artifact('sc_act_pred_{}_{}.csv'.format(model, hierarchy))
        else:
            for hierarchy in ['line', 'group', 'subgroup']:
                print('Load data: Product {}'.format(hierarchy))
                train_data, test_data = load_service_call_data(hierarchy)
                print('Entering {} analysis'.format(model))
                if model in ['weibull', 'gamma', 'lfp', 'tb', 'mm2w']:
                    _, _, _, _, in_out_df1 = get_sc_results(train_data, hierarchy, model=model)
                    mape_irc, mape_ic, mdape_irc, mdape_ic, in_out_df2 = get_sc_results(train_data, hierarchy,
                                                                                        model=model,
                                                                                        test_data=test_data)
                    # Concatenate both dataframes
                    in_out_df = pd.concat([in_out_df1, in_out_df2], ignore_index=True)
                    result_dict.update({'{}_{}_{}_{}'.format(model, hierarchy, 'irc', 'mape'): mape_irc,
                                        '{}_{}_{}_{}'.format(model, hierarchy, 'ic', 'mape'): mape_ic,
                                        '{}_{}_{}_{}'.format(model, hierarchy, 'irc', 'mdape'): mdape_irc,
                                        '{}_{}_{}_{}'.format(model, hierarchy, 'ic', 'mdape'): mdape_ic,
                                        '{}_{}_{}'.format(model, hierarchy, 'in_out_df'): in_out_df})
                elif model in ['avg', 'naive']:
                    mape, mdape, in_out_df1 = get_sc_naive_results(train_data, hierarchy, model=model)
                    mape, mdape, in_out_df2 = get_sc_naive_results(train_data, hierarchy, model=model,
                                                                   test_data=test_data)
                    # Concatenate both dataframes
                    in_out_df = pd.concat([in_out_df1, in_out_df2], ignore_index=True)
                    result_dict.update({'{}_{}_{}'.format(model, hierarchy, 'mape'): mape,
                                        '{}_{}_{}'.format(model, hierarchy, 'mdape'): mdape,
                                        '{}_{}_{}'.format(model, hierarchy, 'in_out_df'): in_out_df})

    ## save dict into pickle
    pickle_name = os.path.join('pickles', 'all_results.pkl')
    f = open(pickle_name, 'wb')
    pickle.dump(result_dict, f)
    if mlflow_flag:
        mlflow.log_artifact(pickle_name)

    df_out = pd.DataFrame(result_dict)
    # Specify the Hive database and table name
    hive_database = 'team_group_quality'
    hive_table = 'service_call_benchmark'

    # Convert the Pandas DataFrame to a PySpark DataFrame
    df_out_spark = spark.createDataFrame(df_out)
    df_out_spark.write.mode("overwrite").option("overwriteSchema","true").option("path", f"abfss://teamspace@elxa2ns0017.dfs.core.windows.net/group_quality/databricks_tables/data_science/{hive_table}").saveAsTable(f"{hive_database}.{hive_table}")

    return df_out



# COMMAND ----------

# MAGIC %md
# MAGIC if __name__ == '__main__':

# COMMAND ----------

output_file = os.path.join('pickles','all_results.pkl')

if os.path.exists(output_file):
  with open(output_file, 'rb') as f:
      result_dict = pickle.load(f)
else:
  df_out = main()

# COMMAND ----------



# COMMAND ----------



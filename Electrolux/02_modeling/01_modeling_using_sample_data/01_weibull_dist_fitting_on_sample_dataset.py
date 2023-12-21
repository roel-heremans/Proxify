# Databricks notebook source
# Install packages
%pip install surpyval

# COMMAND ----------

# import packages
import pandas as pd
import numpy as np
import surpyval as surv
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Data load

# COMMAND ----------

df = spark.sql('select * from team_group_quality.service_call_forecasting')
display(df)

# COMMAND ----------

# MAGIC %md ### Service Call Forecasting
# MAGIC Step by step:
# MAGIC 1. Collect data of parts failed under warranty in Neveda format
# MAGIC 2. Convert data from Nevada format to tabular format
# MAGIC 3. Identify distribution that fits the data well
# MAGIC 4. Estimate distribution parameters
# MAGIC 5. Estimate probability of failure at warranty period
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC ####Filters:
# MAGIC - Warranty Category:	First year warranty
# MAGIC - Service Type:	Service call
# MAGIC - Reference Market Global Options:	Reference Markets Only
# MAGIC - Markets:	Europe
# MAGIC - Plants:	Electrolux Plants
# MAGIC - Global Product Hierarchy:	Food Preservation
# MAGIC - Components:	All
# MAGIC
# MAGIC ####Sample data taken
# MAGIC
# MAGIC - Date	Qty Sold	0	1	2	3	4	5	6	7	8	9	10	11	12	Qty Serv
# MAGIC - 201705	223191	1251	1182	773	602	539	415	322	335	291	259	274	344	202	6789
# MAGIC - 201706	239887	1394	1255	855	658	496	383	350	313	270	274	339	381	239	7207
# MAGIC - 201707	228594	1453	1305	850	590	398	366	324	289	321	319	357	448	222	7242
# MAGIC - 201708	219271	1388	1169	698	471	412	359	247	272	309	346	408	377	183	6639
# MAGIC - 201709	223009	1379	1151	556	501	366	335	294	323	341	381	434	383	197	6641
# MAGIC - 201710	221593	1216	996	  618	447	368	331	364	406	376	416	416	398	177	6529
# MAGIC - 201711	218133	1024	975	  602	400	355	393	386	371	384	325	377	330	109	6031
# MAGIC - 201712	158218	910	  817	  508	388	316	405	376	380	335	351	278	237	135	5436
# MAGIC - 201801	172558	894	  777	  462	365	419	339	346	360	323	243	214	232	87	5061
# MAGIC - 201802	162798	951	  791	  556	444	432	413	326	364	283	213	222	193	113	5301
# MAGIC - 201803	179055	841	  795	  563	458	407	378	374	330	194	197	222	209	90	5058
# MAGIC - 201804	179193	910	  868	  610	521	425	387	312	232	228	199	212	182	118	5204
# MAGIC - 201805	216338	963	  1030	696	544	506	364	272	274	257	206	211	267	149	5739
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md ### Convert data from Naveda fromat to Tabular format

# COMMAND ----------

food_preservation_sample_data = pd.DataFrame(columns=['date','qty_sold','month_1','month_2','month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12', 'month_13'])

food_preservation_sample_data['date']= ['201705','201706','201707','201708','201709','201710','201711','201712','201801','201802','201803','201804','201805']
food_preservation_sample_data['qty_sold']=[ 223191,239887,228594,219271,223009,221593,218133,158218,172558,162798,179055,179193,216338]
food_preservation_sample_data['month_1']= [1251,1394,1453,1388,1379,1216,1024,910	,894	,951	,841	,910	,963]
food_preservation_sample_data['month_2']= [1182,1255,1305,1169,1151,996	,975	,817	,777	,791	,795	,868	,1030]
food_preservation_sample_data['month_3']= [773,855,850,698,556,618,602,508,462,556,563,610,696]
food_preservation_sample_data['month_4']= [ 602,658,590,471,501,447,400,388,365,444,458,521,544]
food_preservation_sample_data['month_5']= [ 539 ,496 ,398 ,412 ,366 ,368 ,355 ,316 ,419 ,432 ,407 ,425 ,506]
food_preservation_sample_data['month_6']= [ 415, 383, 366, 359, 335, 331, 393, 405, 339, 413, 378, 387, 364]
food_preservation_sample_data['month_7']= [ 322, 350, 324, 247, 294, 364, 386, 376, 346, 326, 374, 312, 272]
food_preservation_sample_data['month_8']= [ 335, 313, 289, 272, 323, 406, 371, 380, 360, 364, 330, 232, 274]
food_preservation_sample_data['month_9']= [ 291, 270, 321, 309, 341, 376, 384, 335, 323, 283, 194, 228, 257]
food_preservation_sample_data['month_10']= [ 259, 274, 319, 346, 381, 416, 325, 351, 243, 213, 197, 199, 206]
food_preservation_sample_data['month_11']= [ 274, 339, 357, 408, 434, 416, 377, 278, 214, 222, 222, 212, 211]
food_preservation_sample_data['month_12']= [ 344, 381, 448, 377, 383, 398, 330, 237, 232, 193, 209, 182, 267]
food_preservation_sample_data['month_13']= [ 202, 239, 222, 183, 197, 177, 109, 135, 87, 113, 90, 118, 149]
food_preservation_sample_data

# COMMAND ----------

# MAGIC %md ### Interval & Right Censored Data

# COMMAND ----------


def calculate_interval_n_right_censored_data(df, batch_index) -> pd.DataFrame:
  """ compute interval and right censored data for a single shiptment batch (assuming that row in df corresponds to number of service calls each months per single batch)
  Input Example:


date	qty_sold	month_1	month_2	month_3	month_4	month_5	month_6	month_7	month_8	month_9	month_10	month_11	month_12	month_13
0	201705	223191	1251	1182	773	602	539	415	322	335	291	259	274	344	202
1	201706	239887	1394	1255	855	658	496	383	350	313	270	274	339	381	239
2	201707	228594	1453	1305	850	590	398	366	324	289	321	319	357	448	222
3	201708	219271	1388	1169	698	471	412	359	247	272	309	346	408	377	183
4	201709	223009	1379	1151	556	501	366	335	294	323	341	381	434	383	197
5	201710	221593	1216	996	618	447	368	331	364	406	376	416	416	398	177
6	201711	218133	1024	975	602	400	355	393	386	371	384	325	377	330	109
7	201712	158218	910	817	508	388	316	405	376	380	335	351	278	237	135
8	201801	172558	894	777	462	365	419	339	346	360	323	243	214	232	87
9	201802	162798	951	791	556	444	432	413	326	364	283	213	222	193	113
10	201803	179055	841	795	563	458	407	378	374	330	194	197	222	209	90
11	201804	179193	910	868	610	521	425	387	312	232	228	199	212	182	118
12	201805	216338	963	1030	696	544	506	364	272	274	257	206	211	267	149

Output:
  x= [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], 13]
  c= [2 2 2 2 2 2 2 2 2 2 2 2 1]
  n= [1251, 1182, 773, 602, 539, 415, 322, 335, 291, 259, 274, 344, 202, 216402]
  """

  c= []
  n=[]
  x=[]
  
  # creating failure/service call frequencies
  n= list(df.iloc[batch_index:batch_index+1, 2:df.shape[1]].values[0])
  s= list(df.iloc[batch_index:batch_index+1, 1:2].values[0] - sum(df.iloc[batch_index:batch_index+1, 2:df.shape[1]].values[0]))  
  n.extend(s)

  # create censor array
  c = np.full(shape=len(n)-1, fill_value=2) # interval censored
  c = np.append(c, [1]) # right-censored
  
  x= [[i, i+1] for i in range(len(n)-1)]
  x.append(len(n)-1)

  return x, c, n


# COMMAND ----------

x, c, n = calculate_interval_n_right_censored_data(food_preservation_sample_data, 0)
print(x)
print(c)
print(n)

# COMMAND ----------

# MAGIC %md ### Fitting weibull distribution and find shape and scale params

# COMMAND ----------

model_mle = surv.Weibull.fit(x=x, c=c, n=n, how='MLE', offset=False)
model_mle

# COMMAND ----------

model_mle.plot()

# COMMAND ----------

# generating 100 samples equally spaced between 0 and 13 
x_plot = np.linspace(0, 13, 100)
x_plot

# COMMAND ----------

# DBTITLE 1,Plotting PDF, CDF, Hazard Function of the fitted weibull model


model= model_mle

plt.figure();
fig, ax = plt.subplots(1 , 2, figsize=(20,4))

ax[0].plot(x_plot, model.df(x_plot), color='blue') ## probability density funciton
ax[0].set_xlabel('months')  
##ax[0].set_ylabel('probability')
ax[0].set_title('PDF')

ax[1].plot(x_plot, model.ff(x_plot), color='black') ## cumulative density funciton
ax[1].set_xlabel('months')   
ax[1].set_title('CDF')

#ax[2].plot(x_plot, model.hf(x_plot), color='red') ## Hazard funciton
#ax[2].set_xlabel('months')   
#ax[2].set_title('Hazard Function')


# COMMAND ----------

# MAGIC %md ### Expected number of failures by time t for a given population n is
# MAGIC  F(t)*n
# MAGIC

# COMMAND ----------

model= model_mle
#model = wmm

# COMMAND ----------

m1 = 11
m2 = 12
n  = 223191

# expected number of failure until m1 month
failure_unitl_m1_month=model.ff(m1)*(n)
print(f"Shipment batch: {n};  Expected Failure until {m1} month : {failure_unitl_m1_month} ")

# expected number of failure until m2 month
failure_unitl_m2_month=model.ff(m2)*(n)
print(f"Shipment batch: {n};  Expected Failure until {m2} month : {failure_unitl_m2_month} ")

# expected number of failure in m2 month
failure_in_m2_month = (failure_unitl_m2_month - failure_unitl_m1_month)
print(f"Shipment batch: {n};  Expected Failures in {m2} month : {failure_in_m2_month} ")


# COMMAND ----------

# Expected number of failure at specific time using survival function
# Computing the conditional probability of failure for future month given inital months in service using survival function
# Q(t/T) = 1 - R(t/T)) => Q(t/T) = 1 - (R(T+t)/R(T)))

t = 11 # future months
T = 0 # number of months in service

t2 = 12 # future months
T2 = 0 # number of months in service

n  = 223191

QtT= 1-(model.sf(T+t)/model.sf(T))
# expected number of failure in month t2
failure_in_t_month=QtT*(n)
print(f"Shipment batch: {n};  Expected Failures in next {t} months while being in service for {T} months already: {failure_in_t_month} ")

QtT2= 1-(model.sf(T2+t2)/model.sf(T2))
# expected number of failure in month t2
failure_in_t2_month=QtT2*(n)
print(f"Shipment batch: {n};  Expected Failures in next {t2} months while being in service for {T2} months already: {failure_in_t2_month} ")

print(f"Shipment batch: {n};  Expected Failure in month {t2+T2}: {failure_in_t2_month- failure_in_t_month}  ")


# COMMAND ----------

food_preservation_sample_data

# COMMAND ----------

t = 1 # future months
T = 9 # number of months in service

n  = 223191

QtT= 1-(model.sf(T+t)/model.sf(T))
# expected number of failure in month t2
failure_in_t_month=QtT*(n)
print(f"Shipment batch: {n};  Expected Failures in next {t} months while being in service for {T} months already: {failure_in_t_month} ")

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Plotting Actual vs Predicted Failures

def viz_actual_vs_predicted(model):
  plt.figure();
  fig, ax = plt.subplots(13, 1, figsize=(20,70))

  #N= 223191
  batch_index=0

  for N in [list(food_preservation_sample_data.iloc[i:i+1]['qty_sold'].values)[0] for i in range(13)]:
    actual_failures=[]
    predicted_failures_hf=[]
    predicted_failures_cdf=[]

    actual_failures= list(np.transpose(food_preservation_sample_data.iloc[batch_index:batch_index+1, 2:15].values))
    #predicted_failures_hf= [model.hf(i+1)*N for i in range(13)]
    predicted_failures_cdf= [model.ff(i+1)*(N) - model.ff(i)*(N) for i in range(13)]


    x= [i+1 for i in range(13)]
    ax[batch_index].plot(x, actual_failures, color='blue') ## probability density funciton
    ax[batch_index].set_xlabel('Months')  
    ax[batch_index].set_ylabel('Failures')
    ax[batch_index].set_title(f"Expected Failures: Actual vs Predicted; Batch size N: {N}")
    #ax[batch_index].plot(x, predicted_failures_hf, color='red') ## using hazard function
    ax[batch_index].plot(x, predicted_failures_cdf, color='green') ## using cdf 
    #ax[batch_index].legend(['Actual', 'Predicted (hf)', 'Predicted (cdf)'])  
    ax[batch_index].legend(['Actual', 'Predicted (cdf)'])  

    batch_index = batch_index +1


# COMMAND ----------

# DBTITLE 1,Weibull Model (MLE parameter estimation method )
viz_actual_vs_predicted(model_mle)

# COMMAND ----------

# MAGIC %fs ls abfss://ailab@elxa2ns0017.dfs.core.windows.net/
# MAGIC

# COMMAND ----------

# MAGIC %fs ls abfss://data@elxa2ns0001prd.dfs.core.windows.net/silver/sap_bw/bc_sap_monthly

# COMMAND ----------


df=spark.read.load("abfss://data@elxa2ns0001prd.dfs.core.windows.net/gold/operations__finished_goods_europe/bc_monthly")
display(df)

# COMMAND ----------



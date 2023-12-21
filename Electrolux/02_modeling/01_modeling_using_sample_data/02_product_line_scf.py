# Databricks notebook source
# MAGIC %pip install surpyval
# MAGIC %pip install mlflow

# COMMAND ----------

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import surpyval as surv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Granularity: Product Line
# MAGIC
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

# COMMAND ----------

with mlflow.start_run(run_name = "weibull_mse"):
  food_preservation_sample_data = pd.DataFrame(columns=['date','qty_sold','month_1','month_2','month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12', 'month_13'])
  food_preservation_sample_data['date']= ['201707','201706','201707','201708','201709','201710','201711','201712','201801','201802','201803','201804','201805']
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
  

  plt.figure();
  fig, ax = plt.subplots(13, 2, figsize=(20,70))
  batch_index=0
  mape = []
  food_preservation_sample_data.to_csv('food_preservation_sample_data.csv')
  mlflow.log_artifact('food_preservation_sample_data.csv')
  for N in [list(food_preservation_sample_data.iloc[i:i+1]['qty_sold'].values)[0] for i in range(13)]:
      #mlflow.sklearn.log_model(model,"weibull_mle_model")
      #mlflow.log_param("shape",model.alpha)
      #mlflow.log_param("scale",model.beta)

      actual_failures=[]
      predicted_failures_hf=[]
      predicted_failures_cdf=[]

      actual_failures= list(np.transpose(food_preservation_sample_data.iloc[batch_index:batch_index+1, 2:15].values))
      flattened_actual_failures = [item for sublist in actual_failures for item in sublist.flatten()]
      print("actual_failures_m" + str(batch_index),flattened_actual_failures)
      mlflow.log_param("actual_failures_m" + str(batch_index),flattened_actual_failures)
      append_value = food_preservation_sample_data.iloc[batch_index:batch_index+1, 1].iloc[0] - sum(flattened_actual_failures)
      flattened_actual_failures.append(append_value)
      print("flattened_actual_failures",flattened_actual_failures)

      model = surv.Weibull.fit(x=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], 13], c=[2,2,2,2,2,2,2,2,2,2,2,2,2,1], n=flattened_actual_failures, how='MSE', offset=False)
      mlflow.sklearn.log_model(model,"weibull_mse_model")
      mlflow.log_param("shape_m"+ str(batch_index),model.alpha)
      mlflow.log_param("scale_m"+ str(batch_index),model.beta)


      predicted_failures_cdf = [model.ff(i+1)*(N) - model.ff(i)*(N) for i in range(13)]
      print("predicted_failures_m" + str(batch_index),predicted_failures_cdf)
      mlflow.log_param("predicted_failures_m" + str(batch_index),predicted_failures_cdf)
      del_flattened_actual_failures = flattened_actual_failures[:-1]
      error = mean_absolute_percentage_error(del_flattened_actual_failures, predicted_failures_cdf)
      mlflow.log_metric("MAPE_m" + str(batch_index),error)
      mape.append(error)
      predicted_failures_cdf_arr = np.array(predicted_failures_cdf)
      del_flattened_actual_failures_arr = np.array(del_flattened_actual_failures) 
      volume_difference = (predicted_failures_cdf_arr - del_flattened_actual_failures_arr)/del_flattened_actual_failures_arr * 100
      positive_volume_diff =  [val if val >= 0 else 0 for val in volume_difference]
      negative_volume_diff =  [val if val < 0 else 0 for val in volume_difference]
      

      x= [i+1 for i in range(13)]
      ax[batch_index,0].plot(x, del_flattened_actual_failures, color='blue') ## probability density funciton
      ax[batch_index,0].set_xlabel('Months')  
      ax[batch_index,0].set_ylabel('Failures')
      ax[batch_index,0].set_title(f"Expected Failures: Actual vs Predicted; Batch size N: {N}")
      ax[batch_index,0].plot(x, predicted_failures_cdf, color='green') ## using cdf 
      ax[batch_index,0].legend(['Actual', 'Predicted (cdf)'])
      

      ax[batch_index,1].barh(x, positive_volume_diff, color='green')
      ax[batch_index,1].barh(x, negative_volume_diff, color='red')
      ax[batch_index,1].set_xlim(min(volume_difference), max(volume_difference))
      ax[batch_index,1].set_xlabel('Volume_diff_%')
      ax[batch_index,1].set_ylabel('ttfi')
      ax[batch_index,1].set_title('Change in ttfi compared to the actual ttfi')
      plt.savefig("weibull_mse_vol_diff_%.png")
      mlflow.log_artifact("weibull_mse_vol_diff_%.png")

      batch_index = batch_index +1
  mlflow.log_param("MAPE",mape)


# COMMAND ----------

with mlflow.start_run(run_name = "gamma_mse"):
  food_preservation_sample_data = pd.DataFrame(columns=['date','qty_sold','month_1','month_2','month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12', 'month_13'])
  food_preservation_sample_data['date']= ['201707','201706','201707','201708','201709','201710','201711','201712','201801','201802','201803','201804','201805']
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
  

  plt.figure();
  fig, ax = plt.subplots(13, 2, figsize=(20,70))
  batch_index=0
  mape = []
  food_preservation_sample_data.to_csv('food_preservation_sample_data.csv')
  mlflow.log_artifact('food_preservation_sample_data.csv')
  for N in [list(food_preservation_sample_data.iloc[i:i+1]['qty_sold'].values)[0] for i in range(13)]:
      #mlflow.sklearn.log_model(model,"weibull_mle_model")
      #mlflow.log_param("shape",model.alpha)
      #mlflow.log_param("scale",model.beta)

      actual_failures=[]
      predicted_failures_hf=[]
      predicted_failures_cdf=[]

      actual_failures= list(np.transpose(food_preservation_sample_data.iloc[batch_index:batch_index+1, 2:15].values))
      flattened_actual_failures = [item for sublist in actual_failures for item in sublist.flatten()]
      print("actual_failures_m" + str(batch_index),flattened_actual_failures)
      mlflow.log_param("actual_failures_m" + str(batch_index),flattened_actual_failures)
      append_value = food_preservation_sample_data.iloc[batch_index:batch_index+1, 1].iloc[0] - sum(flattened_actual_failures)
      flattened_actual_failures.append(append_value)
      print("flattened_actual_failures",flattened_actual_failures)

      model = surv.Gamma.fit(x=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], 13], c=[2,2,2,2,2,2,2,2,2,2,2,2,2,1], n=flattened_actual_failures, how='MSE', offset=False)
      mlflow.sklearn.log_model(model,"gamma_mse_model")
      mlflow.log_param("shape_m"+ str(batch_index),model.alpha)
      mlflow.log_param("scale_m"+ str(batch_index),model.beta)


      predicted_failures_cdf = [model.ff(i+1)*(N) - model.ff(i)*(N) for i in range(13)]
      print("predicted_failures_m" + str(batch_index),predicted_failures_cdf)
      mlflow.log_param("predicted_failures_m" + str(batch_index),predicted_failures_cdf)
      del_flattened_actual_failures = flattened_actual_failures[:-1]
      error = mean_absolute_percentage_error(del_flattened_actual_failures, predicted_failures_cdf)
      mlflow.log_metric("MAPE_m" + str(batch_index),error)
      mape.append(error)
      predicted_failures_cdf_arr = np.array(predicted_failures_cdf)
      del_flattened_actual_failures_arr = np.array(del_flattened_actual_failures) 
      volume_difference = (predicted_failures_cdf_arr - del_flattened_actual_failures_arr)/del_flattened_actual_failures_arr * 100
      positive_volume_diff =  [val if val >= 0 else 0 for val in volume_difference]
      negative_volume_diff =  [val if val < 0 else 0 for val in volume_difference]
      

      x= [i+1 for i in range(13)]
      ax[batch_index,0].plot(x, del_flattened_actual_failures, color='blue') ## probability density funciton
      ax[batch_index,0].set_xlabel('Months')  
      ax[batch_index,0].set_ylabel('Failures')
      ax[batch_index,0].set_title(f"Expected Failures: Actual vs Predicted; Batch size N: {N}")
      ax[batch_index,0].plot(x, predicted_failures_cdf, color='green') ## using cdf 
      ax[batch_index,0].legend(['Actual', 'Predicted (cdf)'])
      

      ax[batch_index,1].barh(x, positive_volume_diff, color='green')
      ax[batch_index,1].barh(x, negative_volume_diff, color='red')
      ax[batch_index,1].set_xlim(min(volume_difference), max(volume_difference))
      ax[batch_index,1].set_xlabel('Volume_diff_%')
      ax[batch_index,1].set_ylabel('ttfi')
      ax[batch_index,1].set_title('Change in ttfi compared to the actual ttfi')
      plt.savefig("gamma_mse_vol_diff_%.png")
      mlflow.log_artifact("gamma_mse_vol_diff_%.png")

      batch_index = batch_index +1
  mlflow.log_param("MAPE",mape)


# COMMAND ----------

with mlflow.start_run(run_name = "mixtureModel_mse"):
  food_preservation_sample_data = pd.DataFrame(columns=['date','qty_sold','month_1','month_2','month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12', 'month_13'])
  food_preservation_sample_data['date']= ['201707','201706','201707','201708','201709','201710','201711','201712','201801','201802','201803','201804','201805']
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
  

  plt.figure();
  fig, ax = plt.subplots(13, 2, figsize=(20,70))
  batch_index=0
  mape = []
  food_preservation_sample_data.to_csv('food_preservation_sample_data.csv')
  mlflow.log_artifact('food_preservation_sample_data.csv')
  for N in [list(food_preservation_sample_data.iloc[i:i+1]['qty_sold'].values)[0] for i in range(13)]:
      #mlflow.sklearn.log_model(model,"weibull_mle_model")
      #mlflow.log_param("shape",model.alpha)
      #mlflow.log_param("scale",model.beta)

      actual_failures=[]
      predicted_failures_hf=[]
      predicted_failures_cdf=[]

      actual_failures= list(np.transpose(food_preservation_sample_data.iloc[batch_index:batch_index+1, 2:15].values))
      flattened_actual_failures = [item for sublist in actual_failures for item in sublist.flatten()]
      print("actual_failures_m" + str(batch_index),flattened_actual_failures)
      mlflow.log_param("actual_failures_m" + str(batch_index),flattened_actual_failures)
      append_value = food_preservation_sample_data.iloc[batch_index:batch_index+1, 1].iloc[0] - sum(flattened_actual_failures)
      flattened_actual_failures.append(append_value)
      print("flattened_actual_failures",flattened_actual_failures)

      model = surv.MixtureModel(x=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], 13], c=[2,2,2,2,2,2,2,2,2,2,2,2,2,1], n=flattened_actual_failures, how='MSE', dist=surv.Weibull, m=2)
      mlflow.sklearn.log_model(model,"mixtureModel_mse_model")
      #mlflow.log_param("shape_m"+ str(batch_index),model.alpha)
      #mlflow.log_param("scale_m"+ str(batch_index),model.beta)


      predicted_failures_cdf = [model.ff(i+1)*(N) - model.ff(i)*(N) for i in range(13)]
      print("predicted_failures_m" + str(batch_index),predicted_failures_cdf)
      mlflow.log_param("predicted_failures_m" + str(batch_index),predicted_failures_cdf)
      del_flattened_actual_failures = flattened_actual_failures[:-1]
      error = mean_absolute_percentage_error(del_flattened_actual_failures, predicted_failures_cdf)
      mlflow.log_metric("MAPE_m" + str(batch_index),error)
      mape.append(error)
      predicted_failures_cdf_arr = np.array(predicted_failures_cdf)
      del_flattened_actual_failures_arr = np.array(del_flattened_actual_failures) 
      volume_difference = (predicted_failures_cdf_arr - del_flattened_actual_failures_arr)/del_flattened_actual_failures_arr * 100
      positive_volume_diff =  [val if val >= 0 else 0 for val in volume_difference]
      negative_volume_diff =  [val if val < 0 else 0 for val in volume_difference]
      

      x= [i+1 for i in range(13)]
      ax[batch_index,0].plot(x, del_flattened_actual_failures, color='blue') ## probability density funciton
      ax[batch_index,0].set_xlabel('Months')  
      ax[batch_index,0].set_ylabel('Failures')
      ax[batch_index,0].set_title(f"Expected Failures: Actual vs Predicted; Batch size N: {N}")
      ax[batch_index,0].plot(x, predicted_failures_cdf, color='green') ## using cdf 
      ax[batch_index,0].legend(['Actual', 'Predicted (cdf)'])
      

      ax[batch_index,1].barh(x, positive_volume_diff, color='green')
      ax[batch_index,1].barh(x, negative_volume_diff, color='red')
      ax[batch_index,1].set_xlim(min(volume_difference), max(volume_difference))
      ax[batch_index,1].set_xlabel('Volume_diff_%')
      ax[batch_index,1].set_ylabel('ttfi')
      ax[batch_index,1].set_title('Change in ttfi compared to the actual ttfi')
      plt.savefig("mixtureModel_mse_vol_diff_%.png")
      mlflow.log_artifact("mixtureModel_mse_vol_diff_%.png")

      batch_index = batch_index +1
  mlflow.log_param("MAPE",mape)


# COMMAND ----------

with mlflow.start_run(run_name = "average_model"):
  food_preservation_sample_data = pd.DataFrame(columns=['qty_sold','month_1','month_2','month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12', 'month_13'])
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

  food_preservation_sample_data.to_csv('food_preservation_sample_data.csv')
  mlflow.log_artifact('food_preservation_sample_data.csv')

  food_preservation_sample_data_df = food_preservation_sample_data.transpose()

  food_preservation_sample_data_mean = food_preservation_sample_data_df.mean(axis = 1)


  #convert Series to dataframe
  df_final = pd.DataFrame({'ttfi':food_preservation_sample_data_mean.index, 'average':food_preservation_sample_data_mean.values})
  df_final['actuals'] = [240976,1155,1174,753,544,404,285,283,274,278,221,246,266,195]  #201806 data
  df_final['actuals_1'] = df_final['actuals']/240976
  df_final['average_1'] = df_final['average']/203218.30
  df_final['vol_diff_%'] = (df_final['average_1'] - df_final['actuals_1'])/df_final['actuals_1'] * 100

  

  mlflow.log_param("actual_ttfi" , df_final['actuals_1'])
  mlflow.log_param("predicted_ttfi" , df_final['average_1'])


  df_final_1 = pd.DataFrame(df_final,index = [0,1, 2,3,4,5,6,7,8,9,10,11,12,13])


  df_final_1.drop([0], inplace = True)

  positive_volume_diff =  [val if val >= 0 else 0 for val in df_final_1['vol_diff_%']]
  negative_volume_diff =  [val if val < 0 else 0 for val in df_final_1['vol_diff_%']]

  plt.plot(df_final_1['ttfi'], df_final_1['actuals_1'], label="actual service call volumes")
  plt.plot(df_final_1['ttfi'], df_final_1['average_1'], label="predicted service call volumes")
  plt.xlabel("ttfi")
  plt.ylabel("actuals vs predicted service call volumes")
  plt.legend()
  plt.show()

  plt.barh(df_final_1['ttfi'], positive_volume_diff, color='green')
  plt.barh(df_final_1['ttfi'], negative_volume_diff, color='red')
  plt.xlim(min(df_final['vol_diff_%']), max(df_final['vol_diff_%']))
  plt.xlabel('Volume_diff_%')
  plt.ylabel('ttfi')
  plt.title('Change in ttfi compared to the actual ttfi')
  plt.savefig("average_model_vol_diff_%.png")
  mlflow.log_artifact("average_model_vol_diff_%.png")

  error = mean_absolute_percentage_error(df_final['actuals_1'], df_final['average_1'])
  mlflow.log_metric("MAPE_average_model" , error)



# COMMAND ----------

with mlflow.start_run(run_name = "naive_model"):
  total_qty_serv_201805 = 5739 #201805
  data_201805 = pd.DataFrame({'ttfi_months':[0,1,2,3,4,5,6,7,8,9,10,11,12],'ttfi': [963,1030,696,544,506,364,272,274,257,206,211,267,149]})
  data_201805['empirical_pdf_201805'] = data_201805['ttfi']/total_qty_serv_201805

  total_qty_serv_201705 = 6789 #201705
  data_201705 = pd.DataFrame({'ttfi_months':[0,1,2,3,4,5,6,7,8,9,10,11,12],'ttfi': [1251,1182,773,602,539,415,322,335,291,259,274,344,202]})

  data_201705['empirical_pdf_201705'] = data_201705['ttfi']/total_qty_serv_201705

  mlflow.log_param("actual_ttfi" , data_201805['empirical_pdf_201805'])
  mlflow.log_param("predicted_ttfi" , data_201705['empirical_pdf_201705'])

  vol_diff = (data_201705['empirical_pdf_201705'] - data_201805['empirical_pdf_201805'])/data_201805['empirical_pdf_201805'] * 100

  positive_volume_diff =  [val if val >= 0 else 0 for val in vol_diff]
  negative_volume_diff =  [val if val < 0 else 0 for val in vol_diff]

  plt.plot(data_201705['ttfi_months'], data_201705['empirical_pdf_201705'], label="perdicted_ttfi")
  plt.plot(data_201805['ttfi_months'], data_201805['empirical_pdf_201805'], label="actual_ttfi")
  plt.xlabel("ttfi (months)")
  plt.ylabel("pdf")
  plt.legend()
  plt.show()
  

  plt.barh(data_201705['ttfi_months'], positive_volume_diff, color='green')
  plt.barh(data_201705['ttfi_months'], negative_volume_diff, color='red')
  plt.xlim(min(volume_difference), max(volume_difference))
  plt.xlabel('Volume_diff_%')
  plt.ylabel('ttfi')
  plt.title('Change in ttfi compared to the actual ttfi')
  plt.show()

  plt.savefig("naive_model_vol_diff_%.png")
  mlflow.log_artifact("naive_model_vol_diff_%.png")

  error = mean_absolute_percentage_error(data_201805['empirical_pdf_201805'], data_201705['empirical_pdf_201705'])
  mlflow.log_metric("MAPE_naive_model" , error)


# COMMAND ----------

#MAPE_gamma_mse
run_id = "6ef0a37e7ca547f3bf68853d94b10b14" 

run = mlflow.get_run(run_id)

# Get all metrics for the run
MAPE_gamma_mse = run.data.params.get("MAPE")

MAPE_gamma_mse_list = eval(MAPE_gamma_mse)
print(MAPE_gamma_mse_list)


# COMMAND ----------

#MAPE_weibull_mse
run_id = "5a583281f5ef48c8991ce07e569a2754" 

run = mlflow.get_run(run_id)

# Get all metrics for the run
MAPE_weibull_mse = run.data.params.get("MAPE")
MAPE_weibull_mse_list = eval(MAPE_weibull_mse)
print(MAPE_weibull_mse_list)

# COMMAND ----------

#MAPE_mixtureModel_mse
run_id = "6825dd949f664edca97830afe4560bcc" 

run = mlflow.get_run(run_id)

# Get all metrics for the run
MAPE_mixtureModel_mse = run.data.params.get("MAPE")
MAPE_mixtureModel_mse_list = eval(MAPE_mixtureModel_mse)
print(MAPE_mixtureModel_mse_list)

# COMMAND ----------

#MAPE_average_model
run_id = "22385f62d53c4ea49d37ec3813914380" 

run = mlflow.get_run(run_id)

# Get all metrics for the run
MAPE_average_model = run.data.metrics.get("MAPE_average_model")
print(MAPE_average_model)

# COMMAND ----------

#MAPE_naive_model
run_id = "453fb65aeb7545aca3461773ebecd530" 

run = mlflow.get_run(run_id)

# Get all metrics for the run
MAPE_naive_model = run.data.metrics.get("MAPE_naive_model")
print(MAPE_naive_model)

# COMMAND ----------

x_values = [0,1,2,3,4,5,6,7,8,9,10,11,12]
plt.plot(x_values, MAPE_gamma_mse_list, label='MAPE_gamma_mse', marker='o',linestyle='-', color='blue')
plt.plot(x_values, MAPE_weibull_mse_list, label='MAPE_weibull_mse', marker='s',linestyle='--', color='green')
plt.plot(x_values, MAPE_mixtureModel_mse_list, label='MAPE_mixtureModel_mse',linestyle='-.', color='red')

plt.xlabel('Batches')
plt.ylabel('MAPE')
plt.title('MAPE of various models per Product Line')
plt.legend()

plt.show()



# COMMAND ----------

x_values = [0,1,2,3,4,5,6,7,8,9,10,11,12]
plt.plot(x_values, MAPE_gamma_mse_list, label='MAPE_gamma_mse', marker='o',linestyle='-', color='blue')
plt.plot(x_values, MAPE_weibull_mse_list, label='MAPE_weibull_mse', marker='s',linestyle='--', color='green')

plt.xlabel('Batches')
plt.ylabel('MAPE')
plt.title('MAPE of various models per Product Line')
plt.legend()

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Calling the model saved in MLflow
# MAGIC

# COMMAND ----------

#calling the model saved in MLflow

with mlflow.start_run(run_name = "gamma_mse"):
  food_preservation_sample_data = pd.DataFrame(columns=['date','qty_sold','month_1','month_2','month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12', 'month_13'])
  food_preservation_sample_data['date']= ['201707','201706','201707','201708','201709','201710','201711','201712','201801','201802','201803','201804','201805']
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
  

  plt.figure();
  fig, ax = plt.subplots(13, 2, figsize=(20,70))
  batch_index=0
  mape = []
  food_preservation_sample_data.to_csv('food_preservation_sample_data.csv')
  mlflow.log_artifact('food_preservation_sample_data.csv')
  for N in [list(food_preservation_sample_data.iloc[i:i+1]['qty_sold'].values)[0] for i in range(13)]:
      #mlflow.sklearn.log_model(model,"weibull_mle_model")
      #mlflow.log_param("shape",model.alpha)
      #mlflow.log_param("scale",model.beta)

      actual_failures=[]
      predicted_failures_hf=[]
      predicted_failures_cdf=[]

      actual_failures= list(np.transpose(food_preservation_sample_data.iloc[batch_index:batch_index+1, 2:15].values))
      flattened_actual_failures = [item for sublist in actual_failures for item in sublist.flatten()]
      print("actual_failures_m" + str(batch_index),flattened_actual_failures)
      mlflow.log_param("actual_failures_m" + str(batch_index),flattened_actual_failures)
      append_value = food_preservation_sample_data.iloc[batch_index:batch_index+1, 1].iloc[0] - sum(flattened_actual_failures)
      flattened_actual_failures.append(append_value)
      print("flattened_actual_failures",flattened_actual_failures)

      run_id = "86352b63e4da4388bef454d4e71d83f4"
      artifact_path = "weibull_mse_model"
      model = mlflow.sklearn.load_model("runs:/{}/{}".format(run_id, artifact_path))
      #mlflow.sklearn.log_model(model,"gamma_mse_model")
      mlflow.log_param("shape_m"+ str(batch_index),model.alpha)
      mlflow.log_param("scale_m"+ str(batch_index),model.beta)


      predicted_failures_cdf = [model.ff(i+1)*(N) - model.ff(i)*(N) for i in range(13)]
      print("predicted_failures_m" + str(batch_index),predicted_failures_cdf)
      mlflow.log_param("predicted_failures_m" + str(batch_index),predicted_failures_cdf)
      del_flattened_actual_failures = flattened_actual_failures[:-1]
      error = mean_absolute_percentage_error(del_flattened_actual_failures, predicted_failures_cdf)
      mlflow.log_metric("MAPE_m" + str(batch_index),error)
      mape.append(error)
      predicted_failures_cdf_arr = np.array(predicted_failures_cdf)
      del_flattened_actual_failures_arr = np.array(del_flattened_actual_failures) 
      ttfi = [0,1,2,3,4,5,6,7,8,9,10,11,12]
      volume_difference = (predicted_failures_cdf_arr - del_flattened_actual_failures_arr)/del_flattened_actual_failures_arr * 100
      positive_volume_diff =  [val if val >= 0 else 0 for val in volume_difference]
      negative_volume_diff =  [val if val < 0 else 0 for val in volume_difference]
      

      x= [i+1 for i in range(13)]
      ax[batch_index,0].plot(x, actual_failures, color='blue') ## probability density funciton
      ax[batch_index,0].set_xlabel('Months')  
      ax[batch_index,0].set_ylabel('Failures')
      ax[batch_index,0].set_title(f"Expected Failures: Actual vs Predicted; Batch size N: {N}")
      ax[batch_index,0].plot(x, predicted_failures_cdf, color='green') ## using cdf 
      ax[batch_index,0].legend(['Actual', 'Predicted (cdf)'])
      

      ax[batch_index,1].barh(ttfi, positive_volume_diff, color='green')
      ax[batch_index,1].barh(ttfi, negative_volume_diff, color='red')
      ax[batch_index,1].set_xlim(min(volume_difference), max(volume_difference))
      ax[batch_index,1].set_xlabel('Volume_diff_%')
      ax[batch_index,1].set_ylabel('ttfi')
      ax[batch_index,1].set_title('Change in ttfi compared to the actual ttfi')
      plt.savefig("gamma_mse_vol_diff_%.png")
      mlflow.log_artifact("gamma_mse_vol_diff_%.png")

      batch_index = batch_index +1
  mlflow.log_param("MAPE",mape)


# COMMAND ----------



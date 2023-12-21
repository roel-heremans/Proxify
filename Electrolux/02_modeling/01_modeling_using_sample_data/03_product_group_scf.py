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
# MAGIC ###Granularity: Product Group
# MAGIC
# MAGIC ####Filters:
# MAGIC - Warranty Category:	First year warranty
# MAGIC - Service Type:	Service call
# MAGIC - Reference Market Global Options:	Reference Markets Only
# MAGIC - Markets:	Europe
# MAGIC - Plants:	Electrolux Plants
# MAGIC - Global Product Hierarchy: Free Standing Refrigerator 
# MAGIC - Components:	All
# MAGIC
# MAGIC ####Sample data taken
# MAGIC - Date	Qty Sold	0	1	2	3	4	5	6	7	8	9	10	11	12	Qty Serv
# MAGIC - 201707	23729	234	146	111	53	48	53	43	31	46	34	36	57	28	920
# MAGIC - 201708	25981	204	132	70	56	63	48	35	40	32	28	44	40	23	815
# MAGIC - 201709	20530	181	139	63	47	56	32	38	29	36	48	53	42	23	787
# MAGIC - 201710	20879	156	122	70	64	47	39	47	43	40	36	38	51	17	770
# MAGIC - 201711	20653	147	122	62	39	42	48	33	51	55	29	40	39	9	716
# MAGIC - 201712	18397	107	105	54	46	36	41	37	53	40	46	27	25	16	633
# MAGIC - 201801	17832	142	90	60	31	41	38	36	50	36	29	28	25	9	615
# MAGIC - 201802	15323	113	88	62	50	46	40	42	42	25	26	17	20	9	580
# MAGIC - 201803	16650	97	87	47	50	69	41	39	39	31	26	28	22	17	593
# MAGIC - 201804	18989	136	92	53	66	48	56	30	22	26	20	24	13	15	601
# MAGIC - 201805	23677	127	121	91	64	52	44	30	36	27	25	18	28	18	681
# MAGIC - 201806	25173	185	160	81	58	59	23	23	21	36	21	29	34	19	749
# MAGIC - 201807	26484	223	131	90	52	29	29	28	30	32	34	29	33	9	749

# COMMAND ----------

with mlflow.start_run(run_name = "weibull_mse"):
  #Granularity: Product Group
  food_preservation_sample_data = pd.DataFrame(columns=['date','qty_sold','month_1','month_2','month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12', 'month_13'])
  food_preservation_sample_data['date']= ['201707','201708','201709','201710','201711','201712','201801','201802','201803','201804','201805','201806','201807']
  food_preservation_sample_data['qty_sold']=[23729,25981,20530,20879,20653,18397,17832,15323,16650,18989,23677,25173,26484]
  food_preservation_sample_data['month_1']= [234,204,181,156,147,107,142,113,97,136,127,185,223]
  food_preservation_sample_data['month_2']= [146,132,139,122,122,105,90,88,87,92,121,160,131]
  food_preservation_sample_data['month_3']= [111,70,63,70,62,54,60,62,47,53,91,81,90]
  food_preservation_sample_data['month_4']= [53,56,47,64,39,46,31,50,50,66,64,58,52]
  food_preservation_sample_data['month_5']= [48,63,56,47,42,36,41,46,69,48,52,59,29]
  food_preservation_sample_data['month_6']= [53,48,32,39,48,41,38,40,41,56,44,23,29]
  food_preservation_sample_data['month_7']= [43,35,38,47,33,37,36,42,39,30,30,23,28]
  food_preservation_sample_data['month_8']= [31,40,29,43,51,53,50,42,39,22,36,21,30]
  food_preservation_sample_data['month_9']= [46,32,36,40,55,40,36,25,31,26,27,36,32]
  food_preservation_sample_data['month_10']= [34,28,48,36,29,46,29,26,26,20,25,21,34]
  food_preservation_sample_data['month_11']= [36,44,53,38,40,27,28,17,28,24,18,29,29]
  food_preservation_sample_data['month_12']= [57,40,42,51,39,25,25,20,22,13,28,34,33]
  food_preservation_sample_data['month_13']= [28,23,23,17,9,16,9,9,17,15,18,19,9]
  

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
  #Granularity: Product Group
  food_preservation_sample_data = pd.DataFrame(columns=['date','qty_sold','month_1','month_2','month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12', 'month_13'])
  food_preservation_sample_data['date']= ['201707','201708','201709','201710','201711','201712','201801','201802','201803','201804','201805','201806','201807']
  food_preservation_sample_data['qty_sold']=[23729,25981,20530,20879,20653,18397,17832,15323,16650,18989,23677,25173,26484]
  food_preservation_sample_data['month_1']= [234,204,181,156,147,107,142,113,97,136,127,185,223]
  food_preservation_sample_data['month_2']= [146,132,139,122,122,105,90,88,87,92,121,160,131]
  food_preservation_sample_data['month_3']= [111,70,63,70,62,54,60,62,47,53,91,81,90]
  food_preservation_sample_data['month_4']= [53,56,47,64,39,46,31,50,50,66,64,58,52]
  food_preservation_sample_data['month_5']= [48,63,56,47,42,36,41,46,69,48,52,59,29]
  food_preservation_sample_data['month_6']= [53,48,32,39,48,41,38,40,41,56,44,23,29]
  food_preservation_sample_data['month_7']= [43,35,38,47,33,37,36,42,39,30,30,23,28]
  food_preservation_sample_data['month_8']= [31,40,29,43,51,53,50,42,39,22,36,21,30]
  food_preservation_sample_data['month_9']= [46,32,36,40,55,40,36,25,31,26,27,36,32]
  food_preservation_sample_data['month_10']= [34,28,48,36,29,46,29,26,26,20,25,21,34]
  food_preservation_sample_data['month_11']= [36,44,53,38,40,27,28,17,28,24,18,29,29]
  food_preservation_sample_data['month_12']= [57,40,42,51,39,25,25,20,22,13,28,34,33]
  food_preservation_sample_data['month_13']= [28,23,23,17,9,16,9,9,17,15,18,19,9]
  

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

with mlflow.start_run(run_name = "mixtureModel_mse"):
  food_preservation_sample_data = pd.DataFrame(columns=['date','qty_sold','month_1','month_2','month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12', 'month_13'])
  food_preservation_sample_data['date']= ['201707','201708','201709','201710','201711','201712','201801','201802','201803','201804','201805','201806','201807']
  food_preservation_sample_data['qty_sold']=[23729,25981,20530,20879,20653,18397,17832,15323,16650,18989,23677,25173,26484]
  food_preservation_sample_data['month_1']= [234,204,181,156,147,107,142,113,97,136,127,185,223]
  food_preservation_sample_data['month_2']= [146,132,139,122,122,105,90,88,87,92,121,160,131]
  food_preservation_sample_data['month_3']= [111,70,63,70,62,54,60,62,47,53,91,81,90]
  food_preservation_sample_data['month_4']= [53,56,47,64,39,46,31,50,50,66,64,58,52]
  food_preservation_sample_data['month_5']= [48,63,56,47,42,36,41,46,69,48,52,59,29]
  food_preservation_sample_data['month_6']= [53,48,32,39,48,41,38,40,41,56,44,23,29]
  food_preservation_sample_data['month_7']= [43,35,38,47,33,37,36,42,39,30,30,23,28]
  food_preservation_sample_data['month_8']= [31,40,29,43,51,53,50,42,39,22,36,21,30]
  food_preservation_sample_data['month_9']= [46,32,36,40,55,40,36,25,31,26,27,36,32]
  food_preservation_sample_data['month_10']= [34,28,48,36,29,46,29,26,26,20,25,21,34]
  food_preservation_sample_data['month_11']= [36,44,53,38,40,27,28,17,28,24,18,29,29]
  food_preservation_sample_data['month_12']= [57,40,42,51,39,25,25,20,22,13,28,34,33]
  food_preservation_sample_data['month_13']= [28,23,23,17,9,16,9,9,17,15,18,19,9]
  

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
  food_preservation_sample_data['qty_sold']=[23729,25981,20530,20879,20653,18397,17832,15323,16650,18989,23677,25173,26484]
  food_preservation_sample_data['month_1']= [234,204,181,156,147,107,142,113,97,136,127,185,223]
  food_preservation_sample_data['month_2']= [146,132,139,122,122,105,90,88,87,92,121,160,131]
  food_preservation_sample_data['month_3']= [111,70,63,70,62,54,60,62,47,53,91,81,90]
  food_preservation_sample_data['month_4']= [53,56,47,64,39,46,31,50,50,66,64,58,52]
  food_preservation_sample_data['month_5']= [48,63,56,47,42,36,41,46,69,48,52,59,29]
  food_preservation_sample_data['month_6']= [53,48,32,39,48,41,38,40,41,56,44,23,29]
  food_preservation_sample_data['month_7']= [43,35,38,47,33,37,36,42,39,30,30,23,28]
  food_preservation_sample_data['month_8']= [31,40,29,43,51,53,50,42,39,22,36,21,30]
  food_preservation_sample_data['month_9']= [46,32,36,40,55,40,36,25,31,26,27,36,32]
  food_preservation_sample_data['month_10']= [34,28,48,36,29,46,29,26,26,20,25,21,34]
  food_preservation_sample_data['month_11']= [36,44,53,38,40,27,28,17,28,24,18,29,29]
  food_preservation_sample_data['month_12']= [57,40,42,51,39,25,25,20,22,13,28,34,33]
  food_preservation_sample_data['month_13']= [28,23,23,17,9,16,9,9,17,15,18,19,9]

  food_preservation_sample_data.to_csv('food_preservation_sample_data.csv')
  mlflow.log_artifact('food_preservation_sample_data.csv')

  food_preservation_sample_data_df = food_preservation_sample_data.transpose()

  food_preservation_sample_data_mean = food_preservation_sample_data_df.mean(axis = 1)
  print(food_preservation_sample_data_mean)


  #convert Series to dataframe
  df_final = pd.DataFrame({'ttfi':food_preservation_sample_data_mean.index, 'average':food_preservation_sample_data_mean.values})
  df_final['actuals'] = [27886,214,128,61,43,35,23,25,22,22,28,31,41,20]  #201808 data
  df_final['actuals_1'] = df_final['actuals']/27886
  df_final['average_1'] = df_final['average']/21099.769
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
  total_qty_serv_201807 = 749
  data_201807 = pd.DataFrame({'ttfi_months':[0,1,2,3,4,5,6,7,8,9,10,11,12],'ttfi': [223,131,90,52,29,29,28,30,32,34,29,33,9]})
  data_201807['empirical_pdf_201807'] = data_201807['ttfi']/total_qty_serv_201807

  total_qty_serv_201707 = 920
  data_201707 = pd.DataFrame({'ttfi_months':[0,1,2,3,4,5,6,7,8,9,10,11,12],'ttfi': [234,146,111,53,48,53,43,31,46,34,36,57,28]})

  data_201707['empirical_pdf_201707'] = data_201707['ttfi']/total_qty_serv_201807

  mlflow.log_param("actual_ttfi" , data_201807['empirical_pdf_201807'])
  mlflow.log_param("predicted_ttfi" , data_201707['empirical_pdf_201707'])

  vol_diff = (data_201707['empirical_pdf_201707'] - data_201807['empirical_pdf_201807'])/data_201807['empirical_pdf_201807'] * 100

  positive_volume_diff =  [val if val >= 0 else 0 for val in vol_diff]
  negative_volume_diff =  [val if val < 0 else 0 for val in vol_diff]

  plt.plot(data_201707['ttfi_months'], data_201707['empirical_pdf_201707'], label="perdicted_ttfi")
  plt.plot(data_201807['ttfi_months'], data_201807['empirical_pdf_201807'], label="actual_ttfi")
  plt.xlabel("ttfi (months)")
  plt.ylabel("pdf")
  plt.legend()
  plt.show()
  

  plt.barh(data_201707['ttfi_months'], positive_volume_diff, color='green')
  plt.barh(data_201707['ttfi_months'], negative_volume_diff, color='red')
  plt.xlim(min(volume_difference), max(volume_difference))
  plt.xlabel('Volume_diff_%')
  plt.ylabel('ttfi')
  plt.title('Change in ttfi compared to the actual ttfi')
  plt.show()

  plt.savefig("naive_model_vol_diff_%.png")
  mlflow.log_artifact("naive_model_vol_diff_%.png")

  error = mean_absolute_percentage_error(data_201807['empirical_pdf_201807'], data_201707['empirical_pdf_201707'])
  mlflow.log_metric("MAPE_naive_model" , error)


# COMMAND ----------

#MAPE_gamma_mse
run_id = "924debcace614d8ca43f8a0c6f811f10" 

run = mlflow.get_run(run_id)

# Get all metrics for the run
MAPE_gamma_mse = run.data.params.get("MAPE")

MAPE_gamma_mse_list = eval(MAPE_gamma_mse)
print(MAPE_gamma_mse_list)

# COMMAND ----------

#MAPE_weibull_mse
run_id = "a31b358267f14b40aa57266f02712e0a"

run = mlflow.get_run(run_id)

# Get all metrics for the run
MAPE_weibull_mse = run.data.params.get("MAPE")
MAPE_weibull_mse_list = eval(MAPE_weibull_mse)
print(MAPE_weibull_mse_list)

# COMMAND ----------

#MAPE_mixtureModel_mse
run_id = "18ef2ef76c4c4cd8a25ac2b5975b375e" 

run = mlflow.get_run(run_id)

# Get all metrics for the run
MAPE_mixtureModel_mse = run.data.params.get("MAPE")
MAPE_mixtureModel_mse_list = eval(MAPE_mixtureModel_mse)
print(MAPE_mixtureModel_mse_list)

# COMMAND ----------

#MAPE_average_model
run_id = "a110ed2f320247ecaddb308c6e7c7d7c" 

run = mlflow.get_run(run_id)

# Get all metrics for the run
MAPE_average_model = run.data.metrics.get("MAPE_average_model")
print(MAPE_average_model)

# COMMAND ----------

#MAPE_naive_model
run_id = "917a2fe95234434c90832235e7d72384" 

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



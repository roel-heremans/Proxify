Open in pycharm a terminal and go to the directory where 
you want to create the virtual environment ; for me it is 
~Repos/Proxify/Bef4u.
$ cd ~/Repos/Proxify/Bef4u
$ python3.9 -m venv venv
$ source venv/bin/activate (when successful the cursor changes acordingly)
$ pip install -r requirements.txt

The XGBoost model is learned on 2022 data and tested on 2023 data.
1) To build an XGBoost model use the script 01_train_XGBoost.py
2) To predict anomalies use 02_detect_anomalies.py

Online method where an XGBoost model is learned on some previous train data, with
some successive test data and 1 point of prediction using the trained model. The 
same process is repeated by moving one sample further in time modell is trained again
and the next point is predicted. 
For instance: Train sample size of 500,
              Test sample size of next 100
              Prediction sample 601.
The Z-score is calculated on sample 601 as follows:
(residual(101) - <window_test_residuals> ) Sigma(windo_test_residuals)

Change point detection sliding window method (online method) contains 3 methods to detect anomalies:
3) based on Z-score
4) time series decomposition
5) statistical control charts


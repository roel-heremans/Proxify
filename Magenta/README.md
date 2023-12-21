Open in pycharm a terminal and go to the directory where
you want to create the virtual environment ; for me it is
~Repos/Proxify/Magenta.
$ cd ~/Repos/Proxify/Magenta
$ python3.9 -m venv venv
$ source venv/bin/activate (when successful the cursor changes acordingly)
$ pip install -r requirements.txt

There are 3 scripts that can be run (when running verify that the Working directory is set to "Magenta" and
the script path is for instance "Magenta/source/run_01_dashboard.py"):
- run_01_dashboard: This will generate the browser link http://127.0.0.1:8050/, when opening this in a web browser
                    the dash app will become avaialable. Here you can select the file to be processed, you can adjust
                    and test the effect of the different parameters. Once satisfied with the set of parameters, you can
                    run the batch processing with those parameters. The default directory containing the data is set to
                    "Magenta/data"

- run_02_batch_processing: Check the parameter settings in the config_dict at the beginning of the file. An example is
                            shown at the end of this README file. And run the script. It will generate the excel output
                            in the tables directory called 'processed.xlsx', this file contains multiple tabs. The first
                            tab called 'meta-info' contains information on which files were processed as well as the
                            config_dict with which the processing was done and it also contains the list of pump on
                            detections for the two methods and some summary statistics on those detections.

Some left over code has been moved into the "Magenta/non_used_code":
it contains the xgboost attempt, the logistic regression attempt,
as well as some test concerning the alternating_extrema (code developped and used in the source code for getting
from the peak_detection the alternating minima and maxima)




Explanation of the parameters in the config_dict:
=================================================
config_dict = \
  {
      'dir_to_process': '/home/rheremans/Repos/Proxify/Magenta/data/',
      'file_extension': '.xlsx',
      'smooth_factor': 5,
      'resample_string': '1T',
      'poly_fit_deg': 20,
      'dist_for_maxima': 3,
      'dist_for_minima': 3,
      'peak_prominence': 0.1,
      'ambient_h2o_dropdown': 'amb_gt_h2o',
      'res_thres_minus': -0.5,
      'res_thres_plus': 0.5,
      'timestamp_col_name':'DateTime_EAT',
      'temp_col_name':'Celsius',
      'gt_col_name':'Use_event',
      'detect_start_time': '00:00:00',
      'detect_stop_time': '23:59:00'
  }

  'dir_to_process': specifies the directory containing the data that needs processing
  'file_extension': only files in that directory with this extension will be considered for processing
  'smooth_factor': nr of bins that will be considered for smoothing the temperature, on this smoothed temperature
                   the local extrema will be derrived from. Pay attention that the Resampling of the data is done
                   prior to the smoothing, this means that when a resampling of 5 min is done that the smoothing
                   for a smooth_factor=2 would correspond to 10 minutes temperature resolution.
  'resample_string': resampling of the initial data to bring all the timestamps equidistent. When put on "1T", it means
                     that the timestamps will be each minute, '5T' would be each 5 minutes. When you choose '1T' but the
                     actual data is only smapled each 5 min, an interpolation will be done and the final output will be
                     1 min samples.
  'poly_fit_deg': the degree of the polynomial fit, when you do not have a lot of samples, you should put a lower degree
                  otherwise your fit would just go through all the datapoints because there would be no degree of freedom
                  left (degree should be < Number of datapoints). The polynomial fit is just there to get an idea of
                  what the ambient temperature would be at that moment in time. This is only a correct proxy when there
                  are not too many pump usages that could affect the temperature.
                  The upperbound of the blue band is calculated based on a polyfit that goes through the convex_hull on
                  the maxima, the lower bound of the blue band is calculated based on a polyfit that goes through the
                  convec hull on the minima.
  'dist_for_maxima': minimum number of samples in time between two maxima
  'dist_for_minima': minimum number of samples in time between two minima (this will only affect when the "Amb > H2O Temp")
  'prominence': controls the peak intensity, when you give a low value it will als be able to catch lower local extrema
  'ambient_h2o_dropdown': here you can choose if the ambient temp is bigger than the H2O temp (Ambient > H2O Temp)
                           or if the ambient temp is lower than the H2O temp ( Ambient < H2O Temp)
  'res_thres_minus': can be finetuned to get more and longer detections when lowering the value
                     (this will only affect when the "Amb > H2O Temp" flag is chosen)
  'res_thres_plus': can be finetuned to get more and longer detections when lowering the value
                    (this will only affect when the "Amb < H2O Temp" flag is chosen)
  'timestamp_col_name': default ('DateTime_EAT') column name related to the excel-data-file that contains the time
  'temp_col_name': default ('Celsius') column name related to the excel-data-file that contains the temp ,
  'gt_col_name': default ('Use_event') column name related to the ground truth, when there is no groundtruth this can be
                 empty
  'detect_start_time': default ('00:00:00'), is the start time from which one would like to consider the detection
  'detect_stop_time': default ('23:59:00'), is the end time for the detections
                      when running with the default values no pump detections will be filtered out, when you choose
                      something like '08:00:00' and '18:00:00' only detections will be considered that fall between
                      8AM and 8PM, potetial pump usage that fall out of this time range will be ignored.


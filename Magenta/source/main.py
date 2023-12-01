import os
import pandas as pd
import matplotlib.pyplot as plt

def import_xlsx_to_df(filename, timestamp_col_name='DateTime_EAT', temp_col_name='Celsius', gt_col_name='',
                      gt_start_col_name='', gt_end_col_name='', batch_col_name=''):
    '''
    Reads in the excel datafiles containing at least:
    -a column that reflects on the timestamp (formated like 2023-01-06 12:21:26)
    -a column containing the temperatures
    -optional: a column containing the annotation if the pump was on or off during that timestamp
    -optional: a column wiht the start pump times
    -optional: a column with the end pump times
    -optional: batch reflecting on one location or one sequnece of recordings without timestamp gaps

    :param filename: xlsx file to be processed (path and basename joined into filename)
    :param timestamp_col_name: the name of the column that contains the timestamp
    :return:
    '''
    df = pd.read_excel(filename)
    df[timestamp_col_name] = pd.to_datetime(df[timestamp_col_name])
    df = df.set_index(timestamp_col_name)

    col_name_dict = {'temp_col_name': temp_col_name}
    col_name_optional_dict = {'gt_col_name': gt_col_name,
                              'gt_start_col_name':gt_start_col_name,
                              'gt_end_col_name': gt_end_col_name,
                              'batch_col_name': batch_col_name}

    for key,value in col_name_optional_dict.items():
        if value:   # only when non-empty string, the col_name_dict gets updated
            col_name_dict.update({key: value})

    return df, col_name_dict


def plot_data(df, col_name_dict):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    # Plotting the 'Celcius' column against the timestamp index
    ax1.plot(df.index, df[col_name_dict['temp_col_name']], marker='o', linestyle='-', color='blue')  # Adjust color if needed

    # Customize ax1 plot
    ax1.set_title('Temperature over Time')
    ax1.set_ylabel('Celcius')
    ax1.grid(True)


if __name__ == "__main__":
    '''
    Set the working directory in the configuration to  '/home/rheremans/Repos/Proxify/Magenta/', in this way the data 
    can be read in as mentioned in the code below.
    '''
    file01 = 'Consolidated UG Data Jan 2023'
    file02 = 'Kaliro Use Data (Kakosi Budumba) 230912'

    data_file01 = os.path.join('data', '{}.xlsx'.format(file01))
    data_file02 = os.path.join('data', '{}.xlsx'.format(file02))

    df, col_name_dict = import_xlsx_to_df(data_file01,
                                          gt_col_name='Use_event',
                                          gt_start_col_name='use_start',
                                          gt_end_col_name='use_end',
                                          batch_col_name='batch')

    a=1

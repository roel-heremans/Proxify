import os

from utils.util import create_output_directories, import_xlsx_to_df, plot_data, plotly_data


if __name__ == "__main__":
    '''
    This main vfisualizes the data from the file_names
    Set the working directory in the run configuration to  '/home/rheremans/Repos/Proxify/Magenta/', in this way the data 
    can be read in as mentioned in the code below.
    '''

    create_output_directories()

    file_names = {
        '01': 'Consolidated UG Data Jan 2023',
        '02': 'Kaliro Use Data (Kakosi Budumba) 230912'
    }
    data_files = {key: os.path.join('data', '{}.xlsx'.format(value)) for key, value in file_names.items()}

    for key, value in data_files.items():
        file_id = key
        selected_file = value

        df, col_name_dict = import_xlsx_to_df(selected_file,
                                              gt_col_name='Use_event',
                                              gt_start_col_name='use_start',
                                              gt_end_col_name='use_end',
                                              batch_col_name='batch')

        batch = col_name_dict.get('batch_col_name','')
        if batch:
            for batch_id in df[batch].unique():
                dict_for_plot_title = {'file': [file_id, file_names[file_id]], 'batch': batch_id }
                plot_data(df[df[batch]==batch_id], col_name_dict, dict_for_plot_title)
                plotly_data(df[df[batch]==batch_id], col_name_dict, dict_for_plot_title)

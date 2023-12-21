import pandas as pd
import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

def create_summary_figures(df):
    ''' Creates the Summary.png files showing on the x-axis the two different methods (IC and IRC) and on the y-axis the
    percentage error. All models are show for IC and for IRC (each model is slightly shifted in the x direction so that
    they are not plotted on top of each other. At the same time the information for the output table is collected in a
    dictionary

    INPUT: df that looks like, where the rows correspond to the different sales batches considered:
                weibull_line_irc_mape  ...  naive_subgroup_mdape
            0               47.073074  ...             16.354354
            1               53.975742  ...             44.858134
            2               66.681285  ...             71.473730
            3               47.056594  ...             34.295663
            4               38.404052  ...             47.683364
            5               56.899473  ...             30.220601
            6               64.967194  ...             27.126900
            7               70.164005  ...             46.113739
            8               56.572650  ...             27.790828
            9               99.519527  ...             11.998985
            10              68.838043  ...             32.622844
            11              59.373186  ...             16.343635

            [12 rows x 72 columns]

    OUTPUT:
    -Figures called Summary_<hierarchy>_<method>.png (with hierarchy in {line, group, subgroup} and method in {ic, ircc}
    -table_dict: containing the overal result
        mape_mean	mape_median	mape_std	mdape_mean	mdape_median	mdape_std
        60.793	    58.13632    15.562  	46.2938 	46.6766     	14.8135
        46.987	    45.76299	15.106  	35.0886 	33.5586     	12.6377
        41.958	    43.36924	8.8232  	33.4203 	32.6616     	11.9958
        1478.7	    1452.491	188.50  	1076.91 	1096.31     	130.612
        29.163	    25.34266	10.044  	25.5239 	22.4308     	10.4123
        First row till last row corresponds to the respective models 'weibull', 'gamma', 'lfp', 'mm2w', 'tb'.
    '''
    # mapping between model and [color, horizontal shift for the plot]
    models_cmap = {'weibull': ['b',-3], 'gamma': ['r', -2], 'lfp': ['g', -1], 'mm2w': ['m', 0], 'tb': ['c',1], 'naive': ['y', 2], 'avg': ['k', 3]}

    table_dict = {}

    for hierarchy in ['line', '_group', 'subgroup']:
        for metric in ['irc', 'ic']:
            fig, ax = plt.subplots()
            # Plot data for each method
            for method in ['mape','mdape']:
                for model in models_cmap.keys():
                    if model in ['naive', 'avg']:
                        my_df = df[[col for col in df.columns if (hierarchy in col) and
                            (method in col)  and (model in col)]]
                    else:
                        my_df = df[[col for col in df.columns if (hierarchy in col) and
                            (method in col) and (metric in col) and (model in col)]]

                    mean = my_df.mean()
                    median = my_df.median()
                    std_dev = my_df.std()

                    # Since we do not want toe have the MAPE and MDAPE results on top of each other for the different
                    # models we need to intriduce a small translation arounf the MAPE and around the MdAPE location
                    # expressed in 10 percent quanta per model
                    trans = Affine2D().translate(+0.1*models_cmap[model][1], 0.0) + ax.transData
                    mlt_fac = 1
                    label = '{}'.format(model)

                    if (model == 'mm2w') and (metric == 'irc'):
                        mlt_fac = 0.01
                    if (model == 'mm2w') and (metric == 'ic'):
                        mlt_fac = 0.5

                    if mlt_fac != 1:
                        label = '{} x{}'.format(model,mlt_fac)
                    ax.errorbar(method, mean*mlt_fac, yerr=std_dev*mlt_fac, marker='o', color=models_cmap[model][0], label=label, transform=trans)
                    print('hierarchy: {} - metric: {} - method:{} - model:{} : mean={}, median={}, std={}'.format(hierarchy,metric,method,model,mean, median, std_dev))
                    table_dict.update({'{}-{}-{}-{}'.format(hierarchy, metric,model,method): [mean.values[0], median.values[0], std_dev.values[0]]})

            # Set labels and title
            ax.set_xlabel('Methods')
            ax.set_ylabel('Percentage Error [%]')
            ax.set_title('Product {} - {}'.format(hierarchy.replace("_",""),metric))

            # Create the legend with unique items
            handles, labels = ax.get_legend_handles_labels()
            unique_labels = []
            unique_handles = []
            for handle, label in zip(handles, labels):
                if label not in unique_labels:
                    unique_labels.append(label)
                    unique_handles.append(handle)

            # Display the plot with the modified legend
            ax.legend(unique_handles, unique_labels,loc='upper left', bbox_to_anchor=(1, 1))

            # Display the plot
            plt.tight_layout()
            plt.grid()
            plt.ylim([0,80])
            fig_name = 'Summary_{}_{}.png'.format(hierarchy.replace("_",""), metric)
            plt.savefig(os.path.join('plots',fig_name))
    return table_dict


def export_batch_tables(df):
    '''For each column in the df a table is written in a seperate tab of the xls file, "benchmark.xlsx" with the
    error metric values (mape/mdape) on the service calls (actual - forecasted)/actual. One row in the df corresponds
    to a particular sales_batch of a given month.
    Those tables are mainly used for creating the content of presentations.

    example output:
    in the tab called line_irc_mape:
    weibull_line_irc_mape	gamma_line_irc_mape	lfp_line_irc_mape	tb_line_irc_mape	mm2w_line_irc_mape
    29.6191098943458	    21.1795516658247	21.8082520064724	 0	                1086.27541537477
    30.26018515582	        20.8897647072328	21.6998430551854	 0	                1099.8012710596
    31.3578081866775	    21.8470605155074	23.4639204472821	 0	                1062.89162402157
    30.1655533261579	    22.7562916149429	23.8284852487903	 0	                1131.7535851495
    26.0002177448214	    19.7586794820573	20.7535476989055	 0	                1199.79945009936
    22.984602707114	        20.6134278919791	20.3751995844159	 0	                1263.70929948027
    28.2966560200421	    24.6339279217431	24.2364867232634	 0	                1356.26703468049
    21.6908302018548	    20.2685805360126	16.9705412878182	 0	                1086.80217849729
    30.6312832768039	    26.8270864183793	25.2549866343116	 0	                1236.72634700524
    30.2133383574957	    25.4304827451008	10.614391858825	     0	                1051.02868937124
    37.163175477048	        32.9827660524099	15.2115934798515	 0	                1217.70885351755
    37.3745357799729	    29.1007433924667	28.50238173174	     0	                1131.69006225245
    37.4926299867965	    26.9446864475687	27.5266148809684	 0	                1237.46539895667
    '''
    writer = pd.ExcelWriter(os.path.join('tables','benchmark.xlsx'), engine="xlsxwriter")
    for hierarchy in ['line', '_group', 'subgroup']:
        for method in ['irc', 'ic']:
            for metric in ['mape', 'mdape']:
                my_df = df[[col for col in df.columns if
                            (hierarchy in col) and
                            (method in col) and
                            (metric in col)]]
                for col in my_df:
                    print('{}: mean={}, median={}, std={}'.format(my_df[col].name, my_df[col].mean(), my_df[col].median(), my_df[col].std()))
                my_df.to_excel(writer, sheet_name='{}_{}_{}'.format(hierarchy,method,metric), index=False)
    writer.close()


def export_summary_tables(table_dict):
    '''A summary table is created collecting the overal result. For each hierarchy and method (IC, IRC) a tab is created
      in the final.xlsx file containing the following information:
        mape_mean	mape_median	mape_std	mdape_mean	mdape_median	mdape_std
        60.793	    58.13632    15.562  	46.2938 	46.6766     	14.8135
        46.987	    45.76299	15.106  	35.0886 	33.5586     	12.6377
        41.958	    43.36924	8.8232  	33.4203 	32.6616     	11.9958
        1478.7	    1452.491	188.50  	1076.91 	1096.31     	130.612
        29.163	    25.34266	10.044  	25.5239 	22.4308     	10.4123
      First row till last row corresponds to the respective models 'weibull', 'gamma', 'lfp', 'mm2w', 'tb'.'''

    models = ['weibull', 'gamma', 'lfp', 'mm2w', 'tb']
    columns = ['mape_mean', 'mape_median', 'mape_std', 'mdape_mean', 'mdape_median', 'mdape_std']
    my_df = pd.DataFrame(index=models, columns=columns)

    writer = pd.ExcelWriter(os.path.join('tables','final.xlsx'), engine="xlsxwriter")
    for hierarchy in ['line','_group','subgroup']:
        for method in ['irc', 'ic']:
            my_df = pd.DataFrame(index=models, columns=columns)
            for model in models:
                my_df.loc[model] = np.concatenate((table_dict['{}-{}-{}-mape'.format(hierarchy, method, model)],
                                table_dict['{}-{}-{}-mdape'.format(hierarchy, method, model)]))

            my_df.to_excel(writer, sheet_name='{}_{}'.format(hierarchy,method), index=False)
    writer.close()


def main(result_dict):
    '''

    :param result_dict:
    :return:
    '''

    # selecting the mape and mdape information from the pickle file and store them in a dataframe so that a excel
    # table can be outputted. Those tables are used in the presentations to the Quality team.
    selected_dict_keys = [key for key in result_dict.keys() if 'in_out_df' not in key]
    dict_subset = {key: result_dict[key] for key in selected_dict_keys}

    #The different columns in the df are:
    # df.columns
    # ['weibull_line_irc_mape', 'weibull_line_ic_mape',
    #  'weibull_line_irc_mdape', 'weibull_line_ic_mdape',
    #  'weibull_group_irc_mape', 'weibull_group_ic_mape',
    #  'weibull_group_irc_mdape', 'weibull_group_ic_mdape',
    #  'weibull_subgroup_irc_mape', 'weibull_subgroup_ic_mape',
    #  'weibull_subgroup_irc_mdape', 'weibull_subgroup_ic_mdape',
    #  'gamma_line_irc_mape', 'gamma_line_ic_mape', 'gamma_line_irc_mdape',
    #  'gamma_line_ic_mdape', 'gamma_group_irc_mape', 'gamma_group_ic_mape',
    #  'gamma_group_irc_mdape', 'gamma_group_ic_mdape',
    #  'gamma_subgroup_irc_mape', 'gamma_subgroup_ic_mape',
    #  'gamma_subgroup_irc_mdape', 'gamma_subgroup_ic_mdape',
    #  'lfp_line_irc_mape', 'lfp_line_ic_mape', 'lfp_line_irc_mdape',
    #  'lfp_line_ic_mdape', 'lfp_group_irc_mape', 'lfp_group_ic_mape',
    #  'lfp_group_irc_mdape', 'lfp_group_ic_mdape', 'lfp_subgroup_irc_mape',
    #  'lfp_subgroup_ic_mape', 'lfp_subgroup_irc_mdape',
    #  'lfp_subgroup_ic_mdape', 'mm2w_line_irc_mape', 'mm2w_line_ic_mape',
    #  'mm2w_line_irc_mdape', 'mm2w_line_ic_mdape', 'mm2w_group_irc_mape',
    #  'mm2w_group_ic_mape', 'mm2w_group_irc_mdape', 'mm2w_group_ic_mdape',
    #  'mm2w_subgroup_irc_mape', 'mm2w_subgroup_ic_mape',
    #  'mm2w_subgroup_irc_mdape', 'mm2w_subgroup_ic_mdape', 'tb_line_irc_mape',
    #  'tb_line_ic_mape', 'tb_line_irc_mdape', 'tb_line_ic_mdape',
    #  'tb_group_irc_mape', 'tb_group_ic_mape', 'tb_group_irc_mdape',
    #  'tb_group_ic_mdape', 'tb_subgroup_irc_mape', 'tb_subgroup_ic_mape',
    #  'tb_subgroup_irc_mdape', 'tb_subgroup_ic_mdape', 'avg_line_mape',
    #  'avg_line_mdape', 'avg_group_mape', 'avg_group_mdape',
    #  'avg_subgroup_mape', 'avg_subgroup_mdape', 'naive_line_mape',
    #  'naive_line_mdape', 'naive_group_mape', 'naive_group_mdape',
    #  'naive_subgroup_mape', 'naive_subgroup_mdape']
    df = pd.DataFrame.from_dict(dict_subset)

    # once the df is created the table can be easily written:
    export_batch_tables(df)

    # creates tables of the overall result
    table_dict = create_summary_figures(df)
    export_summary_tables(table_dict)

if __name__ == '__main__':
    output_file = os.path.join('pickles', 'all_results.pkl')

    if os.path.exists(output_file):
        with open(os.path.join('pickles', 'all_results.pkl'), 'rb') as f:
            result_dict = pickle.load(f)
        main(result_dict)
    else:
        print('File: {} does not exist. Run script 05_benchmark_data.py first.'.format(output_file))



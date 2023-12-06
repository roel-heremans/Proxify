from utils.util import my_xgb, my_logistic_regression, my_local_extrema, plot_evaluation
import matplotlib.pyplot as plt

def make_evaluation_plot(res):
    '''
    Comparing the different evaluation metrices for the different models
    :param res: is a dict containing the results of the different models
    :return:
    '''

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    markers = ['+', 'o']  # Markers for train and test data
    colors = ['blue', 'green', 'red']  # Different colors for each model

    models = list(res.keys())
    metrics = list(res[models[0]].keys())
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for idx, metric in enumerate(metrics):
        ax = axs[positions[idx]]
        for i, (model, values) in enumerate(res.items()):
            train_value, test_value = values[metric]
            ax.plot(i, train_value, marker='+', color=colors[i], label=f'{model} Train', markersize=8)
            ax.plot(i, test_value, marker='o', color=colors[i], label=f'{model} Test', markersize=8)

        ax.set_xticks(range(len(res)))
        ax.set_xticklabels(res.keys()) #, rotation=45)
        ax.set_title(metric)
        ax.set_ylim(0, 1)  # Set y-limits to 0 and 1

        # Remove x-labels for subplots at the top
        if idx < 2:
            ax.set_xticklabels([])

        # Remove legend from subplots
        ax.legend().remove()

    # Create a common legend outside the subplots
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')


    plt.tight_layout()



if __name__ == "__main__":
    res = {}
    res.update({'xgb': my_xgb()})
    res.update({'log_reg': my_logistic_regression()})
    res.update({'loc_ext': my_local_extrema()})
    plot_evaluation(res)

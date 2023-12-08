import os
import pandas as pd
import plotly.graph_objects as go

# Import modules/functions/classes to be tested
from utils.util import evaluate_with_tolerance


def get_unittest_data():
    gt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0,
       0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1,
       1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
       0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
       0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0,
       0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    pred = [False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False,  True,  True,  True,
       False, False, False, False, False, False, False, False,  True,
        True,  True,  True,  True,  True,  True,  True,  True, False,
       False, False, False,  True,  True,  True,  True,  True, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True, False, False, False, False, False,
        True,  True,  True,  True, False, False, False, False, False,
       False,  True,  True,  True,  True,  True,  True,  True,  True,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False,  True,  True,  True, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
        True,  True,  True,  True,  True,  True,  True, False, False,
       False,  True,  True,  True,  True, False,  True,  True,  True,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False,  True,  True,  True,  True,
        True,  True,  True,  True,  True, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True, False, False,  True,  True,  True,  True,  True,
        True, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True, False, False, False, False, False, False, False,
        True,  True,  True,  True,  True, False, False,  True,  True,
       False, False, False, False, False, False, False,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True, False,
       False,  True,  True, False, False,  True,  True,  True,  True,
        True,  True, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False]

    #gt   = [0,1,0,1,0,0,0,0,1,1,1,0]
    #pred = [False,False,False,True,False,False,False,True,True,True,True,False]
    index = pd.date_range(start='2023-02-02 08:44:00', periods=len(gt), freq='T')  # Datetime index

    # Creating the Series
    my_ground_truth_data = pd.Series(gt, index=index, name='GroundTruth')
    my_prediction_data = pd.Series(pred, index=index, name='GroundTruth')

    return my_ground_truth_data, my_prediction_data

def plot_evaluation(y,pred, tol_minutes):
    annotation_height = 0.1
    bin_width = y.index[1] - y.index[0]

    temp = y.copy()
    temp.loc[:] = 1
    # Create the base line plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=temp.index, y=temp, mode='lines', name='Temp_smooth'))

    color_map = {0: "#41b6c4", 1: "#2c7fb8"}
    anno_map = {False: "#41b6c4", True: "#2c7fb8"}

    # Add rectangles and legend handles
    legend_handles = []
    for value, color in color_map.items():
        subset = y[y == value]
        for index, row in subset.items():
            bin_start = index - bin_width / 2
            bin_end = index + bin_width / 2
            fig.add_shape(type="rect", x0=bin_start, y0=1, x1=bin_end, y1=2,
                          line=dict(color=color, width=1), fillcolor=color_map[value], opacity=0.4)

        legend_handles.append(go.Scatter(x=[None], y=[None], mode='markers',
                                         marker=dict(color=color, size=0), name=anno_map[value]))

    for value, color in anno_map.items():
        subset = pred[pred == value]
        for index, row in subset.items():
            bin_start = index - bin_width / 2
            bin_end = index + bin_width / 2
            fig.add_shape(type="rect", x0=bin_start, y0=0, x1=bin_end, y1=1,
                          line=dict(color=color, width=1), fillcolor=color_map[value], opacity=0.4)

        legend_handles.append(go.Scatter(x=[None], y=[None], mode='markers',
                                         marker=dict(color=color, size=0), name=anno_map[value]))


    # Adjust the layout
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=1.05, xanchor="center", x=0.5),
        xaxis=dict(title='Timestamp'),
        yaxis=dict(title='Temperature'),
        title='Temperature Extrema Visualization'
    )
    fig.show()
    fig.write_html(os.path.join('plotly','Eval_Tol_{}min.html'.format(tol_minutes)))


def test_case_one():

    tol_minutes_list = [1,5,10,15,30,45,60,90]
    for tol_minutes in tol_minutes_list:
        y, y_pred = get_unittest_data()
        #plot_evaluation(y, y_pred)

        tolerance =  tol_minutes
        accuracy, precision, recall, f1_score = evaluate_with_tolerance(y, y_pred, tolerance)



if __name__ == '__main__':
    test_case_one()




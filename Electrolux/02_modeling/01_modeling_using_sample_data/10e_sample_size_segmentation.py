from utils.util import load_sunburst_data
import plotly.express as px
import os
import pandas as pd
import numpy as np
import plotly.io as pio

def make_sample_size_study():

    sunburst_data = load_sunburst_data('SCR')
    df_sun = pd.DataFrame(sunburst_data, columns=['hierarchy', 'mape', 'mdape', 'mape_std', 'mdape_std','qty'])

    # Assuming df_sun is your DataFrame

    # Categorizing the hierarchy based on conditions
    def categorize_hierarchy(hierarchy):
        if hierarchy.count("_all") == 2:
            return "Product Line"
        elif hierarchy.count("_all") == 1:
            return "Product Group"
        else:
            return "Product Subgroup"

    # Creating a new column for category
    df_sun['category'] = df_sun['hierarchy'].apply(categorize_hierarchy)

    # Creating a column for log(qty)
    df_sun['log_qty'] = np.log(df_sun['qty'])

    # Creating the scatter plot using Plotly Express
    fig = px.scatter(df_sun, x='mape', y='log_qty', color='category', hover_data=['hierarchy'])
    fig.update_traces(marker=dict(size=12))
    # Customizing the layout and displaying the plot
    fig.update_layout(title='Scatter plot of MAPE vs log(Qty) with Hierarchy Categories',
                      xaxis_title='MAPE',
                      yaxis_title='log(<Qty>)',
                      xaxis=dict(title_font=dict(size=29, family='Arial', color='black'),  # Adjust the font properties for x-axis label
                      tickfont=dict(size=16, family='Arial', color='black')),   # Adjust the font properties for x-axis ticks
                      yaxis=dict(title_font=dict(size=20, family='Arial', color='black'),  # Adjust the font properties for y-axis label
                      tickfont=dict(size=16, family='Arial', color='black')),   # Adjust the font properties for y-axis ticks)
                      legend=dict(
                        x=0.80,  # Adjust the x-coordinate to position the legend
                        y=0.98,  # Adjust the y-coordinate to position the legend
                        traceorder='normal',
                        font=dict(family='Arial', size=12, color='black'),
                        bgcolor='rgba(255, 255, 255, 0.5)',
                        bordercolor='rgba(0, 0, 0, 0.5)',
                        borderwidth=2)

                      )
    fig.show()

    # Save the figure as an HTML file
    pio.write_html(fig, file=os.path.join('plotly','qty_segmentation_mape.html'), auto_open=True)
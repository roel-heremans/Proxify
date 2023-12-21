import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.util import load_sunburst_data
from utils.util import create_scr_sunburst_figure_hierarchy


#import os
#import pandas as pd
#import plotly.express as px
#import base64
#from PIL import Image
#
#def create_scr_sunburst_figure_hierarchy(sunburst_data, which):
#    res = []
#    for hierarchy, mape, mdape, mape_std, mdape_std in sunburst_data:
#        h_split = hierarchy.split("_")
#        res.append(h_split + [mape])
#
#    df = pd.DataFrame(res, columns=['Line', 'Group', 'Subgroup', 'mape'])
#
#    # Create a Sunburst figure
#    fig = px.sunburst(df, path=['Line', 'Group', 'Subgroup'], values='mape',
#                      color='mape', color_continuous_scale='rdbu')
#
#    # Function to load the PNG file for a sector
#    def load_png_for_sector(hierarchy):
#        # Define the filename for the PNG
#        pre_fix = 'SCR_Regression (M. Chain)_'
#        post_fix = '_Q0_scf_v2_neutral.png'
#        pre_fix = 'fig'
#        post_fix = '.png'
#        filename = os.path.join('plots', 'SCR', 'Regression', 'all_pngs',
#                                f'{pre_fix}{hierarchy}{post_fix}')
#        # Check if the file exists
#        if os.path.exists(filename):
#            # Read the PNG file and convert it to base64
#            with Image.open(filename) as img:
#                img = img.resize((100, 100))  # Resize to an appropriate dimension
#                with img as img_data:
#                    img_data = img_data.tobytes()
#                    encoded_image = base64.b64encode(img_data).decode()
#                # Create an HTML image tag to display the PNG in the hover
#                return f'<img src="data:image/png;base64,{encoded_image}" width="200" height="200">'  # Adjust width and height as needed
#                # return f'<img src="data:image/png;base64,{encoded_image}">'
#        else:
#            return "No image available"
#
#    # Update the hovertemplate to include the image
#    fig.update_traces(hovertemplate='<b>%{label}</b><br>Parent: %{parent}<br>MAPE: %{value}<br>%{customdata}',
#                      customdata=df.apply(lambda row: load_png_for_sector('_'.join(row[:3])), axis=1))
#
#    # Show the figure
#    fig.show()
#
#    # Save the figure as an HTML file
#    fig.write_html(os.path.join('plotly', f'sunburst_{which}.html'))

def test_sunburst():
    for which in ['SCR', 'SCR_LQ', 'SCR_LY']: # ['SCR']: #['SCR', 'SCR_LQ', 'SCR_LY']:
        sunburst_data = load_sunburst_data(which)
        create_scr_sunburst_figure_hierarchy(sunburst_data, which,'rdbu')

if __name__ == '__main__':


        test_sunburst()

        #sunburst_data = [
        #("Sector1_Subsector1_Category1", 0.1, 0.2, 0.3, 0.4),
        #("Sector1_Subsector2_Category1", 0.2, 0.3, 0.4, 0.5),
        ## Add more data
        #]
        #create_scr_sunburst_figure_hierarchy(sunburst_data, "example")

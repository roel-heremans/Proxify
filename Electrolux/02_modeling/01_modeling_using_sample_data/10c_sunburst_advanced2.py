import os
import base64
import io
import pandas as pd
from PIL import Image
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, no_update, callback
from utils.util import load_sunburst_data
import plotly.graph_objects as go


# Helper function to load and convert image to base64
def np_image_to_base64(image_path):
    if os.path.exists(image_path):
        with Image.open(image_path) as img:
            img = img.resize((1370, 504))  # Resize the image
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            encoded_image = base64.b64encode(buffer.getvalue()).decode()
            return "data:image/png;base64," + encoded_image
    return None


# Function to load data and generate sunburst figure
def create_scr_sunburst_figure_hierarchy(sunburst_data, which):
    # Prepare data for Sunburst
    res = []
    for hierarchy, mape, mdape, mape_std, mdape_std, avg_qty_sold in sunburst_data:
        h_split = hierarchy.split("_")
        res.append(h_split + [mape])

    df = pd.DataFrame(res, columns=['Line', 'Group', 'Subgroup', 'mape'])

    # Create the Sunburst figure
    fig = px.sunburst(df, path=['Line', 'Group', 'Subgroup'], values='mape',
                      color='mape', color_continuous_scale='rdbu')

    fig.update_traces(
        hoverinfo="none",  # Disable default hoverinfo
        hovertemplate=None
    )

    return fig, df


# Initialize the Dash app
app = Dash(__name__)

# Call the actual function to load sunburst data
sunburst_data = load_sunburst_data('SCR')

# Create Sunburst figure and dataframe
fig, df = create_scr_sunburst_figure_hierarchy(sunburst_data, 'example')

# Layout of the app
app.layout = html.Div(
    children=[
        dcc.Graph(id="sunburst-graph", figure=fig, clear_on_unhover=True,
                  style={'width': '40%', 'height': '800px'}, ),
        dcc.Tooltip(id="sunburst-tooltip", direction='bottom'),  # Tooltip for image on hover
    ]
)


# Callback to show the image in the tooltip on hover
@callback(
    Output("sunburst-tooltip", "show"),
    Output("sunburst-tooltip", "bbox"),
    Output("sunburst-tooltip", "children"),
    Input("sunburst-graph", "hoverData")
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    # Extract the hover data (hierarchy path)
    hover_data = hoverData["points"][0]

    # Safely get bbox if available, otherwise fallback to a default empty bbox
    bbox = hover_data.get("bbox", {"x0": 1150, "x1": 0, "y0": 200, "y1": 0})

    # Safely extract label and remove ' > ' separator from the hierarchy label
    parent_path = hover_data['parent'].replace("/", "_")  # Replace "/" with "_"
    label_path = f"{parent_path}_{hover_data['label']}"

    # Construct image path
    pre_fix = 'SCR_Regression (M. Chain)_'
    post_fix = '_Q0_scf_v2_neutral.png'
    image_path = os.path.join('plots', 'SCR', 'Regression', 'all_pngs', f'{pre_fix}{label_path}{post_fix}')

    # Debug: Print the constructed image path to the console
    print(f"Constructed image path: {image_path}")

    # Convert the image to base64
    img_url = np_image_to_base64(image_path)

    # Debug: Check if the image URL is successfully created
    if img_url:
        print("Image loaded successfully.")
        # Display image in tooltip if available
        children = [
            html.Div([
                html.Img(
                    src=img_url,
                    style={"width": "900px", "height": "400px", 'display': 'block', 'margin': '0 auto'},
                ),
                html.P(f"Hierarchy: {label_path}", style={'font-weight': 'bold'})
            ])
        ]
    else:
        # Fallback message when the image is not available
        print("No image found at the specified path.")
        children = html.P("Image not available", style={'font-weight': 'bold'})

    return True, bbox, children


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)

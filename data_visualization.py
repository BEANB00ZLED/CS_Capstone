from dash import Dash, html, dcc, dash_table, Input, Output, State,  callback_context, no_update, ctx
import dash_bootstrap_components as dbc
# For data
import pandas as pd
# For graphing
from mapbox_token import TOKEN
import plotly.graph_objects as go
import ast
from util import first_value_loc

FILE = 'waze_jams_I95.csv'

# Initialize the app
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])




def create_alerts_map():
    df = pd.read_csv(FILE)
    #Convert the data from strings to lists of floats
    df['geometry.coordinates'] = df['geometry.coordinates'].apply(ast.literal_eval)
    df['x.coordinates'] = df['geometry.coordinates'].apply(lambda x: first_value_loc(x, 1))
    df['y.coordinates'] = df['geometry.coordinates'].apply(lambda x: first_value_loc(x, 0))
    # Map/data settings
    trace = go.Scattermapbox(
        lat=df['x.coordinates'],
        lon=df['y.coordinates'],
        mode='markers',
        marker=dict(
            color='rgb(255,0,0)',
            opacity=1,
        ),
    )

    # Layout settings
    layout = go.Layout(
        mapbox=dict(
            style="outdoors",
            zoom=5,
            accesstoken=TOKEN,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        uirevision=True,
        autosize=True,
    )

    fig = go.Figure(data=[trace], layout=layout)
    
    return fig

# Main panel ui layout
main_panel = html.Div(
    children=dcc.Tabs(
        id="tabs",
        value='tab_value',
        children=[
            dcc.Tab(
                label='I-95 Jams',
                children=[
                    dcc.Graph(
                        id='jams_graph',
                        figure=create_alerts_map()
                    )
                ]
            )
        ]
    ),
)

app.layout = main_panel

if __name__ == '__main__':
    app.run(debug=True)
    #app.run_server(host= '0.0.0.0',debug=False)
    '''df = pd.read_csv('waze_jams_I95_total.csv')
    #Convert the data from strings to lists of floats
    df['geometry.coordinates'] = df['geometry.coordinates'].apply(ast.literal_eval)
    print(df['geometry.coordinates'].apply(lambda x: first_value_loc(x, 1)))'''
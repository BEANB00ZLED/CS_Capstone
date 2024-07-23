from dash import Dash, html, dcc, dash_table, Input, Output, State,  callback_context, no_update, ctx
import dash_bootstrap_components as dbc
# For data
import pandas as pd
# For graphing
from mapbox_token import TOKEN
import plotly.graph_objects as go
import ast
import seaborn as sns
import matplotlib.pyplot as plt

def first_value_loc(list_of_lists: list, index: int):
    if len(list_of_lists) > 0:
        while isinstance(list_of_lists[0], list):
            list_of_lists = list_of_lists[0]
        return list_of_lists[index]
    else:
        return None

def create_distribution_graphs():
    df = pd.read_csv('basic_thresh_validated_interstates.csv')
    # Get the interstate of the matches and counts
    interstate_counts = df['properties.street'].apply(lambda x: x[0:-2]).value_counts()
    interstate_counts = interstate_counts.divide(interstate_counts.sum()).multiply(100).round(decimals=1)

    interstate_dist = sns.barplot(
        x=interstate_counts.index,
        y=interstate_counts.values
    )
    interstate_dist.grid(visible=True, axis='y')
    interstate_dist.set_xlabel(xlabel='Interstate', fontsize=13)
    interstate_dist.set_ylabel(ylabel='Percentage of Total Data', fontsize=13)
    interstate_dist.set_title(label='Interstate Distribution of Validated Data', fontsize=15)

    plt.show()
    plt.close()
    
    weather_counts = df['WeatherCon'].str.replace('\r\n', '').value_counts().drop(index='UNKNOWN')
    weather_counts = weather_counts.divide(weather_counts.sum()).multiply(100).round(decimals=1)
    weather_dist = sns.barplot(
        x=weather_counts.index,
        y=weather_counts.values
    )
    weather_dist.grid(visible=True, axis='y')
    weather_dist.set_xlabel(xlabel='Weather', fontsize=13)
    weather_dist.set_ylabel(ylabel='Percentage of Total Data', fontsize=13)
    weather_dist.set_title(label='Weather Distribution of Validated Data', fontsize=15)

    plt.show()

    

JAM_FILE = 'waze_jams_interstates.csv'

def create_alerts_map():
    df = pd.read_csv(JAM_FILE)
    #Convert the data from strings to lists of floats
    df['geometry.coordinates'] = df['geometry.coordinates'].apply(ast.literal_eval)
    df['x.coordinates'] = df['geometry.coordinates'].apply(lambda x: first_value_loc(x, 1))
    df['y.coordinates'] = df['geometry.coordinates'].apply(lambda x: first_value_loc(x, 0))

    # Map/data settings
    jams_trace = go.Scattermapbox(
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

    fig = go.Figure(data=jams_trace, layout=layout)
    
    return fig

def create_crashes_map():
    df_crash = pd.read_csv('interstate_crashes.csv')
    df_crash['X'] = df_crash['X'].astype('float32')
    df_crash['Y'] = df_crash['Y'].astype('float32')

    crash_trace = go.Scattermapbox(
        lat=df_crash['Y'],
        lon=df_crash['X'],
        mode='markers',
        marker=dict(
            opacity=1,
        )
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

    fig = go.Figure(data=crash_trace, layout=layout)
    
    return fig

def create_both_map():
    df_crash = pd.read_csv('interstate_crashes.csv')
    df_crash['X'] = df_crash['X'].astype('float32')
    df_crash['Y'] = df_crash['Y'].astype('float32')

    crash_trace = go.Scattermapbox(
        lat=df_crash['Y'],
        lon=df_crash['X'],
        mode='markers',
        marker=dict(
            opacity=0,
        )
    )

    df = pd.read_csv(JAM_FILE)
    #Convert the data from strings to lists of floats
    df['geometry.coordinates'] = df['geometry.coordinates'].apply(ast.literal_eval)
    df['x.coordinates'] = df['geometry.coordinates'].apply(lambda x: first_value_loc(x, 1))
    df['y.coordinates'] = df['geometry.coordinates'].apply(lambda x: first_value_loc(x, 0))

    # Map/data settings
    jams_trace = go.Scattermapbox(
        lat=df['x.coordinates'],
        lon=df['y.coordinates'],
        mode='markers',
        marker=dict(
            color='rgb(255,0,0)',
            opacity=0,
        ),
    )

    # Layout settings
    layout = go.Layout(
        mapbox=dict(
            style="navigation-day",
            zoom=5,
            accesstoken=TOKEN,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        uirevision=True,
        autosize=True,
    )

    fig = go.Figure(data=[crash_trace, jams_trace], layout=layout)
    
    return fig

def run_webapp(test=True):
    # Initialize the app
    app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
    # Main panel ui layout
    main_panel = html.Div(
        children=dcc.Tabs(
            id="tabs",
            value='tab_value',
            children=[
                dcc.Tab(
                    label='Jams',
                    children=[
                        dcc.Graph(
                            id='jams_graph',
                            figure=create_alerts_map(),
                            style={
                                'height': '60rem'
                            }
                        )
                    ],
                ),
                dcc.Tab(
                    label='Crashes',
                    children=[
                        dcc.Graph(
                            id='crash_graph',
                            figure=create_crashes_map(),
                            style={
                                'height': '60rem'
                            }
                        )
                    ]
                ),
                dcc.Tab(
                    label='Both',
                    children=[
                        dcc.Graph(
                            id='both_graph',
                            figure=create_both_map(),
                            style={
                                'height': '60rem'
                            }
                        )
                    ]
                )
            ]
        ),
    )

    app.layout = main_panel

    if test:
        app.run(debug=True)
    else:
        app.run_server(host= '0.0.0.0',debug=False)

def main():
    create_distribution_graphs()

if __name__ == '__main__':
    main()
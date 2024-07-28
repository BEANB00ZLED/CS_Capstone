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
    df_whole_waze = pd.read_csv('waze_jams_interstates.csv')
    # Interstate distribution
    interstate_counts = df['properties.street'].apply(lambda x: x[0:-2]).value_counts()
    interstate_counts = interstate_counts.divide(interstate_counts.sum()).multiply(100).round(decimals=1)
    whole_interstate_counts = df_whole_waze['properties.street'].apply(lambda x: x[0:-2]).value_counts()
    whole_interstate_counts = whole_interstate_counts.divide(whole_interstate_counts.sum()).multiply(100).round(decimals=1)

    interstate_counts = pd.DataFrame({
        'interstate':interstate_counts.index,
        'percentage':interstate_counts.values,
        'Data Type': 'Validated Data'
    })

    whole_interstate_counts = pd.DataFrame({
        'interstate':whole_interstate_counts.index,
        'percentage':whole_interstate_counts.values,
        'Data Type':'Raw Data'
    })
    whole_interstate_counts = whole_interstate_counts[whole_interstate_counts['percentage'] != 0]
    interstate_counts = pd.concat([interstate_counts, whole_interstate_counts], ignore_index=True)
    
    interstate_dist = sns.barplot(
        data=interstate_counts,
        x='interstate',
        y='percentage',
        hue='Data Type'
    )

    interstate_dist.grid(visible=True, axis='y')
    interstate_dist.set_xlabel(xlabel='Interstate', fontsize=13)
    interstate_dist.set_ylabel(ylabel='Percentage of Total Data', fontsize=13)
    interstate_dist.set_title(label='Interstate Distribution of Validated Data', fontsize=15)

    plt.show()
    plt.close()
    
    df_whole_crash = pd.read_csv('interstate_crashes.csv')
    # Weather distribution
    weather_counts = df['WeatherCon'].str.replace('\r\n', '').value_counts().drop(index='UNKNOWN')
    weather_counts = weather_counts.divide(weather_counts.sum()).multiply(100).round(decimals=1)

    whole_weather_counts = df_whole_crash['WeatherCon'].str.replace('\r\n', '').value_counts().drop(index='UNKNOWN')
    whole_weather_counts = whole_weather_counts.divide(whole_weather_counts.sum()).multiply(100).round(decimals=1)
    

    weather_counts = pd.DataFrame({
        'weather':weather_counts.index,
        'percentage':weather_counts.values,
        'Data Type': 'Validated Data'
    })

    whole_weather_counts = pd.DataFrame({
        'weather':whole_weather_counts.index,
        'percentage':whole_weather_counts.values,
        'Data Type':'Raw Data'
    })

    whole_weather_counts = whole_weather_counts[whole_weather_counts['percentage'] > 0.3]
    weather_counts = pd.concat([weather_counts, whole_weather_counts], ignore_index=True)

    weather_dist = sns.barplot(
        data=weather_counts,
        x='weather',
        y='percentage',
        hue='Data Type'
    )
    weather_dist.grid(visible=True, axis='y')
    weather_dist.set_xlabel(xlabel='Weather', fontsize=13)
    weather_dist.set_ylabel(ylabel='Percentage of Total Data', fontsize=13)
    weather_dist.set_title(label='Weather Distribution of Validated Data', fontsize=15)

    plt.show()
    plt.close()

    # Time distribution
    hour_counts = pd.to_datetime(df['CrashTimeF']).dt.hour.value_counts().sort_index()
    hour_counts = hour_counts.divide(hour_counts.sum()).multiply(100).round(decimals=1)

    whole_hour_counts = pd.to_datetime(df_whole_crash['CrashTimeF']).dt.hour.value_counts().sort_index()
    whole_hour_counts = whole_hour_counts.divide(whole_hour_counts.sum()).multiply(100).round(decimals=1)

    hour_counts = pd.DataFrame({
        'hour':hour_counts.index,
        'percentage':hour_counts.values,
        'Data Type': 'Validated Data'
    })

    whole_hour_counts = pd.DataFrame({
        'hour':whole_hour_counts.index,
        'percentage':whole_hour_counts.values,
        'Data Type': 'Raw Data'
    })
    hour_counts = pd.concat([hour_counts, whole_hour_counts], ignore_index=True)

    hour_dist = sns.barplot(
        data=hour_counts,
        x='hour',
        y='percentage',
        hue='Data Type'
    )
    hour_dist.grid(visible=True, axis='y')
    hour_dist.set_xlabel(xlabel='Time of Day [24 hour time]', fontsize=13)
    hour_dist.set_ylabel(ylabel='Percentage of Total Data', fontsize=13)
    hour_dist.set_title(label='Time of Day Distribution of Validated Data', fontsize=15)

    plt.show()
    plt.close()

    

JAM_FILE = 'waze_jams_interstates.csv'

def create_alerts_map():
    df = pd.read_csv(JAM_FILE)
    # Convert the data from strings to lists of floats
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
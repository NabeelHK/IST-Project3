#!/usr/bin/env python
# coding: utf-8

# In[434]:


import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import base64
import json
import geopandas

from sklearn.model_selection import train_test_split

#from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

#import os
from dash.exceptions import PreventUpdate


# In[435]:


external_stylesheets = ["assets/style.css.css"]
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[external_stylesheets,dbc.themes.MATERIA, dbc.icons.FONT_AWESOME],)


# In[436]:


app.title = "CO2_forecast_PT"
server = app.server


# In[437]:


CONTENT_STYLE = {
    "transition": "margin-left .1s",
    "padding": "1rem 1rem",}


# In[438]:


white_text_style = {'color': 'white'}


# In[439]:


with open('IST_Logo.png', 'rb') as img_file:
    image_data = img_file.read()
    encoded_image = base64.b64encode(image_data).decode()


# ### Layout
# #### Sidebar

# In[440]:


sidebar = html.Div(
    [
        html.Div(
            [
                html.Img(src=f'data:image/png;base64,{encoded_image}',style={'height': '70px','width': '150px',}),
            ],
            className="sidebar-header",
        ),
        html.Br(),
        html.Div(style={"border-top": "2px solid white"}),
        html.Br(),
        # nav component
        dbc.Nav(
            [
                dbc.NavLink(
                    [
                        html.I(className="fas fa-solid fa-star me-2"),
                        html.Span("Introduction"),
                    ],
                    href="/",
                    active="exact",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-solid fa-magnifying-glass-chart me-2"),
                        html.Span("Data Analysis"),
                    ],
                    href="/analysis",
                    active="exact",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-solid fa-sliders me-2"),
                        html.Span("Feature Selection"),
                    ],
                    href="/features",
                    active="exact",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-duotone fa-layer-group me-2"),
                        html.Span("Regression Models"),
                    ],
                    href="/regression",
                    active="exact",
                ),
                
                dbc.NavLink(
                    [
                        html.I(className="fas fa-solid fa-arrow-trend-up me-2"),
                        html.Span("C02 Prediction"),
                    ],
                    href="/prediction",
                    active="exact",
                ),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    className="sidebar",
)


# #### Main Window Layout

# In[441]:


app.layout = html.Div(
    [
        dcc.Location(id="url"),
        sidebar,
        html.Div(
            [
                dash.page_container,
            ],
            className="content",
            style=CONTENT_STYLE,
            id="page-content",
        ),
    ]
)


# In[442]:


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return introduction_layout()
    elif pathname == "/analysis":
        return analysis_layout()
    elif pathname == "/features":
        return features_layout()
    elif pathname == "/regression":
        return regression_layout()
    elif pathname == "/prediction":
        return prediction_layout()
    return dbc.Container(
        children=[
            html.H1(
                "404 Error: Page Not found",
                style={"textAlign": "center", "color": "#082446"},
            ),
            html.Br(),
            html.P(
                f"Oh no! The pathname '{pathname}' was not recognised...",
                style={"textAlign": "center"},
            ),
            # image
            html.Div(
                style={"display": "flex", "justifyContent": "center"},
                children=[
                    html.Img(
                        src="https://elephant.art/wp-content/uploads/2020/02/gu_announcement_01-1.jpg",
                        alt="hokie",
                        style={"width": "400px"},
                    ),
                ],
            ),
        ]
    )


# #### Intro Layout

# In[443]:


def introduction_layout():
    layout = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Img(
                                src="https://images.unsplash.com/photo-1614850523425-eec693b15af5?q=80&w=1470&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
                                style={
                                    "width": "100%",
                                    "height": "auto",
                                    "position": "relative",
                                },
                            ),
                        ],
                        style={
                            "height": "200px",
                            "overflow": "hidden",
                            "position": "relative",
                        },
                    ),
                    html.H1(
                        "Introduction",
                        style={
                            "position": "absolute",
                            "top": "80%",
                            "left": "50%",
                            "transform": "translate(-50%, -50%)",
                            "color": "white",
                            "text-align": "center",
                            "width": "100%",
                        },
                    ),
                ],
                style={
                    "position": "relative",
                    "text-align": "center",
                    "color": "white",
                },
            ),
            html.Br(),
            html.Div(
                [
                    html.H3(" WELCOME! "),
                    html.H3(" ⚡ Energy Services ⚡ "),
                    html.H4("IST 1109231, IST 1109180 "),
                    html.H5("Topic : Estimate the CO2 emissions of the electricity consumption of a country for the next day"),
                    html.P("In this tool, we focus on the electricity consumption of Portugal and observe the corresponding CO2 emmissions. We use different analysis methods to analyse the data"),
                ],
                style={"text-align": "center"}
            ),
            html.Br(),
        ]
    )

    return layout


# ### Data

# In[444]:


df_main = pd.read_csv("df_main.csv", index_col=0, parse_dates=True)
columns_to_drop = ['Wind Gust (km/s)', 'Pressure (mbar)']
df_main = df_main.drop(columns=columns_to_drop)
columns = df_main.columns.tolist()
start_date = df_main.index.min()
end_date = df_main.index.max()


# In[445]:


#df_main.index = df_main.index.str.strip()

# Convert the index of df_total to datetime if it's not already in datetime format
df_main.index = pd.to_datetime(df_main.index, format='%d/%m/%Y %H:%M')
df_main = df_main.sort_index()

# Define the cutoff date
test_cutoff_date = '01/01/2024'
test_cutoff_date = pd.to_datetime(test_cutoff_date, format='%d/%m/%Y')

# Split the DataFrame
df_data = df_main[df_main.index < test_cutoff_date]
df_2024 = df_main[df_main.index >= test_cutoff_date]

# Drop NaN values from df_data
df_data = df_data.dropna()


# In[446]:


df_dataFS = df_data.copy()
df_dataFS = df_dataFS.drop("Power (kW)", axis=1)


# In[447]:


# df_real = pd.read_csv('df_real.csv')
# df_real['Date'] = pd.to_datetime(df_real['Date'],format='%d/%m/%Y')
# df_real


# In[448]:


df_meteo_2024 = df_2024.drop('Power (kW)', axis=1)


# In[449]:


#Data for heat map

#Load GIS data (SHP files)
country=geopandas.read_file("concelhos.shp")
#clean portuguese Names
nomes=pd.read_csv('municipalities_names.csv')
country['Municipality']=nomes['NAME']

#Load data from Electricity consumption

electricity_pivot19 = pd.read_csv('ElectricityData_PT_2019.csv')
electricity_pivot20 = pd.read_csv('ElectricityData_PT_2020.csv')
electricity_pivot21 = pd.read_csv('ElectricityData_PT_2021.csv')

#Merge Data using MUNICIPALITY as a KEY
geo_electricity19=pd.merge(country,electricity_pivot19,on='Municipality') 
geo_electricity20=pd.merge(country,electricity_pivot20,on='Municipality') 
geo_electricity21=pd.merge(country,electricity_pivot21,on='Municipality') 

#Rename  data
geo_df19=geo_electricity19
sectors19 = electricity_pivot19.columns

geo_df20=geo_electricity20
sectors20 = electricity_pivot20.columns

geo_df21=geo_electricity21
sectors21 = electricity_pivot21.columns

# Convert the columns in kWh to gWh
columns_to_convert = ['Agriculture', 'CommercialServices', 'Domestic', 'Industry', 'PublicBuildings', 'PublicLighting', 'Total']
geo_df19[columns_to_convert] = geo_df19[columns_to_convert].apply(lambda x: x / 1000000)
geo_df20[columns_to_convert] = geo_df20[columns_to_convert].apply(lambda x: x / 1000000)
geo_df21[columns_to_convert] = geo_df21[columns_to_convert].apply(lambda x: x / 1000000)


# ### Analysis Layout

# In[450]:


def analysis_layout():
    layout = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Img(
                                src="https://images.unsplash.com/photo-1557683316-973673baf926?q=80&w=1129&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
                                style={
                                    "width": "100%",
                                    "height": "auto",
                                    "position": "relative",
                                },
                            ),
                        ],
                        style={
                            "height": "200px",
                            "overflow": "hidden",
                            "position": "relative",
                        },
                    ),
                    html.H1(
                        "Data Visualization",
                        style={
                            "position": "absolute",
                            "top": "80%",
                            "left": "50%",
                            "transform": "translate(-50%, -50%)",
                            "color": "white",
                            "text-align": "center",
                            "width": "100%",
                        },
                    ),
                ],
                style={
                    "position": "relative",
                    "text-align": "center",
                    "color": "white",
                },
            ),
            html.Br(),
            html.Div(
                style={"display": "flex"},
                children=[
                    # tab
                    html.Div(
                        [
                            dbc.Tabs(
                                id="analysis_selected_tab",
                                children=[
                                    dbc.Tab(
                                        label="PT Electricity consumption",
                                        tab_id="electricity_consumption",
                                    ),
                                    dbc.Tab(
                                        label="CO2 Emmissions",
                                        tab_id="CO2_emmissions",
                                    ),
                                ],
                                active_tab="electricity_consumption",
                            ),
                        ]
                    ),
                ],
            ),
            html.Br(),
            # content
            html.Div(
                style={"display": "flex"},
                children=[
                    html.Div(
                        style={
                            "width": "30%",
                            "padding": "10px",
                        },
                        children=[
                            html.Div(id="analysis_tab_content_layout"),
                        ],
                    ),
                    html.Div(
                        style={
                            "width": "70%",
                            "padding": "10px",
                        },
                        children=[
                            html.Div(id="analysis_tab_plot_layout"),
                        ],
                    ),
                ],
            ),
            html.Br(),
            html.Br(),
        ]
    )

    return layout


# In[451]:


@app.callback(
    [
        Output(
            component_id="analysis_tab_content_layout", component_property="children",
        ),
        Output(
            component_id="analysis_tab_plot_layout", component_property="children"
        ),
    ],
    [Input(component_id="analysis_selected_tab", component_property="active_tab")],
)
def render_tab(tab_choice):
    """Renders the selected subtab's layout

    Args:
        tab_choice (str): selected subtab

    Returns:
        selected subtab's layout
    """
    if tab_choice == "electricity_consumption":
        return electricity_consumption()
    if tab_choice == "CO2_emmissions":
        return CO2_emmissions()


# In[452]:


def consumption_content():
    return html.Div(
        [
            html.Div([html.H3("Select Data for Analysis")]),
            html.Div(
                [
                dcc.Dropdown(
                    id='column-dropdown',
                    options=[{'label': i, 'value': i} for i in df_main.columns],
                    value=[df_main.columns[0]],
                    multi=True
                ),
                dcc.DatePickerRange(
                    id='date-picker',
                    min_date_allowed=df_data.index.min(),
                    max_date_allowed=df_data.index.max(),
                    start_date=df_data.index.min(),
                    end_date=df_data.index.max()
                ),
                ]
            ),
        ]
    )


# In[453]:


def consumption_layout():
    layout = html.Div(
        [
            dcc.Loading(
                children=[dcc.Graph(id="analysis_graph")],
            ),
        ]
    )
    return layout


# In[454]:


def generate_analysis_graph(df, columns, start_date, end_date):
    filtered_df = df.loc[start_date:end_date, columns]
    
    # Define a list to hold the y-axis configurations
    y_axis_config = []
    
    # Loop through each column and define a new y-axis configuration
    for i, column in enumerate(columns):
        y_axis_config.append({'title': column, 'overlaying': 'y', 'side': 'right', 'position': i * 0.1})
    
    # Define the data and layout of the figure
    data = [go.Scatter(x=filtered_df.index, y=filtered_df[column], name=column) for column in filtered_df.columns]
    layout = go.Layout(title=', '.join(columns), xaxis_title='Date')
    
    # Update the layout to include the y-axis configurations
    layout.update({'yaxis{}'.format(i + 1): y_axis_config[i] for i in range(len(y_axis_config))})
    
    # Create the figure with the data and layout
    fig = go.Figure(data=data, layout=layout)
    
    return fig


# In[455]:


@app.callback(Output('analysis_graph', 'figure'),
              Input('column-dropdown', 'value'),
              Input('date-picker', 'start_date'),
              Input('date-picker', 'end_date')
)
def update_figure(columns, start_date, end_date):
    
    filtered_df = df_main.loc[start_date:end_date, columns]
    
    # Define a list to hold the y-axis configurations
    y_axis_config = []
    
    # Loop through each column and define a new y-axis configuration
    for i, column in enumerate(columns):
        y_axis_config.append({'overlaying': 'y', 'side': 'right', 'position': 1 - i * 0.1})
    
    # Define the data and layout of the figure
    data = [{'x': filtered_df.index, 'y': filtered_df[column], 'type': 'line', 'name': column} for column in filtered_df.columns]
    layout = {'title': {'text': ', '.join(columns)}, 'xaxis': {'title': 'Date'}}
    
    # Update the layout to include the y-axis configurations
    layout.update({'yaxis{}'.format(i + 1): y_axis_config[i] for i in range(len(y_axis_config))})
    
    # Create the figure with the data and layout
    fig = {'data': data, 'layout': layout}
    
    return fig


# In[456]:


def electricity_consumption():
    return (consumption_content(), consumption_layout())


# #### CO2 tab

# In[457]:


def CO2_content():
    return html.Div(
        [
            html.Div([html.H6("Heat Map of Portugal CO2 Emmisions")]),
            html.Br(),
            dcc.Dropdown(
                    id='year-dropdown',
                    options=[
                        {'label': '2019', 'value': '2019'},
                        {'label': '2020', 'value': '2020'},
                        {'label': '2021', 'value': '2021'}
                    ],
                    value='2019',
                ),
            html.Br(),
            dcc.Dropdown(
                id='sector-dropdown',
                options=[{'label': col, 'value': col} for col in sectors19 if col != 'Municipality'],
                value='Total',
                ),
              
        ]
    )


# In[458]:


def CO2_layout():
    layout = html.Div(
        [
            dcc.Loading(
                children=[dcc.Graph(id="CO2_map")],
            ),
        ]
    )
    return layout


# In[459]:


@app.callback(
    Output('CO2_map', 'figure'),
    Input('year-dropdown', 'value'),
    Input('sector-dropdown', 'value')
    #State('clicks_store', 'data')
)

def update_figure(year,Total):
    if year == '2019':
        figHist = px.choropleth_mapbox(
                geo_df19, geojson=geo_df19.geometry, color='Total',
                locations=geo_df19.index, 
                center={"lat": 39.6, "lon": -7.9}, zoom=6,
                range_color=[0, max(geo_df19['Total'])], 
                mapbox_style="carto-positron",
                width=500, height=800,
                title='Consumption in ' + Total + ' for ' + year + ' in GWh')
        
    elif year == '2020':
        figHist = px.choropleth_mapbox(
                geo_df20, geojson=geo_df19.geometry, color='Total',
                locations=geo_df19.index, 
                center={"lat": 39.6, "lon": -7.9}, zoom=6,
                range_color=[0, max(geo_df20['Total'])], 
                mapbox_style="carto-positron",
                width=500, height=800,
                title='Consumption in ' + Total + ' for ' + year + ' in GWh')
    else:
        figHist = px.choropleth_mapbox(
                geo_df21, geojson=geo_df19.geometry, color='Total',
                locations=geo_df19.index, 
                center={"lat": 39.6, "lon": -7.9}, zoom=6,
                range_color=[0, max(geo_df21['Total'])], 
                mapbox_style="carto-positron",
                width=500, height=800,
                title='Emmissions in ' + Total + ' for ' + year + ' in kg/GWh')

    return figHist


# In[460]:


def CO2_emmissions():
    return (CO2_content(), CO2_layout())


# ### Features Layout

# In[461]:


def features_layout():
    layout = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Img(
                                src="https://images.unsplash.com/photo-1614851099511-773084f6911d?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
                                style={
                                    "width": "100%",
                                    "height": "auto",
                                    "position": "relative",
                                },
                            ),
                        ],
                        style={
                            "height": "200px",
                            "overflow": "hidden",
                            "position": "relative",
                        },
                    ),
                    html.H1(
                        "Features Selection",
                        style={
                            "position": "absolute",
                            "top": "80%",
                            "left": "50%",
                            "transform": "translate(-50%, -50%)",
                            "color": "white",
                            "text-align": "center",
                            "width": "100%",
                        },
                    ),
                ],
                style={
                    "position": "relative",
                    "text-align": "center",
                    "color": "white",
                },
            ),
            html.Br(),
            html.Div(
                style={"display": "flex"},
                children=[
                    # content
                    html.Div(
                        style={
                            "width": "30%",
                            "padding": "10px",
                        },
                        children=[
                            #html.Div(id="features_content"),
                            html.Div([html.H3("Select the features for prediction")]),
                            html.Div(
                            [
                                dcc.Dropdown(
                                    id='feature-dropdown',
                                    options=[{'label': col, 'value': col} for col in df_dataFS.columns],
                                    value=[df_dataFS.columns[0]],
                                    multi=True
                                ),
                                html.Br(),
                                dbc.Button(
                                    "Select Features",
                                    color="success",
                                    className="me-1",
                                    id="split-button",
                                    n_clicks=0,
                                ),
                                html.Br(),
                                html.Div(id='split-values'),
                                    ]
                                ),
                        ],
                    ),
                    html.Div(
                        style={
                            "width": "70%",
                            "padding": "10px",
                        },
                        children=[
                            #html.Div(id="features_table"),
                            #dcc.Loading(
                                #children=[
                                    html.Div(id='feature-table-div'),
                                    #html.Div(id='split-values'),
                                    html.Div([
                                        html.H6(""),
                                        html.Pre(id="x-values", style=white_text_style)
                                    ]),
                                    html.Div([
                                        html.H6(""),
                                        html.Pre(id="y-values", style=white_text_style)
                                    ]),
                            html.Div([
                            html.H6(""),
                            html.Pre(id="x-2019-values", style=white_text_style)
                            ]),
                             #   ],
                           # ),
                        ],
                    ),
                ],
            ),    
            html.Br(),
            html.Br(),
        ]
    )

    return layout


# In[462]:


def generate_table(dataframe, max_rows=10):
    # Apply some CSS styles to the table
    table_style = {
        'borderCollapse': 'collapse',
        'borderSpacing': '0',
        'width': '100%',
        'border': '1px solid #ddd',
        'fontFamily': 'Arial, sans-serif',
        'fontSize': '14px'
    }
    
    th_style = {
        'border': '1px solid #ddd',
        'padding': '8px',
        'textAlign': 'left',
        'backgroundColor': '#f2f2f2',
        'fontWeight': 'bold',
        'color': '#333'
    }
    
    td_style = {
        'border': '1px solid #ddd',
        'padding': '8px',
        'textAlign': 'left'
    }
    
    return html.Table(
        # Apply the table style
        style=table_style,
        children=[
            # Add the table header
            html.Thead(
                html.Tr([
                    html.Th('Index', style=th_style),
                    *[html.Th(col, style=th_style) for col in dataframe.columns]
                ])
            ),
            # Add the table body
            html.Tbody([
                html.Tr([
                    html.Td(dataframe.index[i], style=td_style),
                    *[html.Td(dataframe.iloc[i][col], style=td_style) for col in dataframe.columns]
                ])
                for i in range(min(len(dataframe), max_rows))
            ])
        ]
    )


# In[463]:


@app.callback(
    Output('feature-table-div', 'children'),
    Input('feature-dropdown', 'value')
)
def update_feature_table(selected_features):
    if selected_features:
        global df_model
        df_model = df_dataFS[selected_features]
        table = generate_table(df_model)
        return table
    else:
        return html.Div()


# In[464]:


@app.callback(
    Output('x-values', 'children'),
    Output('y-values', 'children'),
    Output('x-2019-values', 'children'),
    Input('feature-dropdown', 'value')
)
def update_x_y(selected_features):
    global X, Y, X_2019
    if selected_features:
        X = df_model.iloc[:, :].values
        Y = df_data.loc[:, 'Power (kW)'].values
        X_2019 = df_meteo_2024[selected_features].values
        return str(X), str(Y), str(X_2019)
    else:
        return "", ""


# In[465]:


@app.callback(
    Output('split-values', 'children'),
    Input('split-button', 'n_clicks')
)
def generate_train_test_split(n_clicks):
    global X_train, X_test, y_train, y_test
    if n_clicks:
        X_train, X_test, y_train, y_test = train_test_split(X, Y)
        return html.Span('Done!', style={'fontWeight': 'bold', 'color': 'green'})
    else:
        return ""


# ### Regression Layout

# In[466]:


def regression_layout():
    layout = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Img(
                                src="https://images.unsplash.com/photo-1614851099511-773084f6911d?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
                                style={
                                    "width": "100%",
                                    "height": "auto",
                                    "position": "relative",
                                },
                            ),
                        ],
                        style={
                            "height": "200px",
                            "overflow": "hidden",
                            "position": "relative",
                        },
                    ),
                    html.H1(
                        "Regression Models",
                        style={
                            "position": "absolute",
                            "top": "80%",
                            "left": "50%",
                            "transform": "translate(-50%, -50%)",
                            "color": "white",
                            "text-align": "center",
                            "width": "100%",
                        },
                    ),
                ],
                style={
                    "position": "relative",
                    "text-align": "center",
                    "color": "white",
                },
            ),
            html.Br(),
            html.Div(
                style={"display": "flex"},
                children=[
                    # content
                    html.Div(
                        style={
                            "width": "30%",
                            "padding": "10px",
                        },
                        children=[
                            #html.Div(id="features_content"),
                            html.Div([html.H3("Select the Training Model")]),
                            html.Div(
                            [
                                dcc.Dropdown(
                                    id='model-dropdown',
                                    options=[
                                        # {'label': 'Linear Regression', 'value': 'linear'},
                                        # {'label': 'Random Forests', 'value': 'random_forests'},
                                        {'label': 'Bootstrapping', 'value': 'bootstrapping'},
                                {'label': 'Decision Tree Regressor', 'value': 'decision_trees'}
                                    ],
                                value='decision_trees'
                                ),
                                html.Br(),
                                dbc.Button(
                                    "Train Model",
                                    color="success",
                                    className="me-1",
                                    id="train-model-button",
                                    n_clicks=0,
                                ),
                                html.Br(),
                                    ]
                                ),
                        ],
                    ),
                    html.Div(
                        style={
                            "width": "70%",
                            "padding": "10px",
                        },
                        children=[
                            html.Div(id="regression"),
                            dcc.Loading(
                                children=[
                                    html.Div([dcc.Graph(id="lr-graph")])
                                ],
                            ),
                        ],
                    ),
                ],
            ),    
            html.Br(),
            html.Br(),
        ]
    )

    return layout


# In[467]:


# define global variables
y_pred_list = []
y_pred_2019 = []
y_pred_CO2 = []

X = None
Y = None

X_train = None
X_test = None
y_train = None
y_test = None

X_2019 = None


# In[468]:


@app.callback(
    Output('lr-graph', 'figure'),
    Input('train-model-button', 'n_clicks'),
    State('model-dropdown', 'value')
)
def train_and_predict(n_clicks, model_type):
    global y_pred_list, y_pred_2019, y_pred_CO2  # access global variable

    if n_clicks == 0:
        return dash.no_update 
    else:
        # if model_type == 'linear':
        #     from sklearn import linear_model
            
        #     # Create linear regression object
        #     model = linear_model.LinearRegression()

        #     # Train the model using the training sets
        #     model.fit(X_train, y_train)

        #     #Save the trained model
        #     with open('model.pkl', 'wb') as file:
        #       pickle.dump(model, file)
        #       file.close()

        #     y_pred = model.predict(X_test)
        #     y_pred_list.append(y_pred)
            
        #     y_pred2019 = model.predict(X_2019)
        #     y_pred_2019 = y_pred2019

        #     # Generate scatter plot of predicted vs actual values
        #     fig = go.Figure()
        #     fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers'))
        #     fig.update_layout(title='Linear Regression Predictions',title_x=0.5,title_y=0.9)
        #     return fig
        # elif model_type == 'random_forests':            
        #     parameters = {'bootstrap': True,
        #                   'min_samples_leaf': 3,
        #                   'n_estimators': 200, 
        #                   'min_samples_split': 15,
        #                   'max_features': 'sqrt',
        #                   'max_depth': 20,
        #                   'max_leaf_nodes': None}
            
        #     # Create random forests model object
        #     model = RandomForestRegressor(**parameters)

        #     # Train the model using the training sets
        #     model.fit(X_train, y_train)

        #     # Save the trained model
        #     with open('model.pkl', 'wb') as file:
        #         pickle.dump(model, file)
        #         file.close()

        #     y_pred = model.predict(X_test)
        #     y_pred_list.append(y_pred)
            
        #     y_pred2019 = model.predict(X_2019)
        #     y_pred_2019 = y_pred2019

        #     # Generate scatter plot of predicted vs actual values
        #     fig = go.Figure()
        #     fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers'))
        #     fig.update_layout(title='Random Forests Predictions',title_x=0.5,title_y=0.9)
        #     return fig
        
        if model_type == 'bootstrapping':
            
            model = BaggingRegressor()
            model.fit(X_train, y_train)

            # Save the trained model
            with open('model.pkl', 'wb') as file:
                pickle.dump(model, file)
                file.close()

            y_pred = model.predict(X_test)
            y_pred_list.append(y_pred)
            
            y_pred2019 = model.predict(X_2019)
            y_pred_2019 = y_pred2019
            y_pred_CO2 = y_pred_2019 * 0.9175
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers'))
            fig.update_layout(title='Bootstrapping Predictions',title_x=0.5,title_y=0.9)
            return fig
        
        elif model_type == 'decision_trees':
            model = DecisionTreeRegressor() 
            model.fit(X_train, y_train)

            # Save the trained model
            with open('model.pkl', 'wb') as file:
                pickle.dump(model, file)
                file.close()

            y_pred = model.predict(X_test)
            y_pred_list.append(y_pred)
            
            y_pred2019 = model.predict(X_2019)
            y_pred_2019 = y_pred2019
            y_pred_CO2 = y_pred_2019 * 0.9175
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers'))
            fig.update_layout(title='Decision Tree Regressor Predictions',title_x=0.5,title_y=0.9)
            return fig


# ### Prediction Layout

# In[469]:


def prediction_layout():
    layout = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Img(
                                src="https://images.unsplash.com/photo-1614851099511-773084f6911d?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
                                style={
                                    "width": "100%",
                                    "height": "auto",
                                    "position": "relative",
                                },
                            ),
                        ],
                        style={
                            "height": "200px",
                            "overflow": "hidden",
                            "position": "relative",
                        },
                    ),
                    html.H1(
                        "CO2 Prediction",
                        style={
                            "position": "absolute",
                            "top": "80%",
                            "left": "50%",
                            "transform": "translate(-50%, -50%)",
                            "color": "white",
                            "text-align": "center",
                            "width": "100%",
                        },
                    ),
                ],
                style={
                    "position": "relative",
                    "text-align": "center",
                    "color": "white",
                },
            ),
            html.Br(),
            html.Div(
                style={"display": "flex"},
                children=[
                    # content
                    html.Div(
                        style={
                            "width": "20%",
                            "padding": "10px",
                        },
                        children=[
                            #html.Div(id="features_content"),
                            #html.Div([html.H3("Select the Training Model")]),
                            
                                html.Br(),
                                dbc.Button(
                                    "Predict Values",
                                    color="success",
                                    className="me-1",
                                    id="button_model",
                                    n_clicks=0,
                                ),
                                #dcc.Store(id='clicks_store', data=0),
                                html.Br(),
                                    ]
                                ),
                    html.Div(
                        style={
                            "width": "80%",
                            "padding": "10px",
                        },
                        children=[
                            html.Div(id="predict"),
                            dcc.Loading(
                                children=[
                                    html.Div([dcc.Graph(id="predict-graph")])
                                ],
                            ),
                        ],
                    ),
                ],
            ),    
            html.Br(),
            html.Br(),
        ]
    )

    return layout


# In[470]:


df_CO2 = pd.DataFrame(index=df_data.index).assign(CO2_emmisions=df_data['Power (kW)'] * 0.9175)


# In[471]:


@app.callback(
    Output('predict-graph', 'figure'),
    Input('button_model', 'n_clicks'),
    #State('clicks_store', 'data')
)
def run_model(n_clicks):
    if n_clicks == 0:
        raise PreventUpdate
    # elif n_clicks_store == 0:
    #     return {}
    else:
        # if 'Date' in df_data.columns:
        #     df_data['Date'] = pd.to_datetime(df_data['Date'])
        #     df_data.set_index('Date', inplace=True) 
            
        # fig = go.Figure(layout=go.Layout(title='Real & Predicted Power Consumption'))
        # fig.add_scatter(x=df_data.index, y=df_data['Power (kW)'], name='Real Power (kW)')
        # fig.add_scatter(x=df_2024.index, y=y_pred_2019, name='Predicted Power (kW)')   

        fig = go.Figure(layout=go.Layout(title='Real & Predicted CO2 emmisions'))
        fig.add_scatter(x=df_CO2.index, y=df_CO2['CO2_emmisions'], name='Real CO2 Emmissions(kg/kWh)')
        fig.add_scatter(x=df_2024.index, y=y_pred_CO2, name='Predicted CO2 Emmissions(kg/kWh)')     
        return fig


# In[472]:


if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port=8011,debug = True)


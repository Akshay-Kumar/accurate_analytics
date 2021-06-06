from pathlib import Path
import dash_daq as daq
import pandas as pd
import dash
import dash_table as dt
import plotly.express as px
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from music_prediction_app.analysis import pop_rf_regressor, predict_features
from music_prediction_app.helper_lib import get_numeric_columns, get_agg_song_props_by_artist, plotly_scatter_plot, \
    plotly_histogram_plot, plotly_line_chart, plotly_area_plot, plotly_bar_chart, read_dataset
from music_prediction_app.theme_river import create_theme_river


def spotify_db_app(df: pd.DataFrame):
    """
    :return app:
    This method generates the layout of the Dash app and returns a reference of the Dash object
    """
    ##########################################
    # Visualizations/Table Tabs Implementation
    ##########################################
    df['index'] = range(1, len(df) + 1)
    PAGE_SIZE = 5
    all_columns = df.columns.tolist()
    default_columns = all_columns[0: 5]
    numeric_columns = get_numeric_columns(df=df)
    feature_for_visuals = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness',
                           'valence', 'popularity']
    graph_type = ['bar-chart', 'scatter-plot', 'line-chart', 'area-plot', 'histogram-plot']

    # Set up animation dashboard.
    # Animation figure uses different dataset with labels/values not optimized for ML, but good for visualization.
    animation_data = pd.read_csv(Path('../data/data_animation_fig.csv'))

    # Values must be sort in terms of the animation column.
    animation_data = animation_data.sort_values('Year')

    # Set dropdown options for animation figure.
    color_options = [dict(label=col, value=i) for i, col in enumerate(animation_data.columns)]
    size_options = [dict(label=col, value=i) for i, col in enumerate(get_numeric_columns(animation_data))]

    ################################
    # Predictions Tab Implementation
    ################################
    # Define columns to train for popularity.
    regression_cols = ["acousticness", "danceability", "energy", "instrumentalness", "key",
                       "mode", "speechiness", "year", "valence", "popularity"]

    # Generate Model using selected columns. Duplicate entries (due to multiple artists) are removed for model training.
    pop_rfr_model = pop_rf_regressor(df.drop_duplicates(subset=['id']), regression_cols)

    # Slider Marks for continuous features.
    slider_marks = {0: {'label': '0', 'style': {'color': '#77b0b1'}},
                    0.2: {'label': '0.2'}, 0.4: {'label': '0.4'}, 0.6: {'label': '0.6'}, 0.8: {'label': '0.8'},
                    1: {'label': '1.0', 'style': {'color': '#f50'}}}

    # Toggle Switch styling. Defined separately because it is used frequently in the UI.
    switch_color = '#119DFF'
    switch_div_style = {'margin-left': '45%'}

    # Slider styling. Defined separately because it is used frequently in the UI.
    label_style = {'font-weight': 'bold', 'text-align': 'center', 'display': 'block'}
    slider_height = 350
    slider_step = 0.001

    # Mode/Key options for drop downs.
    keys = {0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F', 6: 'F#', 7: 'G',
            8: 'G#', 9: 'A', 10: 'A#', 11: 'B'}
    key_options = [dict(label=val, value=key) for key, val in keys.items()]
    mode_options = [{'label': 'Minor', 'value': 0}, {'label': 'Major', 'value': 1}]

    #########################
    # Create app, define UI #
    #########################
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
    app.layout = dbc.Container([
        html.Center([html.H2(id='page-title', children='An Accurate Analytics App', style={"margin-bottom": '1em'})]),
        dcc.Tabs(id="tabs", value='tab-1', children=[
            dcc.Tab(label='Dataset Exploration', value='tab-1', children=[
                dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            dbc.FormGroup([
                                html.Label('Select features'),
                                dcc.Dropdown(
                                    id='data-dropdown',
                                    options=[{'label': k, 'value': k} for k in all_columns],
                                    multi=True,
                                    value=default_columns
                                )])]),
                        dbc.Col([
                            dbc.FormGroup([
                                html.Label('Records per page'),
                                html.Br(),
                                dcc.Slider(
                                    id='records-per-page-slider',
                                    min=0,
                                    max=50,
                                    value=PAGE_SIZE,
                                    step=5,
                                    marks={
                                        5: {'label': '5', 'style': {'color': '#77b0b1'}},
                                        10: {'label': '10'},
                                        15: {'label': '15'},
                                        20: {'label': '20'},
                                        25: {'label': '25'},
                                        30: {'label': '30'},
                                        35: {'label': '35'},
                                        40: {'label': '40'},
                                        45: {'label': '45'},
                                        50: {'label': '50', 'style': {'color': '#f50'}}
                                    }
                                )
                            ])]),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Get data", id='data-button', color="primary"),
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            dcc.Loading(
                                id="loading-element",
                                children=[
                                    html.Div([
                                        dt.DataTable(
                                            id='data-table',
                                            page_current=0,
                                            page_size=PAGE_SIZE,
                                            page_action='custom',
                                            sort_action='custom',
                                            sort_mode='multi',
                                            sort_by=[],
                                            style_cell_conditional=[{'textAlign': 'left'}],
                                            style_data_conditional=[{
                                                'if': {'row_index': 'odd'},
                                                'backgroundColor': 'rgb(248, 248, 248)'
                                            }],
                                            style_header={
                                                'backgroundColor': 'rgb(230, 230, 230)',
                                                'fontWeight': 'bold'
                                            }
                                        )
                                    ])],
                                type="circle",
                            )
                        ])
                    ])
                ])
            ]),
            dcc.Tab(label='Visualizations by Artist', value='tab-2', children=[
                dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            dbc.FormGroup([
                                html.Label('Select features for Y-axis'),
                                dcc.Dropdown(
                                    id='data-dropdown-2',
                                    options=[{'label': k, 'value': k} for k in feature_for_visuals],
                                    value=[feature_for_visuals[0]],
                                    multi=True,
                                )
                            ])
                        ]),
                        dbc.Col([
                            dbc.FormGroup([
                                dbc.Label("Choose graph type"),
                                dcc.Dropdown(id="graph-dropdown-2",
                                             options=[{'label': k, 'value': k} for k in graph_type],
                                             value='histogram-plot'),
                            ])]),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.FormGroup([
                                html.Label('Record limit'),
                                dcc.RangeSlider(
                                    id='record-limiter-slider',
                                    min=0,
                                    max=df.shape[0],
                                    value=[0, 100],
                                    marks={
                                        0: {'label': '0', 'style': {'color': '#77b0b1'}},
                                        25000: {'label': '25000'},
                                        50000: {'label': '50000'},
                                        75000: {'label': '75000'},
                                        100000: {'label': '100000'},
                                        125000: {'label': '125000'},
                                        150000: {'label': '150000'},
                                        175000: {'label': '175000'},
                                        200000: {'label': '200000'},
                                        df.shape[0]: {'label': str(df.shape[0]), 'style': {'color': '#f50'}},
                                    }
                                ),
                            ])]),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Loading(
                                id="loading-element-2",
                                children=[html.Div([
                                    dcc.Graph(id='spotify-graph')
                                ])],
                                type="circle",
                            ),
                        ]),
                    ]),
                ])
            ]),

            # Theme River Tab
            dcc.Tab(label='Trends', value='tab-3', children=[
                html.H1(children='Theme River', style={"margin-top": '2em'}),
                html.H6(children='Feature comparison of 20 most popular artists.'),
                dbc.Row(dbc.Col(
                    dcc.Graph(figure=create_theme_river()),
                )),

                # Animation dashboard.
                html.H1(children='Through the Years...', style={"margin-top": '2em'}),
                html.Div(children='Spotify Dataset'),
                html.Hr(),

                # Visualization
                dbc.Row([dbc.Col(
                    dcc.Loading(dcc.Graph(id='animation_fig'))
                )]),
                dbc.Row([
                    dbc.Card([  # Card to display number of rows in df.
                        dbc.CardBody([html.H4(id='num_rows_card', className='card-title')])])
                ]),

                # Drop downs for user selection.
                dbc.Row([
                    dbc.Col([
                        dbc.FormGroup([
                            dbc.Label("Select Color"),
                            dcc.Dropdown(id="color_drop", value=0, options=color_options)
                        ]),
                        dbc.FormGroup([
                            dbc.Label("Select Size"),
                            dcc.Dropdown(id="size_drop", value=1, options=size_options)
                        ])
                    ]),
                    dbc.Col([
                        dbc.FormGroup([
                            dbc.Label("Select X"),
                            dcc.Dropdown(id="x_drop", value=2, options=color_options)
                        ]),
                        dbc.FormGroup([
                            dbc.Label("Select Y"),
                            dcc.Dropdown(id="y_drop", value=3, options=color_options)
                        ])
                    ]),
                ], style={'margin-top': '1em'}),

                # Button to update based on selection.
                dbc.Button('Update Graph!', id='update_button', color='primary', style={'margin-bottom': '2em'},
                           block=True),
            ]),

            # Prediction Tab
            dcc.Tab(label='Music Popularity Prediction', value='tab-4', children=[
                html.H1(children='Popularity and Feature Predictor', style={'margin-top': '1em'}),
                html.Div(children='Spotify Dataset'),
                html.Hr(),

                # Define feature sliders and corresponding switches.
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Acousticness", html_for="acousticness_slider", style=label_style),
                        html.Div([
                            dcc.Slider(id='acousticness_slider', min=0, max=1, value=0.5, step=slider_step,
                                       vertical=True,
                                       verticalHeight=slider_height, marks=slider_marks)], style=switch_div_style),
                        daq.BooleanSwitch(id='a_switch', on=True, color=switch_color)
                    ]),
                    dbc.Col([
                        dbc.Label("Danceability", html_for="danceability_slider", style=label_style),
                        html.Div([
                            dcc.Slider(id='danceability_slider', min=0, max=1, value=0.5, step=slider_step,
                                       vertical=True,
                                       verticalHeight=slider_height, marks=slider_marks)], style=switch_div_style),
                        daq.BooleanSwitch(id='d_switch', on=True, color=switch_color)
                    ]),
                    dbc.Col([
                        dbc.Label("Energy", html_for="energy_slider", style=label_style),
                        html.Div([
                            dcc.Slider(id='energy_slider', min=0, max=1, value=0.5, step=slider_step, vertical=True,
                                       verticalHeight=slider_height, marks=slider_marks)], style=switch_div_style),
                        daq.BooleanSwitch(id='e_switch', on=True, color=switch_color)
                    ]),
                    dbc.Col([
                        dbc.Label("Instrumentalness", html_for="instrumentalness_slider", style=label_style),
                        html.Div([
                            dcc.Slider(id='instrumentalness_slider', min=0, max=1, value=0.5, step=slider_step,
                                       vertical=True,
                                       verticalHeight=slider_height, marks=slider_marks)], style=switch_div_style),
                        daq.BooleanSwitch(id='i_switch', on=True, color=switch_color)
                    ]),
                    dbc.Col([
                        dbc.Label("Speechiness", html_for="speechiness_slider",
                                  style={'font-weight': 'bold', 'display': 'block', 'margin-left': '25%'}),
                        html.Div([
                            dcc.Slider(id='speechiness_slider', min=0, max=1, value=0.5, step=slider_step,
                                       vertical=True,
                                       verticalHeight=slider_height, marks=slider_marks)], style=switch_div_style),
                        daq.BooleanSwitch(id='s_switch', on=True, color=switch_color)
                    ]),
                    dbc.Col([
                        dbc.Label("Valence", html_for="valence_slider", style=label_style),
                        html.Div([
                            dcc.Slider(id='valence_slider', min=0, max=1, value=0.5, step=slider_step, vertical=True,
                                       verticalHeight=slider_height, marks=slider_marks)], style=switch_div_style),
                        daq.BooleanSwitch(id='v_switch', on=True, color=switch_color)
                    ]),
                ], style={'margin-bottom': '2em'}),

                # Additional feature selection.
                dbc.Row([
                    dbc.Col(
                        dbc.FormGroup([
                            dbc.Label("Select Key"),
                            dcc.Dropdown(id="key_drop", value=0, options=key_options)
                        ])),
                    dbc.Col(
                        dbc.FormGroup([
                            dbc.Label("Select Mode"),
                            dcc.Dropdown(id="mode_drop", value=0, options=mode_options)
                        ])),
                    dbc.Col(
                        dbc.FormGroup([
                            dbc.Label("Select Year", style={'margin-right': '1em'}),
                            dbc.Input(id="year_input", type='number', placeholder="2020", value=2020, min=0)
                        ]))
                ]),
                dbc.Button('Optimize!', id='optimize_button', color='success', style={'margin-bottom': '2em'},
                           block=True),

                # Create cards for output of predictions.
                html.Hr(),
                dbc.Collapse(dcc.Loading([  # Wrapped in collapse to hide if no prediction made.
                    html.H3(children='Predictions', className='card-title'),
                    html.H6(id='predicted_popularity', className='card-text'),
                    html.P(id='year_avg', className='card-text', style={'font-style': 'italic'}),
                    dbc.Row([
                        dbc.Col(dbc.Card([
                            dbc.CardHeader(id='prediction_a', className='card-title'),
                            dbc.CardBody([html.P('''Measure indicating that a song is acoustic - 
                                      it does not contain electrical amplification. 
                                      Songs with more acoustic elements score closer to 1.0''')
                                          ])
                        ], color="danger", inverse=True)),
                        dbc.Col(dbc.Card([
                            dbc.CardHeader(id='prediction_d', className='card-title'),
                            dbc.CardBody([html.P('''Measure describes how suitable a track is for dancing 
                                      based on a combination of musical elements including tempo, rhythm stability, 
                                      beat strength, and overall regularity.''')
                                          ])
                        ], color="primary", inverse=True)),
                        dbc.Col(dbc.Card([
                            dbc.CardHeader(id='prediction_e', className='card-title'),
                            dbc.CardBody([html.P('''Represents a perceptual measure of intensity and activity. 
                                      Typically, energetic tracks feel fast, loud, and noisy.''')
                                          ])
                        ], color="success", inverse=True)),
                    ], style={'margin-top': '1em', 'margin-bottom': '2em'}),
                    dbc.Row([
                        dbc.Col(dbc.Card([
                            dbc.CardHeader(id='prediction_i', className='card-title'),
                            dbc.CardBody([html.P('''Measure indicating whether the song is an instrumental - 
                                      it does not contain vocal content.''')
                                          ])
                        ], color="secondary")),
                        dbc.Col(dbc.Card([
                            dbc.CardHeader(id='prediction_s', className='card-title'),
                            dbc.CardBody([html.P('''Indicates the presence of spoken words in the song. Mid-range values 
                                      indicate both music and speech (e.g. rap music), 
                                      lower values represent non-speech-like tracks.''')
                                          ])
                        ], color="info", inverse=True)),
                        dbc.Col(dbc.Card([
                            dbc.CardHeader(id='prediction_v', className='card-title'),
                            dbc.CardBody([html.P('''Describes the musical positiveness conveyed by a track. Tracks with 
                                      high valence sound more positive, tracks with low valence sound more negative.''')
                                          ])
                        ], color="warning", inverse=True)),
                    ], style={'margin-bottom': '2em'}),
                    dbc.Row([
                        dbc.Col(dbc.Card([
                            dbc.CardHeader(id='prediction_k', className='card-title'),
                            dbc.CardBody([html.P('''Indicates the overall key of a song.''')
                                          ])
                        ], color="light")),
                        dbc.Col(dbc.Card([
                            dbc.CardHeader(id='prediction_m', className='card-title'),
                            dbc.CardBody([html.P('''Indicates the modality of a song - Major or Minor. 
                                      ''')
                                          ])
                        ], color="dark", inverse=True))
                    ]),
                ]),
                    id='predictions_collapse', is_open=False),

                html.Hr(),
            ])
        ]),
    ])

    ##############################
    # Callbacks for visualizations
    ##############################
    @app.callback(
        [Output('data-table', 'data'),
         Output('data-table', 'columns'),
         Output('data-table', "page_size")],
        [Input('data-button', 'n_clicks'),
         Input('data-table', "page_current"),
         Input('data-table', "sort_by"),
         Input('records-per-page-slider', 'value')],
        [State('data-dropdown', 'value')],
        prevent_initial_call=True
    )
    def update_dashboard(_, page_current, sort_by, row_limit, selected_features):
        output = []
        try:
            df_selected = df[selected_features]
            df_selected = df_selected.iloc[page_current * row_limit:(page_current + 1) * row_limit]

            # select specific features
            if len(sort_by):
                dff = df_selected.sort_values(
                    [col['column_id'] for col in sort_by],
                    ascending=[
                        col['direction'] == 'asc'
                        for col in sort_by
                    ],
                    inplace=False
                )
            else:
                # No sort is applied
                dff = df_selected

            column_list = [{'id': i, 'name': i} for i in selected_features]
            output = [dff.to_dict('records'), column_list, row_limit]
        except Exception as ex:
            print("Exception : {}".format(ex))
        return output

    @app.callback(
        [Output('spotify-graph', 'figure')],
        [Input('graph-dropdown-2', 'value'),
         Input('record-limiter-slider', 'value'),
         Input('data-dropdown-2', 'value')],
        prevent_initial_call=True
    )
    def update_figure(graph_type_value, row_limit, selected_y_features):
        output = []

        df_dataset = get_agg_song_props_by_artist(df=df, columns=selected_y_features, row_limit=row_limit)
        x_column_name = df_dataset.index
        y_column_name = selected_y_features

        if graph_type_value == 'scatter-plot':
            output = [plotly_scatter_plot(df_dataset, x_column_name, y_column_name)]
        elif graph_type_value == 'histogram-plot':
            output = [plotly_histogram_plot(df_dataset, x_column_name, y_column_name)]
        elif graph_type_value == 'line-chart':
            output = [plotly_line_chart(df_dataset, x_column_name, y_column_name)]
        elif graph_type_value == 'area-plot':
            output = [plotly_area_plot(df_dataset, x_column_name, y_column_name)]
        elif graph_type_value == 'bar-chart':
            output = [plotly_bar_chart(df_dataset, x_column_name, y_column_name)]
        return output

    # Callbacks for Theme River Tab animation figure.
    @app.callback(
        [Output('animation_fig', 'figure'), Output('num_rows_card', 'children')],
        [Input('update_button', 'n_clicks')],
        [State('color_drop', 'value'),
         State('size_drop', 'value'),
         State('x_drop', 'value'),
         State('y_drop', 'value')])
    def update_animation_fig(_, color_val, size_val, x_val, y_val):
        # Get list of columns.
        columns = list(animation_data.columns)

        # Set the number of rows for the card.
        num_rows_text = f"Number of Rows in Dataset: {len(animation_data)}"

        # Set X and Y range depending on max and min values.
        x_range = [animation_data[columns[x_val]].min(), animation_data[columns[x_val]].max()]
        y_range = [animation_data[columns[y_val]].min(), animation_data[columns[y_val]].max()]

        # Return chart based on selected type and x/y values.
        fig = px.scatter(animation_data, x=columns[x_val], y=columns[y_val],
                         animation_frame='Year',
                         size=columns[size_val],
                         color=columns[color_val],
                         hover_name='Name',
                         range_x=x_range,
                         range_y=y_range,
                         size_max=55)

        return fig, num_rows_text

    # Prevent from updating figure without values.
    @app.callback(Output('update_button', 'disabled'),
                  [Input('color_drop', 'value'), Input('size_drop', 'value'),
                   Input('x_drop', 'value'), Input('y_drop', 'value')])
    def enable_button(color_val, size_val, x_val, y_val):
        return False if not [x for x in (color_val, size_val, x_val, y_val) if x is None] else True

    #####################################
    # Callbacks for music prediction tab.
    #####################################

    # Callbacks for de/activating all sliders via BooleanSwitches.
    @app.callback([Output('acousticness_slider', 'disabled'), Output('acousticness_slider', 'value')],
                  [Input('a_switch', 'on')])
    def disable_acousticness(switch_val):
        return (False, 0.5) if switch_val else (True, None)

    @app.callback([Output('danceability_slider', 'disabled'), Output('danceability_slider', 'value')],
                  [Input('d_switch', 'on')])
    def disable_danceability(switch_val):
        return (False, 0.5) if switch_val else (True, None)

    @app.callback([Output('energy_slider', 'disabled'), Output('energy_slider', 'value')],
                  [Input('e_switch', 'on')])
    def disable_energy(switch_val):
        return (False, 0.5) if switch_val else (True, None)

    @app.callback([Output('instrumentalness_slider', 'disabled'), Output('instrumentalness_slider', 'value')],
                  [Input('i_switch', 'on')])
    def disable_instrumentalness(switch_val):
        return (False, 0.5) if switch_val else (True, None)

    @app.callback([Output('speechiness_slider', 'disabled'), Output('speechiness_slider', 'value')],
                  [Input('s_switch', 'on')])
    def disable_speechiness(switch_val):
        return (False, 0.5) if switch_val else (True, None)

    @app.callback([Output('valence_slider', 'disabled'), Output('valence_slider', 'value')],
                  [Input('v_switch', 'on')])
    def disable_valence(switch_val):
        return (False, 0.5) if switch_val else (True, None)

    # Callback to run machine learning prediction.
    @app.callback(
        [Output('predictions_collapse', 'is_open'),
         Output('predicted_popularity', 'children'),
         Output('year_avg', 'children'),
         Output('prediction_a', 'children'),
         Output('prediction_d', 'children'),
         Output('prediction_e', 'children'),
         Output('prediction_i', 'children'),
         Output('prediction_s', 'children'),
         Output('prediction_v', 'children'),
         Output('prediction_k', 'children'),
         Output('prediction_m', 'children')],
        [Input('optimize_button', 'n_clicks')],
        [State('acousticness_slider', 'value'),
         State('danceability_slider', 'value'),
         State('energy_slider', 'value'),
         State('instrumentalness_slider', 'value'),
         State('speechiness_slider', 'value'),
         State('valence_slider', 'value'),
         State('key_drop', 'value'),
         State('mode_drop', 'value'),
         State('year_input', 'value')])
    def update_predictions(n_clicks, acoustic, dance, energy, instrumental, speech, valence, key, mode, year):
        # Keep collapse closed and update nothing if button is not clicked.
        if not n_clicks:
            return False, None, None, None, None, None, None, None, None, None, None

        # Ensure year has a value.
        year = 2020 if not year else round(year)

        # Create dictionary with all feature values. If left blank by user, None is passed as a value for the feature.
        known_features = dict(
            acousticness=[acoustic],
            danceability=[dance],
            energy=[energy],
            instrumentalness=[instrumental],
            speechiness=[speech],
            valence=[valence],
            key=[key],
            mode=[mode],
            year=[year],
        )

        # Get predicted feature values and predicted popularity using model. Method is explained in analysis.py.
        # Slice regression_cols[:-1] removes popularity column, as it is what is being predicted.
        features, popularity = predict_features(model=pop_rfr_model['model'], possible_values=known_features,
                                                reg_cols=regression_cols[:-1])

        # Add average year data if applicable.
        if 1928 <= year <= 2020:
            current_year_df = df[df['year'] == year]
            pop_mean = current_year_df['popularity'].mean()
            year_avg = "Average popularity for {} is {:.2f}%".format(year, pop_mean)
        else:
            year_avg = None

        # Format return values.
        popularity = "Predicted Popularity: {:.2f}%".format(popularity)
        acousticness = "Acousticness: {:.2f}".format(features['acousticness'])
        danceability = "Danceability: {:.2f}".format(features['danceability'])
        energy = "Energy: {:.2f}".format(features['energy'])
        instrumentalness = "Instrumentalness: {:.2f}".format(features['instrumentalness'])
        speechiness = "Speechiness: {:.2f}".format(features['speechiness'])
        valence = "Valence: {:.2f}".format(features['valence'])
        key = f"Key: {keys[round(features['key'])]}"
        mode = f"Mode: {mode_options[int(features['mode'])]['label']}"

        return True, popularity, year_avg, acousticness, danceability, energy, instrumentalness, \
            speechiness, valence, key, mode

    return app


if __name__ == "__main__":
    # Create DataFrame from preprocessed data.
    df_spotify = read_dataset(Path('..', 'data', 'data_clean.csv'))

    # Create and run app.
    app_t = spotify_db_app(df=df_spotify)
    app_t.run_server(debug=False)

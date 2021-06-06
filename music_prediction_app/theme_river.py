from plotly.graph_objs import Figure
import pandas as pd

from music_prediction_app.helper_lib import get_numeric_columns, standardize_column


def create_theme_river():
    """
    :return fig:
    This method is used to generate the Theme River plot from the spotify dataset based on various song features
    """
    df = pd.read_csv("../data/data_by_artist.csv")
    # Selecting top 20 popular artists
    df = df.sort_values(by='popularity', ascending=False)
    df_artist_names = df.iloc[0: 20]

    def get_agg_song_props_by_artist(df: pd.DataFrame):
        # fetch all selected columns
        row_limit = 20
        numeric_columns = get_numeric_columns(df)  # to be modified
        # filter records
        df_artist = df.iloc[0: row_limit]
        numeric_columns.append('artists')
        df_artist = df_artist[numeric_columns].copy()
        # perform aggregation
        numeric_columns.remove('artists')
        df_artist_data_agg = df_artist.groupby('artists')[numeric_columns].mean()
        for nc in numeric_columns:
            df_artist_data_agg.loc[:, nc] = standardize_column(df_artist_data_agg.loc[:, nc])

        # print(df_artist_data_agg)
        return df_artist_data_agg

    artist_df = get_agg_song_props_by_artist(df)

    trace1 = {
        "line": {"width": 0},
        "mode": "lines",
        "name": "acousticness",
        "type": "scatter",
        "x": df_artist_names['artists'],
        "y": artist_df['acousticness']
    }
    trace2 = {
        "fill": "tonexty",
        "line": {
            "color": "rgb(204, 234, 156)",
            "shape": "spline",
            "width": 0
        },
        "name": "danceability",
        "type": "scatter",
        "x": df_artist_names['artists'],
        "y": artist_df['danceability'],
        "fillcolor": "rgb(204, 234, 156)"
    }
    trace3 = {
        "fill": "tonexty",
        "line": {
            "color": "rgb(243, 250, 182)",
            "shape": "spline",
            "width": 0
        },
        "name": "duration_ms",
        "type": "scatter",
        "x": df_artist_names['artists'],
        "y": artist_df['duration_ms'],
        "fillcolor": "rgb(243, 250, 182)"
    }
    trace4 = {
        "fill": "tonexty",
        "line": {
            "color": "rgb(43, 143, 74)",
            "shape": "spline",
            "width": 0
        },
        "name": "energy",
        "type": "scatter",
        "x": df_artist_names['artists'],
        "y": artist_df['energy'],
        "fillcolor": "rgb(43, 143, 74)"
    }
    trace5 = {
        "fill": "tonexty",
        "line": {
            "color": "rgb(150, 211, 133)",
            "shape": "spline",
            "width": 0
        },
        "name": "instrumentalness",
        "type": "scatter",
        "x": df_artist_names['artists'],
        "y": artist_df['instrumentalness'],
        "fillcolor": "rgb(150, 211, 133)"
    }
    trace6 = {
        "fill": "tonexty",
        "line": {
            "color": "rgb(4, 107, 56)",
            "shape": "spline",
            "width": 0
        },
        "name": "liveness",
        "type": "scatter",
        "x": df_artist_names['artists'],
        "y": artist_df['liveness'],
        "fillcolor": "rgb(4, 107, 56)"
    }
    trace7 = {
        "fill": "tonexty",
        "line": {
            "color": "rgb(88, 182, 104)",
            "shape": "spline",
            "width": 0
        },
        "name": "loudness",
        "type": "scatter",
        "x": df_artist_names['artists'],
        "y": artist_df['loudness'],
        "fillcolor": "rgb(88, 182, 104)"
    }
    trace8 = {
        "fill": "tonexty",
        "line": {
            "color": "rgb(71, 209, 71)",
            "shape": "spline",
            "width": 0
        },
        "name": "speechiness",
        "type": "scatter",
        "x": df_artist_names['artists'],
        "y": artist_df['speechiness'],
        "fillcolor": "rgb(71, 209, 71)"
    }
    trace9 = {
        "fill": "tonexty",
        "line": {
            "color": "rgb(0, 77, 26)",
            "shape": "spline",
            "width": 0
        },
        "name": "tempo",
        "type": "scatter",
        "x": df_artist_names['artists'],
        "y": artist_df['tempo'],
        "fillcolor": "rgb(0, 77, 26)"
    }
    trace10 = {
        "fill": "tonexty",
        "line": {
            "color": "rgb(0, 179, 60)",
            "shape": "spline",
            "width": 0
        },
        "name": "valence",
        "type": "scatter",
        "x": df_artist_names['artists'],
        "y": artist_df['valence'],
        "fillcolor": "rgb(0, 179, 60)"
    }
    trace11 = {
        "fill": "tonexty",
        "line": {
            "color": "rgb(0, 153, 0)",
            "shape": "spline",
            "width": 0
        },
        "name": "popularity",
        "type": "scatter",
        "x": df_artist_names['artists'],
        "y": artist_df['popularity'],
        "fillcolor": "rgb(0, 153, 0)"
    }
    trace12 = {
        "fill": "tonexty",
        "line": {
            "color": "rgb(91, 215, 91)",
            "shape": "spline",
            "width": 0
        },
        "name": "key",
        "type": "scatter",
        "x": df_artist_names['artists'],
        "y": artist_df['key'],
        "fillcolor": "rgb(91, 215, 91)"
    }
    trace13 = {
        "fill": "tonexty",
        "line": {
            "color": "rgb(0, 230, 115)",
            "shape": "spline",
            "width": 0
        },
        "name": "mode",
        "type": "scatter",
        "x": df_artist_names['artists'],
        "y": artist_df['mode'],
        "fillcolor": "rgb(0, 230, 115)"
    }
    trace14 = {
        "fill": "tonexty",
        "line": {
            "color": "rgb(179, 255, 102)",
            "shape": "spline",
            "width": 0
        },
        "name": "number of songs",
        "type": "scatter",
        "x": df_artist_names['artists'],
        "y": artist_df['count'],
        "fillcolor": "rgb(179, 255, 102)"
    }

    data = list(
        [trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10, trace11, trace12, trace13,
         trace14])
    layout = {
        "font": {"family": "Balto"},
        "width": 1100,
        "xaxis": {
            "mirror": True,
            "ticklen": 30,
            "showgrid": False,
            "showline": True,
            "tickfont": {"size": 11},
            "zeroline": False,
            "showticklabels": True
        },
        "yaxis": {
            "mirror": True,
            "ticklen": 4,
            "showgrid": False,
            "showline": True,
            "tickfont": {"size": 11},
            "zeroline": False,
            "showticklabels": True
        },
        "height": 500,
        "margin": {
            "b": 60,
            "l": 60,
            "r": 60,
            "t": 80
        },
        "autosize": False,
        "hovermode": "x",
        "showlegend": False
    }
    fig = Figure(data=data, layout=layout)
    return fig


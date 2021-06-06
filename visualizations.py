from typing import Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from spotify_analysis import load_clean_spotify, rename_keys_modes


def year_slider(df: pd.DataFrame):
    # Values must be sort in terms of the animation column.
    df = df.sort_values('year')

    fig = px.scatter(df, x="energy", y="popularity",
                     animation_frame="year",  # set animation column
                     size="duration_ms",
                     color="tempo",
                     hover_name="name",
                     size_max=55,
                     range_x=[0, 1],
                     range_y=[0, 100])

    # Uncomment to remove auto-animate.
    # fig["layout"].pop("updatemenus")

    return fig


def corr_heatmap(df: pd.DataFrame):

    # Obtain 2D matrix of pearson correlations.
    cors = df.corr(method='pearson')

    # Generate heatmap.
    fig = px.imshow(cors, color_continuous_scale='picnic')
    return fig


def genre_tree_map(df: pd.DataFrame):
    # Add root container.
    df["root"] = "Spotify Music"

    # Generate treemap using values as colors.
    fig = px.treemap(df, path=['root', 'modality', 'key_letter', 'genres'], hover_name='genres', color='popularity')
    return fig


if __name__ == "__main__":
    spotify_df = load_clean_spotify()
    genre_df = pd.read_csv('archive/data_by_genres.csv')

    # Slider
    # year_slider = year_slider(spotify_df)
    # year_slider.show()

    # Heatmap
    # heatmap = corr_heatmap(spotify_df)
    # heatmap.show()

    # Tree map
    # tree_map = genre_tree_map(rename_keys_modes(genre_df))
    # tree_map.show()

from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
import plotly.express as px


def read_dataset(path: Path) -> pd.DataFrame:
    """
    :param path:
    :return df:
    This method will be responsible to read the dataset.
    """
    return pd.read_csv(path)


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    :param df:
    :return list:
    This methods gets all numeric columns from the pandas dataframe and returns a list of numeric column names
    """
    return list(df.select_dtypes(include=[np.number]).columns.values)


def get_agg_song_props_by_artist(df: pd.DataFrame, columns: list, row_limit: list):
    """
    :param df:
    :param columns:
    :param row_limit:
    :return df:
    This method aggregates song features grouped by the respective artists
    """
    # filter records
    df_artist = df.iloc[row_limit[0]: row_limit[1], :]
    df_artist_data_agg = df_artist.groupby('artist')[columns].mean()
    return df_artist_data_agg


def prepare_table_from_dataframe(df: pd.DataFrame):
    """
    :param df:
    :return dbc.Table:
    This method converts a dataframe into dbc table
    """
    return dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True)


def plotly_histogram_plot(df: pd.DataFrame, x: str, y: list):
    """
    :param df:
    :param x:
    :param y:
    :return fig:
    This method returns a histogram figure object
    """
    fig = px.histogram(df, x=x, y=y, histfunc="sum")
    return fig


def plotly_bar_chart(df: pd.DataFrame, x: str, y: list):
    """
    :param df:
    :param x:
    :param y:
    :return fig:
    This method returns a bar chart figure object
    """
    fig = px.bar(df, x=x, y=y)
    return fig


def plotly_line_chart(df: pd.DataFrame, x: str, y: list):
    """
    :param df:
    :param x:
    :param y:
    :return fig:
    This method returns a line chart figure object
    """
    df.sort_values(by=y)
    fig = px.line(df, x=x, y=y)
    return fig


def plotly_scatter_plot(df: pd.DataFrame, x: str, y: list):
    """
    :param df:
    :param x:
    :param y:
    :return fig:
    This method returns a scatter-plot figure object
    """
    df.sort_values(by=y)
    fig = px.scatter(df, x=x, y=y)
    return fig


def plotly_area_plot(df: pd.DataFrame, x: str, y: list):
    """
    :param df:
    :param x:
    :param y:
    :return fig:
    This method returns a area-plot figure object
    """
    df.sort_values(by=y)
    fig = px.area(df, x=x, y=y)
    return fig


def fix_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    :param df:
    :param column:
    :return df:
    This method is used to fix outliers in the dataset column provided as argument
    """
    # Function ensures that the column data type is numeric for outlier detection.
    # If not numeric, df is returned unchanged.
    if df[column].dtypes == np.number:

        # The z-score technique is applied to detect outliers.
        # Threshold of 3 is selected as it should include over 99% of population.
        threshold = 3
        mean = df[column].mean()
        std = df[column].std(skipna=True)
        z_scores = [(data_point - mean) / std for data_point in df[column]]

        # List of bools to track non-outliers is created.
        not_outliers = []
        for val in z_scores:
            # Ensures that NANs are not removed, as this function is only concerned with outliers.
            if np.isnan(val):
                not_outliers.append(True)
            else:
                not_outliers.append(np.abs(val) < threshold)

        # Return only points which are not outliers.
        return df[not_outliers]
    return df


def fix_nans(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    :param df:
    :param column:
    :return df:
    This method is used to fix Nans in the dataset column provided as argument
    """
    df_cpy = df.copy()

    # If data is numeric, the simple data imputation of assigning the mean is implemented
    if df_cpy[column].dtypes == np.number:
        mean = df_cpy[column].mean()
        df_cpy[column] = df_cpy[column].fillna(value=mean)

    # If data is not numeric, and sample size if large enough, then missing data is dropped.
    elif df_cpy.size > 30:
        df_cpy.dropna(subset=[column], inplace=True)

    # If column is not numeric and sample size is small, DataFrame is returned unchanged.
    return df_cpy


def normalize_column(df_column: pd.Series) -> pd.Series:
    """
    :param df_column:
    :return Series:
    This method is used to normalize dataset
    """
    # Check if column is categorical.
    if df_column.dtypes == pd.Categorical:
        return df_column

    # Returns series with all values normalized between 0 and 1.
    max_val = df_column.max()
    min_val = df_column.min()

    # If the max and min are the same, all data points are the same, therefore arbitrary value of 0 can be assigned.
    if min_val == max_val:
        norm_col = [0 for _ in df_column]

    # Otherwise the normalized value is calculated.
    else:
        norm_col = (df_column - min_val) / (max_val - min_val)
    return norm_col


def standardize_column(df_column: pd.Series) -> pd.Series:
    """
    :param df_column:
    :return Series:
    This method is used to standardize the dataset
    """
    # Z-scores are calculated to standardize.
    mean = df_column.mean()
    std = df_column.std()
    std_col = (df_column - mean) / std
    return std_col

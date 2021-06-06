import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from enum import Enum
from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype, is_bool_dtype
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


##################################################################################
# Spotify Specific.
##################################################################################
def rename_keys_modes(df: pd.DataFrame):
    df_cpy = df.copy()
    keys = {0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F', 6: 'F#', 7: 'G',
            8: 'G#', 9: 'A', 10: 'A#', 11: 'B', -1: 'None'}

    # Add mapped keys.
    df_cpy['key_decode'] = df_cpy['key'].map(keys)

    # Add modes.
    df_cpy['mode_decode'] = np.where(df_cpy['mode'] == 1, 'Major', 'Minor')
    return df_cpy


def encode_keys_modes(df: pd.DataFrame):
    df_cpy = df.copy()
    keys = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7,
            'G#': 8, 'A': 9, 'A#': 10, 'B': 11, 'None': -1}

    # Add mapped keys.
    df_cpy['key_encoded'] = df_cpy['key'].map(keys)

    # Add modes.
    df_cpy['mode_encoded'] = np.where(df_cpy['mode'] == 'Major', 1, 0)
    return df_cpy


##################################################################################
# Basic retrieving.
##################################################################################
def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = []
    for col in df:
        if is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    return numeric_cols


def get_binary_columns(df: pd.DataFrame) -> List[str]:
    binary_cols = []
    for col in df:
        if is_bool_dtype(df[col]):
            binary_cols.append(col)
    return binary_cols


def get_text_categorical_columns(df: pd.DataFrame) -> List[str]:
    categorical = []
    for col in df:
        if is_string_dtype(df[col]):
            categorical.append(col)
    return categorical


##################################################################################
# Basic cleaning.
##################################################################################
class WrongValueNumericRule(Enum):
    MUST_BE_POSITIVE = 0
    MUST_BE_NEGATIVE = 1
    MUST_BE_GREATER_THAN = 2
    MUST_BE_LESS_THAN = 3


def fix_numeric_wrong_values(df: pd.DataFrame,
                             column: str,
                             must_be_rule: WrongValueNumericRule,
                             must_be_rule_optional_parameter: Optional[float] = None) -> pd.DataFrame:
    df_cpy = df.copy()  # Copy is created to preserve original DataFrame.

    # An assumption is made that 0 may be included for both MUST_BE_POSITIVE and MUST_BE_NEGATIVE.
    if must_be_rule == WrongValueNumericRule.MUST_BE_POSITIVE:
        df_cpy[column].values[df_cpy[column] < 0] = np.nan
    elif must_be_rule == WrongValueNumericRule.MUST_BE_NEGATIVE:
        df_cpy[column].values[df_cpy[column] > 0] = np.nan
    elif must_be_rule == WrongValueNumericRule.MUST_BE_GREATER_THAN:
        df_cpy[column].values[df_cpy[column] < must_be_rule_optional_parameter] = np.nan
    elif must_be_rule == WrongValueNumericRule.MUST_BE_LESS_THAN:
        df_cpy[column].values[df_cpy[column] > must_be_rule_optional_parameter] = np.nan

    return df_cpy


def fix_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
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
    # Z-scores are calculated to standardize.
    mean = df_column.mean()
    std = df_column.std()
    std_col = (df_column - mean) / std
    return std_col


##################################################################################
# Encoding Methods.
##################################################################################
def generate_label_encoder(df_column: pd.Series) -> LabelEncoder:
    label_encoder = LabelEncoder()
    return label_encoder.fit(df_column)


def generate_one_hot_encoder(df_column: pd.Series) -> OneHotEncoder:
    hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    # A temp DataFrame is created to ensure proper dimensions.
    col_df = pd.DataFrame({df_column.name: df_column})
    return hot_encoder.fit(col_df)


def replace_with_label_encoder(df: pd.DataFrame, column: str, le: LabelEncoder) -> pd.DataFrame:
    # Function replaces column in original df with the array produced by the label encoder's transformation.
    df_cpy = df.copy()
    df_cpy[column] = le.transform(df_cpy[column])
    return df_cpy


def replace_with_one_hot_encoder(df: pd.DataFrame, column: str, ohe: OneHotEncoder, ohe_cols: List[str]) -> pd.DataFrame:
    # The array returned by the transformation is converted to DataFrame to add columns and enable concatenation.
    ct = ColumnTransformer([(column, ohe, [column])], remainder='drop')
    df2 = pd.DataFrame(ct.fit_transform(df), columns=ohe_cols)

    # New DataFrame is created with original column removed and encoded columns added.
    df.reset_index(inplace=True, drop=True)
    new_df = pd.concat([df.drop(columns=[column]), df2], axis=1)

    return new_df


def replace_label_encoder_with_original_column(df: pd.DataFrame, column: str, le: LabelEncoder) -> pd.DataFrame:
    df_cpy = df.copy()

    # Inverse transformation array is assigned to the encoded column.
    df_cpy[column] = le.inverse_transform(df_cpy[column])
    return df_cpy


def replace_one_hot_encoder_with_original_column(df: pd.DataFrame,
                                                 columns: List[str],
                                                 ohe: OneHotEncoder,
                                                 original_column_name: str) -> pd.DataFrame:
    # The array returned by the inverse transformation is converted to a df to add columns and enable concatenation.
    df2 = pd.DataFrame(ohe.inverse_transform(df.filter(columns)), columns=[original_column_name])

    # New DataFrame is created with encoded columns removed and original columns added.
    new_df = pd.concat([df.drop(columns=columns), df2], axis=1)
    return new_df


##################################################################################
# Classification.
##################################################################################
def simple_random_forest_classifier(X: pd.DataFrame, y: pd.Series) -> Dict:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    ###########
    # Remember to mess with n_estimators, max_depth, and max_leaf_nodes
    ###########
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)

    accuracy = model.score(X_test, y_test)
    # accuracy_score(y_test, y_predict)  # This is equivalent to above calculation.
    return dict(model=model, accuracy=accuracy, test_prediction=y_predict)


def decision_tree_classifier(X: pd.DataFrame, y: pd.Series) -> Dict:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    ###########
    # Remember to mess with max_depth, and max_leaf_nodes
    ###########
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    return dict(model=model, accuracy=accuracy, test_prediction=y_predict)


##################################################################################
# Regression.
##################################################################################
def simple_random_forest_regressor(X: pd.DataFrame, y: pd.Series) -> Dict:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    # Change n_estimators, max_depth and max_leaf_nodes
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    score = model.score(X_test, y_test)

    return dict(model=model, score=score, test_prediction=y_predict)


def decision_tree_regressor(X: pd.DataFrame, y: pd.Series) -> Dict:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)

    score = model.score(X_test, y_test)
    return dict(model=model, score=score, test_prediction=y_predict)


##################################################################################
# Clustering.
##################################################################################
def simple_k_means(X: pd.DataFrame, n_clusters=3, score_metric='euclidean') -> Dict:
    model = KMeans(n_clusters=n_clusters)
    clusters = model.fit_transform(X)

    # Using silhouette score.
    score = silhouette_score(X, model.labels_, metric=score_metric)
    return dict(model=model, score=score, clusters=clusters)


def mini_k_means(X: pd.DataFrame, n_clusters=3, score_metric='euclidean') -> Dict:
    model = MiniBatchKMeans(n_clusters=n_clusters)
    clusters = model.fit_predict(X)

    # There are many methods of deciding a score of a cluster model. Here is one example:
    score = silhouette_score(X, model.labels_, metric=score_metric)
    return dict(model=model, score=score, clusters=clusters)


def custom_clustering(X: pd.DataFrame) -> Dict:
    ######
    # Mess with EPS values.
    ######
    model = DBSCAN(eps=0.26)
    clusters = model.fit_predict(X)

    # Using silhouette score.
    score = silhouette_score(X, model.labels_, metric='euclidean')
    return dict(model=model, score=score, clusters=clusters)

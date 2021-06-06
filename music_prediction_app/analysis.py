from itertools import product
from typing import Optional, Dict, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def pop_rf_regressor(df: pd.DataFrame, features: Optional[list] = None) -> Dict:
    """
    :param df:
    :param features:
    :return Dict:
    This method is used to predict song popularity based on the list of features supplied as argument
    """
    # Get columns
    df2 = df.filter(items=features) if features is not None else df.copy()

    X, y = df2.drop(columns=['popularity']), df2['popularity']

    # Generate random forest regressor.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    # Create Model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Get score
    score = model.score(X_test, y_test)

    return dict(model=model, score=score)


def predict_features(model: RandomForestRegressor, possible_values: dict, reg_cols: list) -> Tuple:
    """
    :param model:
    :param possible_values:
    :param reg_cols:
    :return Tuple:
    This method is used to predict song features based on the possible values provided as argument
    """
    # Define special cols.
    binary_features = ['mode', 'explicit']
    discrete_features = ['key']

    # Get unknown features.
    unknown_features = []
    for key, val in possible_values.items():
        if val[0] is None:
            unknown_features.append(key)

    # Set precision based on number of unknowns.
    num_unknown = len(unknown_features)
    if num_unknown < 3:
        precision = 101
    elif num_unknown < 4:
        precision = 21
    elif num_unknown < 6:
        precision = 11
    else:
        precision = 6

    # Make dictionary with all possible values for each feature.
    for feature in unknown_features:
        if feature in binary_features:
            possible_values[feature] = [0, 1]
        elif feature in discrete_features:  # For 'key' column, values range 0 to 11.
            possible_values[feature] = list(range(0, 12))
        else:  # For all normalized values, the possible values range between 0 and 1, incrementing by 0.2 or 0.1.
            possible_values[feature] = list(np.linspace(0, 1, precision))

    # Cross multiply possible values and create list of dictionaries with all possible combinations.
    # Method written by Seth Johnson was adapted to create a cartesian product of a dictionary of lists.
    # Source: https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    prediction_data = list(dict(zip(possible_values.keys(), values)) for values in product(*possible_values.values()))

    # Load possible values into df to make predictions from.
    df = pd.DataFrame(prediction_data)

    # Ensure columns are in same order training data.
    df = df.filter(items=reg_cols)

    # Get all predicted popularity values for all combinations.
    predictions = model.predict(df)

    # Get index value of highest predicted popularity value.
    max_idx = np.argmax(predictions)

    # Return row corresponding with max predicted value, and the predicted value itself.
    return df.iloc[max_idx].to_dict(), predictions[max_idx]


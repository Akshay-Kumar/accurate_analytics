from helpers import *


# Used to load same version of data set in below functions.
def load_clean_spotify() -> pd.DataFrame:
    df = pd.read_csv('archive/data.csv')

    # Create artist column from first artist mentioned.
    df['artist'] = df['artists'].str.split('\'').str[1]

    # Add column for non label encoded columns for key and mode (for visualizations).
    df = rename_keys_modes(df)

    df['artist_count'] = df.groupby('artist')['artist'].transform('count')

    # Prep for visualizations. Not needed for analysis.
    # df = df.filter(items=['tempo', 'duration_ms', 'energy', 'popularity', 'artist', 'key_decode', 'mode_decode', 'name',
    #                       'acousticness', 'danceability', 'explicit', 'instrumentalness', 'liveness', 'loudness',
    #                       'speechiness', 'valence', 'year', 'artist_count', 'id'])
    # df.reset_index(inplace=True, drop=True)
    # df.to_csv('./data/data.csv')
    return df


def classify_key(df: pd.DataFrame, col: Optional[list] = None) -> Dict:
    # Get columns of interest chosen in main method.
    features = col if col is not None else df.columns

    df2 = df.filter(items=features)

    X, y = df2.drop(columns=['key']), df2['key']

    dt_classifier = decision_tree_classifier(X, y)
    rf_classifier = simple_random_forest_classifier(X, y)

    # Accuracy comparison.
    print("decision tree accuracy: " + str(dt_classifier['accuracy']))
    print("random forest accuracy: " + str(rf_classifier['accuracy']))

    if dt_classifier['accuracy'] > rf_classifier['accuracy']:
        return dt_classifier
    else:
        return rf_classifier


def classify_popularity(df: pd.DataFrame, col: Optional[list] = None) -> Dict:
    # Get columns
    features = col if col is not None else df.columns

    # Classify popularity.
    pop_mean = df['popularity'].mean()
    pop_std = df['popularity'].std()
    df2 = df.filter(items=features)

    # Classify popular music.
    df2['popularity'].values[df['popularity'] < pop_mean - 2 * pop_std] = 0
    df2['popularity'].values[(pop_mean - 2 * pop_std < df['popularity']) & (df['popularity'] < pop_mean)] = 1
    df2['popularity'].values[(pop_mean < df['popularity']) & (df['popularity'] < pop_mean + 2 * pop_std)] = 2
    df2['popularity'].values[df['popularity'] > pop_mean + 2 * pop_std] = 3

    print(pop_mean, pop_std, '\n', df2['popularity'].value_counts(sort=True))

    # Specify years.
    # df2 = df2[df2['year'] > 2000]

    # Do classification.
    X, y = df2.drop(columns=['popularity']), df2['popularity']
    dt_classifier = decision_tree_classifier(X, y)
    rf_classifier = simple_random_forest_classifier(X, y)

    # Accuracy comparison.
    print("decision tree accuracy: " + str(dt_classifier['accuracy']))
    print("random forest accuracy: " + str(rf_classifier['accuracy']))

    if dt_classifier['accuracy'] > rf_classifier['accuracy']:
        return dt_classifier
    else:
        return rf_classifier


def regress_popularity(df: pd.DataFrame, col: Optional[list] = None) -> Dict:
    # Get columns
    features = col if col is not None else df.columns
    df2 = df.filter(items=features)

    X, y = df2.drop(columns=['popularity']), df2['popularity']

    # Generate decision tree regressor.
    dt_regressor = decision_tree_regressor(X, y)

    # Generate random forest regressor.
    rf_regressor = simple_random_forest_regressor(X, y)

    # Compare results.
    print("decision tree score: " + str(dt_regressor['score']))
    print("random forest score: " + str(rf_regressor['score']))
    if dt_regressor['score'] > rf_regressor['score']:
        return dt_regressor
    else:
        return rf_regressor


def db_scan_popularity(df: pd.DataFrame, col: Optional[list] = None) -> Dict:
    features = col if col is not None else df.columns
    df2 = df.filter(items=features)

    for c in list(df2.columns):
        df2[c] = normalize_column(df2[c])

    # Tooo slow, clustering algorithm needs tweaking.
    return custom_clustering(df2)


if __name__ == "__main__":
    # Load dataset.
    spotify_df = load_clean_spotify()

    # Train for key.
    key_columns = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "key", "liveness",
                   "mode", "speechiness", "tempo", "year", "valence", "popularity"]
    # key_classified = classify_key(spotify_df, key_columns)
    # print(key_classified)

    # Classify popularity.
    # pop_classified = classify_popularity(spotify_df, key_columns)
    # print(pop_classified)

    # Popularity regression.
    # pop_regress = regress_popularity(spotify_df, key_columns)
    # print(pop_regress)

    # Popularity clustering
    cluster_columns = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "key", "liveness",
                       "speechiness", "tempo", "year", "valence", "popularity"]
    pop_cluster = db_scan_popularity(spotify_df, key_columns)
    print(pop_cluster)

import kagglehub
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def pre_process(df):
    boolean_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]

    for col in boolean_cols:
        df[col] = df[col].replace({"yes": 1, "no": 0})

    df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True, dtype=int)

    df["total_rooms"] = df["bedrooms"] + df["bathrooms"] + df["guestroom"]
    df["prime_location"] = np.bitwise_and(df["prefarea"], df["mainroad"])

    df["furnishing"] = np.bitwise_not(np.bitwise_or(df["furnishingstatus_unfurnished"], df["furnishingstatus_semi-furnished"]))
    df["total_furnishing"] = df["airconditioning"] + df["furnishing"] + df["hotwaterheating"]

    '''
    1 _ 0 = 0
    0 _ 1 = 0
    1 _ 1 = 0
    0 _ 0 = 1
    '''

    df = df.astype(float)
    return df

def get_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)

    return mi_scores

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    plt.show()

def main():
    path = kagglehub.dataset_download("yasserh/housing-prices-dataset")

    df = pd.read_csv(f"/{path}/housing.csv")
    df = pre_process(df)

    X = df.drop(columns=['price'])
    y = df['price']
    # All discrete features should now have integer dtypes (double-check this before using MI!)
    discrete_features = X.dtypes == int

    mi_scores = get_mi_scores(X, y, discrete_features)
    print(mi_scores)

    plt.figure(dpi=100, figsize=(8, 5))
    plot_mi_scores(mi_scores)

    # sns.relplot(x="area", y="price", hue="area", data=df)
    # sns.relplot(x="area_squared", y="price", hue="area", data=df)
    # sns.relplot(x="area_log", y="price", hue="area", data=df)
    # sns.relplot(x="prime_location", y="price", hue="price", data=df)
    plt.show()

if __name__ == "__main__":
    main()
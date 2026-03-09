import kagglehub
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

def pre_process(df):
    boolean_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]

    for col in boolean_cols:
        df[col] = df[col].replace({"yes": 1, "no": 0})

    df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True, dtype=int)
    df = df.astype(int)

    return df

def get_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)

    return mi_scores

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

if __name__ == "__main__":
    main()
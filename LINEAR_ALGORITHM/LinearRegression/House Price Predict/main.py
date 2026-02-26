import kagglehub
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def data_process(df):
    boolean_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]

    for col in boolean_cols:
        df[col] = df[col].replace({"yes": 1, "no": 0})

    df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True, dtype=int)
    # recommend using this method to pre-process the ranking data
    # mapping method sometimes make the model less accurate
    # drop_first = True -> avoid multicollinearity trap

    df = df.astype(float)

    return df

def main():
    path = kagglehub.dataset_download("yasserh/housing-prices-dataset")

    df = pd.read_csv(f"/{path}/housing.csv")

    df = data_process(df)

    y = df["price"]
    X = df.drop("price", axis=1)

    create_graph(X, y, "House Price Prediction")

    print(df.head())

if __name__ == "__main__":
    main()


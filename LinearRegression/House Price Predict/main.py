import kagglehub
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def data_process(df):
    boolean_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]

    for col in boolean_cols:
        df[col] = df[col].replace({"yes": 1, "no": 0})

    df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True, dtype=int)
    # recommend using this method to pre-process the ranking data
    # mapping method sometimes make the model less accurate
    # drop_first = True -> avoid multicollinearity trap

    df["total_rooms"] = df["bedrooms"] + df["bathrooms"] + df["guestroom"]
    df["prime_location"] = np.bitwise_or(df["prefarea"], df["mainroad"])

    df["furnishing"] = 1 - np.bitwise_or(df["furnishingstatus_unfurnished"], df["furnishingstatus_semi-furnished"])
    df["total_furnishing"] = df["airconditioning"] + df["furnishing"] + df["hotwaterheating"]

    df = df.astype(float)

    y = np.log1p(df["price"]) # log(1+df) avoid log(0)
    X = df[["total_rooms", "total_furnishing", "mainroad", "parking", "area", "stories", "basement", "stories", "prefarea"]]
    #process_linearity(X, y, "Price House")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # user scaler in case of some columns is much greater than the others
    # this dataset we can see column "area" and the others

    return X_train, X_test, y_train, y_test

def main():
    path = kagglehub.dataset_download("yasserh/housing-prices-dataset")

    df = pd.read_csv(f"{path}/housing.csv")

    X_train, X_test, y_train, y_test = data_process(df)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    MSE = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MSE: ", MSE)
    print("R2: ", r2)


if __name__ == "__main__":
    main()


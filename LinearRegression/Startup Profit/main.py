import kagglehub
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy import stats

def pre_process(df):
    df = pd.get_dummies(df, columns=["State"], drop_first=True, dtype=float)
    df = df[(np.abs(stats.zscore(df.select_dtypes(include=[np.number]))) < 3).all(axis=1)]

    print(df.keys())

    y = df["Profit"]
    X = df.drop("Profit", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

    # process_linearity(df, "Profit", "Startup Profit")

def main():
    path = kagglehub.dataset_download("karthickveerakumar/startup-logistic-regression")

    df = pd.read_csv(f"/{path}/50_Startups.csv")

    X_train, X_test, y_train, y_test = pre_process(df)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    MSE = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {MSE}")
    print(f"R2: {r2}")

if __name__ == "__main__":
    main()
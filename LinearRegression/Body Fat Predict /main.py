import kagglehub
import pandas as pd
from check_linear import process_linearity

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def preprocess(df):
    X = df.drop("BodyFat", axis=1)
    y = df["BodyFat"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def main():
    path = kagglehub.dataset_download("fedesoriano/body-fat-prediction-dataset")

    df = pd.read_csv(f'{path}/bodyfat.csv')

    print(df.head())

    # process_linearity(df, 'BodyFat', 'bodyfat dataset')

    X_train, X_test, y_train, y_test = preprocess(df)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    MSE = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MSE = ", MSE)
    print("R2 = ", r2)

if __name__ == '__main__':
    main()


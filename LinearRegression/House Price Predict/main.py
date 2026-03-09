import kagglehub
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def main():
    path = kagglehub.dataset_download("yasserh/housing-prices-dataset")

    df = pd.read_csv(f"/{path}/housing.csv")

    print(df.head())

    X_train, X_test, y_train, y_test = data_process(df)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    MSE = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MSE: ", MSE)
    print("R2: ", r2)

def data_process(df):
    boolean_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]

    for col in boolean_cols:
        df[col] = df[col].replace({"yes": 1, "no": 0})

    df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True, dtype=int)
    # recommend using this method to pre-process the ranking data
    # mapping method sometimes make the model less accurate
    # drop_first = True -> avoid multicollinearity trap

    df = df.astype(float)

    y = np.log1p(df["price"]) # log(1+df) avoid log(0)
    X = df.drop("price", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # user scaler in case of some columns is much greater than the others
    # this dataset we can see column "area" and the others

    print("X_train: ", X_train)
    print("y_train: ", y_train)
    print("X_test: ", X_test)
    print("y_test: ", y_test)

    return X_train, X_test, y_train, y_test

def gemini_main():
    path = kagglehub.dataset_download("yasserh/housing-prices-dataset")

    df = pd.read_csv(f"/{path}/housing.csv")

    print(df.head())

    boolean_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]

    for col in boolean_cols:
        df[col] = df[col].replace({"yes": 1, "no": 0})

    df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True, dtype=int)

    # Tính Z-score để loại bỏ các điểm nằm quá xa trung bình (thường là > 3 độ lệch chuẩn)
    from scipy import stats
    df = df[(np.abs(stats.zscore(df.select_dtypes(include=[np.number]))) < 3).all(axis=1)]

    df = df.astype(float)

    y = np.log1p(df["price"])
    X = df.drop("price", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Chuẩn hóa dữ liệu X
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred_log = model.predict(X_test_scaled)

    # 3. Đánh giá trên đơn vị gốc (nếu muốn xem MSE thật)
    y_test_original = np.expm1(y_test)
    y_pred_original = np.expm1(y_pred_log)

    print("R2 Score mới:", r2_score(y_test, y_pred_log))
    print("MSE thực tế (sau khi nghịch đảo log):", mean_squared_error(y_test_original, y_pred_original))

if __name__ == "__main__":
    gemini_main()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
y = np.array([[49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

# f(155) = 52, f(160) = 56

def no_library_prediction(height: int):
    one = np.ones((X.shape[0], 1))
    X_bar = np.concatenate((one, X), axis=1)

    A = X_bar.T @ X_bar
    B = X_bar.T @ y
    w = np.linalg.pinv(A) @ B

    w0 = w[0][0]
    w1 = w[1][0]

    return w1 * height + w0

def predict_with_library():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X, y)

    test = [[155], [160]]
    prediction = model.predict(X_test)

    mse = mean_squared_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)

    print(mse)
    print(r2)

    return prediction


def visualize_data():
    plt.plot(X.T, y.T, 'ro')
    plt.axis((140, 190, 45, 75))
    plt.xlabel('Height (cm)')
    plt.ylabel('Weight (kg)')
    plt.show()


if __name__ == '__main__':
    #visualize_data()
    print(no_library_prediction(160))
    print(predict_with_library())
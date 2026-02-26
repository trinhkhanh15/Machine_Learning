import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_harvey_collier

# 1. Generate Dummy Data
np.random.seed(42)
x = np.linspace(0, 10, 100)

# Linear relationship (Passes test)
y_linear = 2 * x + np.random.normal(0, 1, 100)

# Non-linear/Quadratic relationship (Fails test)
y_quad = x ** 2 + np.random.normal(0, 5, 100)


def check_linearity(X, y, name):
    # Statsmodels requires a constant (intercept)
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()

    # Perform Harvey-Collier Test
    # Note: The data must be sorted by the predictor for this test
    skip, p_value = linear_harvey_collier(model)

    print(f"--- Results for {name} ---")
    print(f"t-statistic: {skip:.4f}")
    print(f"p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("Result: Reject Null Hypothesis (Data is likely NON-LINEAR)")
    else:
        print("Result: Fail to Reject Null Hypothesis (Data appears LINEAR)")
    print("\n")


import matplotlib.pyplot as plt
import seaborn as sns

def create_graph(X, y, title):
    # Khởi tạo mô hình
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()

    # Lấy giá trị dự đoán và phần dư
    fitted_values = model.predict(X_with_const)
    residuals = model.resid

    # Vẽ biểu đồ
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=fitted_values, y=residuals)
    plt.axhline(y=0, color='red', linestyle='--')  # Đường chuẩn 0
    plt.title(f'Residual Plot: {title}')
    plt.xlabel('Giá trị dự đoán (Fitted Values)')
    plt.ylabel('Phần dư (Residuals)')
    plt.show()


if __name__ == "__main__":
    main()
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_harvey_collier

def process_linearity(X, y, title):
    check_linearity(X, y, title)
    create_graph(X, y, title)


def check_linearity(X, y, name):
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()

    skip, p_value = linear_harvey_collier(model)

    print(f"--- Results for {name} ---")
    print(f"t-statistic: {skip}")
    print(f"p-value: {p_value}")

    if p_value < 0.05:
        print("Result: Reject Null Hypothesis (Data is likely NON-LINEAR)")
    else:
        print("Result: Fail to Reject Null Hypothesis (Data appears LINEAR)")
    print("\n")

import matplotlib.pyplot as plt
import seaborn as sns

def create_graph(X, y, title):
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()

    fitted_values = model.predict(X_with_const)
    residuals = model.resid

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=fitted_values, y=residuals)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title(f'Residual Plot: {title}')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.show()

if __name__ == "__main__":
    main()
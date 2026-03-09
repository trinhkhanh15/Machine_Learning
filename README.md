
# Machine Learning?
- in every life situation, we see many function: 
  - hours studied → high exam score (sometimes it's not)
  - history of stock price → its next movement
- we want to predict these things with computer, but the computer doesn't understand these fucking things
- so we make a function 
- > $f(x) = ax + b$ and $y = f(x) + noise$ 
- > we need to find a and b such that $y - f(x)$ is as small as possible 
- this thing called Loss Function
- > $$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
- making ML model = choosing:
  - model form
  - loss function
  - optimize method 

## I. Linear Algorithm
### some important mistake:
1. **Non-linear data**
- In 2 dimension graph, linear data is a line. In 3 dimension, linear data is a plane. In 3+ dimension, concepts of line and plane are not suitable, some new concept appears is hyperplane. 
- Therefore, if we want to apply linear algorithm, we need to visualize the data.  
- using Harvey-Collier Linearity test is the real solution here: the main idea is recursive residual (phần dư tiến lũy):
  - sort the data list, then pick the first 5 point to create the temporary line
  - then pick the next 6th point, see how far is it to the line, the same process with 7th, 8th,... → here is recursive residual
  - if the data is linear: the average of the residual (phần dư) has a value approximately equal to 0, because the next point sometimes tilted upwards, sometime downwards
  - if it's non-linear: the value here is far away from 0, the reason is these points are skewed to one side of the line
- using "statsmodels" library to use this test: just focus on p_value, if it > 0.05 → it's linear, non-linear if it < 0.05
- don't just stop at comparing this value with 0.05, see the graph figure make from residuals, it's non-linear if it looked like the letter U
2. **Update soon**
### something make model less accurate
1. data in some columns are much greater than the others (using scaler)
2. the different between maximum and minimum is too large, residuals are too big (using log transform for result column)
3. the outliers: in result column, some values are much greater or much lower than the others (check and remove the rows which are rows with values exceeding 3 times the standard deviation)
4. missing feature engineering (making some feature from the raw data)
### 1. Linear Regression
- https://machinelearningcoban.com/2016/12/28/linearregression/ 
### 2. K-means Clustering
- Concept: https://machinelearningcoban.com/2017/01/01/kmeans/
- Application:
## II. Feature Engineering
### 1. Mutual Information (ML)
- the dataset always has a lot of feature, we don't know which feature actually have influence to the target.
- therefore, we need to construct the ranking of feature utility metric, that meansure the relationship between the feature and target.
- then pick the most useful feature, → this process called Mutual Information (MI)
- 3 main cons: 
  - High cardinality: the feature which has a lot of unique values like ID, phone number,... MI consider that feature having too much score 
  - The interaction of several features: MI just rate the score between the single feature and the target:
    - if u add the acreage feature A (m^2) and B (feet^2) to the model, it will be redundancy
    - as I said, MI just rate between the single feature and target, so we have a problem that A_score = 0, B_score = 0, but the combination of A and B is too important → we miss potential data
  - Sensitivity of noise: with the small dataset, MI is easily fooled by random coincidence 


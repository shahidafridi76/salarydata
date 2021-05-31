import pandas
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
ds = pandas.read_csv('SalaryData.csv')
ds
ds.columns
ds.info()
x = ds['YearsExperience'].values.reshape(30,1)
x.shape
y = ds['Salary']
model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
X_train.shape
X_test.shape
# algo 
model.fit(X_train , y_train)
model.coef_
X_test
y_test
y_pred = model.predict(X_test)
y_pred
y_test
112635/115790 * 100
# y = c +  wx
# 9449 * 1.1
model.predict([[ 1.1 ]] )
36187/39343 * 100
9449 * 1.1
c = model.intercept_
c
25792 + 9449 * 1.1
sns.set()
plt.scatter(X_train, y_train)
plt.scatter(X_test , y_test, color='red')
plt.plot(X_test , y_pred)
plt.xlabel("exp")
plt.ylabel("salary")
plt.title("exp vs salary pred")
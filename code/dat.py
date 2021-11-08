import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ridge_regression
from sklearn.metrics import mean_squared_error, r2_score
import os

os.chdir("E:\Hackathon\Machine_Hack\Shiv_Nadar_DAC")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
dat = pd.read_csv('data/test.csv')
dat.drop(['Item_ID'], axis=1, inplace=True)
print(dat.head())

fd = pd.get_dummies(dat, drop_first=True)
print(fd.head())
fd.to_csv("testdf2.csv", index=False)
#
# # Splitting date for Training and Testing
# x = fd.drop(["Sales"], axis=1)
# y = pd.DataFrame(fd["Sales"])
# y[['Sales']] = StandardScaler().fit_transform(y[['Sales']])
# print(x.shape, y.shape)
# print(y.head())
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=33)
#
# # Creating Simple Regression model
# reg = LinearRegression()
# reg.fit(x_train, y_train)
#
# y_pred = reg.predict(x_test)
# print(pd.DataFrame(y_pred))
# print(y_test)
#
# # The coefficients
# print("Coefficients: \n", reg.coef_)
# # The mean squared error
# print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# # The coefficient of determination: 1 is perfect prediction
# print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ridge_regression
from sklearn.metrics import mean_squared_error, r2_score
import os
os.chdir("E:\Hackathon\Machine_Hack\Shiv_Nadar_DAC")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

dtf = pd.read_csv("data/df.csv")
print(dtf.head())

print(dtf.info())

df = dtf.drop(['Item_ID'], axis=1)
# df['age'] = 2021 - df['Outlet_Year']
# df.drop(['Outlet_Year'], axis=1, inplace=True)

catcols = [col for col in df.columns if df[col].dtype == "O"]
numcol = [col for col in df.columns if df[col].dtype != "O"]


# fig, ax = plt.subplots(2, 3)
# axe = ax.ravel()
# for k, i in enumerate(catcols):
#     print(i)
#     print(dtf[i].value_counts())
#     dtf[i].value_counts().plot(kind="bar", ax=axe[k]).set_title(i)
# plt.show()



print(df.head())

fdf = pd.concat([df[numcol + ['Item_Type']], pd.get_dummies(df[[x for x in catcols[1:]]], drop_first=True)], axis=1)

print(pd.get_dummies(df[[x for x in catcols[1:]]], drop_first=True))
print(fdf.head())
fdf.to_csv("df2.csv", index=False)

# Label Encoder for multi class categorical predictor
ohe = OneHotEncoder()
ore = OrdinalEncoder()
le = LabelEncoder()
le.fit(fdf["Item_Type"])
fdf["Item_Type"] = le.transform(fdf["Item_Type"])
# fdf.drop(["Item_Type"], axis=1)
print(fdf.head())

# Splitting date for Training and Testing
x = fdf.drop(["Sales"], axis=1)
y = pd.DataFrame(fdf["Sales"])
print(x.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=33)

# Creating Simple Regression model
reg = LinearRegression()
reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)
print(pd.DataFrame(y_pred))
print(y_test)

# The coefficients
print("Coefficients: \n", reg.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

# Plot outputs
# plt.scatter(x_test, y_test, color="black")
plt.plot(x_test, y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

# plt.show()
# fdf.to_csv("cleaned1.csv", index=False)


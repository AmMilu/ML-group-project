# -*- coding = utf-8 -*-
# @Time : 2021/11/25 2:43 下午
# @Author : Kiser
# @File : LinearRegression.py
# @Software : PyCharm
import csv
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import pandas as pd
# import hvplot.pandas
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


def output_evaluate(target_value, predict_value):
    mean_absolute_error = metrics.mean_absolute_error(target_value, predict_value)
    mean_square_error = metrics.mean_squared_error(target_value, predict_value)
    root_mean_squared_error = np.sqrt(metrics.mean_squared_error(target_value, predict_value))
    coefficient_of_determination = metrics.r2_score(target_value, predict_value)
    print('Mean absolute error:', mean_absolute_error)
    print('Mean squared error:', mean_square_error)
    print('Root mean squared error:', root_mean_squared_error)
    print('Coefficient of determination', coefficient_of_determination)


# import the data
rental_house = pd.read_csv('updated_data.csv', header=0)
address = rental_house.iloc[:,2]
bed = rental_house.iloc[:,3]
bath = rental_house.iloc[:,4]
type = rental_house.iloc[:,5]
X = np.column_stack((address, bed, bath, type))
y = rental_house.iloc[:,1]/1000

# check out the data
# print(rental_house.head())
# print(rental_house.info())
# pd.set_option('display.max_columns', 20)  # 给最大列设置为10列
# print(rental_house.describe())
# sns.pairplot(rental_house)
# sns.heatmap(rental_house.corr(), annot=True)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(model.intercept_, model.coef_)

# draw the y_test-y_pred picture
plt.scatter(y_test, y_pred)
example = np.array([1, 2, 3, 4, 5, 6, 7])
plt.plot(example, example, color='red')
plt.xlim(1, 7)
plt.ylim(1, 7)
plt.xlabel('target value')
plt.ylabel('predict value')
plt.show()

# output the evaluation
scores = cross_val_score(model, X_test, y_test, cv=5)
print("Cross Validation Score:", scores.mean())
output_evaluate(y_test, y_pred)

# baseline
# from sklearn.dummy import DummyRegressor
# dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
# ydummy = dummy.predict(X_test)
# print(mean_squared_error(y_test, ydummy))
# # draw the y_test-y_pred picture
# plt.scatter(y_test, ydummy)
# example = np.array([1, 2, 3, 4, 5, 6, 7])
# plt.plot(example, example, color='red')
# plt.xlim(1, 7)
# plt.ylim(1, 7)
# plt.xlabel('target value')
# plt.ylabel('predict value')
# plt.show()
# output_evaluate(ydummy, y_pred)
# -*- coding = utf-8 -*-
# @Time : 2021/11/25 2:43 下午
# @Author : Kiser
# @File : LinearRegression.py
# @Software : PyCharm
from sklearn import metrics
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures


def output_evaluate(model, target_value, predict_value):
    mean_absolute_error = metrics.mean_absolute_error(target_value, predict_value)
    mean_square_error = metrics.mean_squared_error(target_value, predict_value)
    root_mean_squared_error = np.sqrt(metrics.mean_squared_error(target_value, predict_value))
    coefficient_of_determination = metrics.r2_score(target_value, predict_value)
    print('Mean absolute error:', mean_absolute_error)
    print('Mean squared error:', mean_square_error)
    print('Root mean squared error:', root_mean_squared_error)
    print('Coefficient of determination', coefficient_of_determination)
    scores = cross_val_score(model, X_test, y_test, cv=5)
    print("Cross Validation Score:", scores.mean())


def draw_picture(y_test, y_pred):
    plt.scatter(y_test, y_pred)
    example = np.array([1, 2, 3, 4, 5, 6, 7])
    plt.plot(example, example, color='red')
    plt.xlim(1, 7)
    plt.ylim(1, 7)
    plt.xlabel('target value')
    plt.ylabel('predict value')
    plt.show()


# import the data
rental_house = pd.read_csv('updated_data.csv', header=0)
address = rental_house.iloc[:,2]
bed = rental_house.iloc[:,3]
bath = rental_house.iloc[:,4]
type = rental_house.iloc[:,5]
X = np.column_stack((address, bed, bath, type))
y = rental_house.iloc[:,1]/1000

# check out the data
print(rental_house.head())
print(rental_house.info())
pd.set_option('display.max_columns', 20)  # 给最大列设置为10列
print(rental_house.describe())
col = ['price', 'address', 'bed', 'bath', 'type']
data = pd.DataFrame(rental_house, columns=col)
print(data)
sns.heatmap(data.corr(), annot=True)
sns.pairplot(data)
plt.show()

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

# train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(model.intercept_, model.coef_)

# draw the y_test-y_pred picture
draw_picture(y_test, y_pred)

# output the evaluation
output_evaluate(model, y_test, y_pred)

# use polynomial features to train the model
poly = PolynomialFeatures(2)
polyX = poly.fit_transform(X)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

# train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(model.intercept_, model.coef_)

# draw the y_test-y_pred picture
draw_picture(y_test, y_pred)

# output the evaluation
output_evaluate(model, y_test, y_pred)
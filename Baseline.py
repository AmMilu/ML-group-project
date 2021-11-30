# -*- coding = utf-8 -*-
# @Time : 2021/11/30 6:52 下午
# @Author : Kiser
# @File : Baseline.py
# @Software : PyCharm
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
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
    print('\n')


def draw_picture(y_test, ydummy):
    plt.scatter(y_test, ydummy)
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

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

# baseline mean
dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
ydummy = dummy.predict(X_test)
# draw the y_test-y_pred picture
draw_picture(y_test, ydummy)
# output the evaluation
print("Evaluation of DummyRegression:mean")
output_evaluate(dummy, y_test, ydummy)

# baseline median
dummy = DummyRegressor(strategy="median").fit(X_train, y_train)
ydummy = dummy.predict(X_test)
# draw the y_test-y_pred picture
draw_picture(y_test, ydummy)
# output the evaluation
print("Evaluation of DummyRegression:median")
output_evaluate(dummy, y_test, ydummy)

# baseline constant
dummy = DummyRegressor(strategy="constant", constant=2.841).fit(X_train, y_train)
ydummy = dummy.predict(X_test)
# draw the y_test-y_pred picture
draw_picture(y_test, ydummy)
# output the evaluation
print("Evaluation of DummyRegression:constant")
output_evaluate(dummy, y_test, ydummy)

print("Use polynomial features to train the model")
# use polynomial features to train the model
poly = PolynomialFeatures(2)
polyX = poly.fit_transform(X)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

# baseline mean
dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
ydummy = dummy.predict(X_test)
# draw the y_test-y_pred picture
draw_picture(y_test, ydummy)
# output the evaluation
print("Evaluation of DummyRegression:mean")
output_evaluate(dummy, y_test, ydummy)

# baseline median
dummy = DummyRegressor(strategy="median").fit(X_train, y_train)
ydummy = dummy.predict(X_test)
# draw the y_test-y_pred picture
draw_picture(y_test, ydummy)
# output the evaluation
print("Evaluation of DummyRegression:median")
output_evaluate(dummy, y_test, ydummy)

# baseline constant
dummy = DummyRegressor(strategy="constant", constant=2.841).fit(X_train, y_train)
ydummy = dummy.predict(X_test)
# draw the y_test-y_pred picture
draw_picture(y_test, ydummy)
# output the evaluation
print("Evaluation of DummyRegression:constant")
output_evaluate(dummy, y_test, ydummy)
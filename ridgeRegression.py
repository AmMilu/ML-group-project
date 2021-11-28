import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn import metrics

# start ---- read and split data
df = pandas.read_csv('updated_data.csv', header=0)
# print(df.head()) #return first 5 rows
X1_addressCode = df.iloc[:, 2]
X2_BedNum = df.iloc[:, 3]
X3_BathNum = df.iloc[:, 4]
X4_type = df.iloc[:, 5]
X = numpy.column_stack((X1_addressCode, X2_BedNum, X3_BathNum, X4_type))
y_price = numpy.array(df.iloc[:, 1] / 1000)  # use thousand as unit
# split out 10% use as final test and evaluate data
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y_price, test_size=0.1)
# end ---- read and split data

# start ---- select C using cross validation
mean_error = []
std_error = []
Ci_range = [0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1]
for Ci in Ci_range:
    squared_error = []
    kf = KFold(n_splits=5)
    for train, test in kf.split(Xtrain):
        RidgeModel = Ridge(alpha=1 / (2 * Ci)).fit(Xtrain[train], ytrain[train])
        ypred = RidgeModel.predict(Xtrain[test])
        squared_error.append(mean_squared_error(ytrain[test], ypred))
    squared_error = numpy.array(squared_error)
    mean_error.append(squared_error.mean())
    std_error.append(squared_error.std())
print("Ci_range:\n" + str(Ci_range))
print("mean error:\n" + str(mean_error))
# print(mean_error.index(numpy.array(mean_error).min()))
print("standard deviation:\n" + str(std_error))
print("------------------------")
plt.errorbar(Ci_range, mean_error, yerr=std_error)
plt.xlabel("Ci")
plt.ylabel("Mean square error")
plt.show()
# end ---- select C using cross validation

C = 0.05
RidgeModel = Ridge(alpha=1 / (2 * C)).fit(Xtrain, ytrain)
ypred = RidgeModel.predict(Xtest)
plt.scatter(ytest, ypred)
plt.xlim(1, 7)
plt.xlabel("ytest")
plt.ylim(1, 7)
plt.ylabel("ypred")
axis = numpy.array([1, 2, 3, 4, 5, 6, 7])
plt.plot(axis, axis, 'r')
plt.show()


def output_evaluate(target_value, predict_value):
    mean_absolute_error = metrics.mean_absolute_error(target_value, predict_value)
    mean_square_error = metrics.mean_squared_error(target_value, predict_value)
    root_mean_squared_error = numpy.sqrt(metrics.mean_squared_error(target_value, predict_value))
    coefficient_of_determination = metrics.r2_score(target_value, predict_value)
    print('Mean absolute error:', mean_absolute_error)
    print('Mean squared error:', mean_square_error)
    print('Root mean squared error:', root_mean_squared_error)
    print('Coefficient of determination', coefficient_of_determination)


output_evaluate(ytest, ypred)

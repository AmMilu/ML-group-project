from sklearn import metrics
def output_evaluate(target_value, predict_value):
    mean_absolute_error = metrics.mean_absolute_error(target_value, predict_value)
    mean_square_error = metrics.mean_squared_error(target_value, predict_value)
    root_mean_squared_error = np.sqrt(metrics.mean_squared_error(target_value, predict_value))
    coefficient_of_determination = metrics.r2_score(target_value, predict_value)
    print('Mean absolute error:', mean_absolute_error)
    print('Mean squared error:', mean_square_error)
    print('Root mean squared error:', root_mean_squared_error)
    print('Coefficient of determination', coefficient_of_determination)

#read data
import numpy as np
import pandas as pd
df = pd.read_csv('updated_data.csv', header=0)
#print(df.head()) #return first 5 rows
X1_addressCode = df.iloc[:,2]
X2_BedNum = df.iloc[:,3]
X3_BathNum = df.iloc[:,4]
X4_type = df.iloc[:,5]
X = np.column_stack((X1_addressCode,X2_BedNum,X3_BathNum,X4_type))
y = df.iloc[:,1]/1000 #rent is mostly four digits, to make value smaller and easier to read, divided price by 1000

#split data into 80% for training and 20% for testing
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.2,shuffle=True)

neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9,10, 11, 12, 13, 14, 15]
from sklearn.model_selection import KFold
kf = KFold(n_splits=5,shuffle=True)
mean_error = []
std_error = []
for ni in neighbors:
    from sklearn.neighbors import KNeighborsRegressor
    model = KNeighborsRegressor(n_neighbors=ni, weights="uniform")
    temp = []
    for train, test in kf.split(Xtrain,ytrain):
        model.fit(Xtrain[train], ytrain.to_numpy()[train])
        ypred = model.predict(Xtrain[test])
        from sklearn.metrics import mean_squared_error
        temp.append(mean_squared_error(ytrain.to_numpy()[test],ypred))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())
import matplotlib.pyplot as plt
plt.errorbar(neighbors, mean_error, yerr=std_error, linewidth=3)
plt.ylabel('Mean square error')
plt.xlabel('ni')
plt.show()

n = 4
model = KNeighborsRegressor(n_neighbors=n, weights = "uniform")
model.fit(Xtrain,ytrain)
ypred = model.predict(Xtest)

#on the 45 degree line, the predict value is same as the original value
#above the line, predicted value is larger and below the line, predicted value is smaller
plt.scatter(ytest, ypred)
plt.xlim(1,7)
plt.xlabel("ytest")
plt.ylim(1,7)
plt.ylabel("ypred")
a=np.array([1,2,3,4,5,6,7])
plt.plot(a,a,'r')
plt.show()

output_evaluate(ytest,ypred)
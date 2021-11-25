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
y = df.iloc[:,1]/1000

#split data into 80% for training and 20% for testing
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.2)

#train training data using F-fold cross validation and Lasso regression model
Ci_range = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10,20,30,40, 50, 100]
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
mean_error = []
std_error = []
for Ci in Ci_range:
    from sklearn import linear_model
    alpha = 1/(2*Ci)
    model = linear_model.Lasso(alpha=alpha)
    temp = []
    for train, test in kf.split(Xtrain,ytrain):
        model.fit(Xtrain[train], ytrain.to_numpy()[train])
        ypred = model.predict(Xtrain[test])
        from sklearn.metrics import mean_squared_error
        temp.append(mean_squared_error(ytrain.to_numpy()[test],ypred))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())
import matplotlib.pyplot as plt
plt.errorbar(Ci_range, mean_error, yerr=std_error, linewidth=3)
plt.ylabel('Mean square error')
plt.xlabel('Ci')
plt.show()

C = 30
alpha = 1/(2*C)
model = linear_model.Lasso(alpha=alpha)
model.fit(Xtrain,ytrain)
ypred = model.predict(Xtest)
print(model.coef_)
print(mean_squared_error(ytest,ypred))

plt.scatter(ytest, ypred)
plt.show()
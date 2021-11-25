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

#train training data using F-fold cross validation and Lasso regression model
Ci_range = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 30, 40, 50, 60 ,70]
from sklearn.model_selection import KFold
kf = KFold(n_splits=5,shuffle=True)
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
#[ 0.0701224   0.46132721  0.37570138 -0.46279684]

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

#Mean absolute error: 0.3396449413132823
#Mean squared error: 0.20640265523280518
#Root mean squared error: 0.4543155899072859
#Coefficient of determination 0.7083906018848933

from sklearn.dummy import DummyClassifier
#generates predictions by respecting the training set’s class distribution
dummy = DummyClassifier(strategy="stratified").fit(Xtrain, ytrain)  
ydummy = dummy.predict(Xtest)

plt.scatter(ytest, ydummy)
plt.xlim(1,7)
plt.xlabel("ydummy")
plt.ylim(1,7)
plt.ylabel("ypred")
a=np.array([1,2,3,4,5,6,7])
plt.plot(a,a,'r')
plt.show()

output_evaluate(ytest,ydummy)

#Mean absolute error: 0.8598621621621622
#Mean squared error: 1.377223564864865
#Root mean squared error: 1.1735516881948
#Coefficient of determination -0.9813899309591099
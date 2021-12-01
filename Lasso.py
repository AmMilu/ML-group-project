import numpy as np
import pandas as pd
from sklearn import metrics, linear_model
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

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
df = pd.read_csv('updated_data.csv', header=0)
#print(df.head()) #return first 5 rows
X1_addressCode = df.iloc[:,2]
X2_BedNum = df.iloc[:,3]
X3_BathNum = df.iloc[:,4]
X4_type = df.iloc[:,5]
X = np.column_stack((X1_addressCode,X2_BedNum,X3_BathNum,X4_type))
y = df.iloc[:,1]/1000 #rent is mostly four digits, to make value smaller and easier to read, divided price by 1000

#split data into 90% for training and 10% for testing
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.1,shuffle=True)

#use polynomial features to train the model
poly = PolynomialFeatures(2)
polyX = poly.fit_transform(X)

#split data into 80% for training and 20% for testing
Xtrainp, Xtestp, ytrainp, ytestp = train_test_split(polyX,y,test_size=0.1,shuffle=True)

plt.rc('font', size=18)

#train training data using F-fold cross validation and Lasso regression model
Ci_range = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 30, 40, 50]
poly = [1, 2, 3, 4, 5]
kf = KFold(n_splits=5,shuffle=True)
mean_error_mse = []
std_error_mse = []
mean_error_mse_poly = []
std_error_mse_poly = []
for Ci in Ci_range:
    alpha = 1/(2*Ci)
    model = linear_model.Lasso(alpha=alpha)
    temp = []
    temp_poly = []
    for train, test in kf.split(Xtrain,ytrain):
        model.fit(Xtrain[train], ytrain.to_numpy()[train])
        ypred = model.predict(Xtrain[test])
        temp.append(mean_squared_error(ytrain.to_numpy()[test],ypred))
    mean_error_mse.append(np.array(temp).mean())
    std_error_mse.append(np.array(temp).std())
    for train, test in kf.split(Xtrainp,ytrainp):
        model.fit(Xtrainp[train], ytrainp.to_numpy()[train])
        ypred = model.predict(Xtrainp[test])
        temp_poly.append(mean_squared_error(ytrainp.to_numpy()[test],ypred))
    mean_error_mse_poly.append(np.array(temp_poly).mean())
    std_error_mse_poly.append(np.array(temp_poly).std())
plt.errorbar(Ci_range, mean_error_mse, yerr=std_error_mse, linewidth=3, label="original features")
plt.errorbar(Ci_range, mean_error_mse_poly, yerr=std_error_mse_poly, linewidth=3, label="polynomial features")
plt.ylabel('Mean square error')
plt.xlabel('Ci')
plt.title('C vs Mean square error')
plt.legend()
plt.show()

C = 20
alpha = 1/(2*C)
model = linear_model.Lasso(alpha=alpha)
model.fit(Xtrain,ytrain)
ypred = model.predict(Xtest)
print(model.coef_)
#[ 0.07042885  0.44125777  0.40176972 -0.44564026]

#on the 45 degree line, the predict value is same as the original value
#above the line, predicted value is larger and below the line, predicted value is smaller
#plt.scatter(ytest, ypred)
#plt.xlim(1,7)
#plt.xlabel("ytest")
#plt.ylim(1,7)
#plt.ylabel("ypred")
#a=np.array([1,2,3,4,5,6,7])
#plt.plot(a,a,'r')
#plt.show()

output_evaluate(ytest,ypred)

#Mean absolute error: 0.3535018164052539
#Mean squared error: 0.23809428198655086
#Root mean squared error: 0.48794905675341854
#Coefficient of determination 0.7014576929886107

C = 5 #30
alpha = 1/(2*C)
model = linear_model.Lasso(alpha=alpha)
model.fit(Xtrainp,ytrainp)
ypred = model.predict(Xtestp)
print(model.coef_)
#[ 0.         -0.          0.          0.         -0.         -0.0015977
#  0.01867471  0.05731545 -0.03286495  0.04038638  0.          0.
#  0.         -0.         -0.        ]

#on the 45 degree line, the predict value is same as the original value
#above the line, predicted value is larger and below the line, predicted value is smaller
#plt.scatter(ytest, ypred)
#plt.xlim(1,7)
#plt.xlabel("ytest")
#plt.ylim(1,7)
#plt.ylabel("ypred")
#a=np.array([1,2,3,4,5,6,7])
#plt.plot(a,a,'r')
#plt.show()

output_evaluate(ytestp,ypred)

#Mean absolute error: 0.26931791240197156
#Mean squared error: 0.1206018654505888
#Root mean squared error: 0.3472777929130926
#Coefficient of determination 0.8155633000171529
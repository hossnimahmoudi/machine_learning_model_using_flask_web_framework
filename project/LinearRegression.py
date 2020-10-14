from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
import pickle

filename = 'files/BostonHousing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = read_csv(filename, names=names)
#print(dataframe)
array = dataframe.values

X = array[:, 0:13]
Y = array[:, 13]
num_folds = 10

kfold = KFold(n_splits=10, random_state=7)
Linear_Regression = LinearRegression()

scoring = 'neg_mean_squared_error'
results = cross_val_score(Linear_Regression, X, Y, cv=kfold, scoring=scoring)
print("Mean Estimated Linear_Regression", results.mean())

pickle.dump(Linear_Regression, open("models/modelLinear_Regression.pkl", "wb"))
#"crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","b","lstat","medv"
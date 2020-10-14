from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
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
Ridge_Regression = Ridge()

scoring = 'neg_mean_squared_error'
results = cross_val_score(Ridge_Regression, X, Y, cv=kfold, scoring=scoring)
print("Mean Estimated Ridge_Regression", results.mean())

pickle.dump(Ridge_Regression, open("models/modelRidge_Regression.pkl", "wb"))
#"crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","b","lstat","medv"
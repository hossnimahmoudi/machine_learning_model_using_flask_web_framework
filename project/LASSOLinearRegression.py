from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
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
LASSO_Regression = Lasso()

scoring = 'neg_mean_squared_error'
results = cross_val_score(LASSO_Regression, X, Y, cv=kfold, scoring=scoring)
print("Mean Estimated LASSO_Regression", results.mean())

pickle.dump(LASSO_Regression, open("models/modelLASSO_Regression.pkl", "wb"))
#"crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","b","lstat","medv"
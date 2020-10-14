from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
import pickle

filename = 'files/BostonHousing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = read_csv(filename, names=names)
print(dataframe)
array = dataframe.values

X = array[:, 0:13]
Y = array[:, 13]
num_folds = 10

kfold = KFold(n_splits=10, random_state=7)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
SVR = SVR(gamma='auto')


scoring = 'neg_mean_squared_error'

results = cross_val_score(SVR, X, Y, cv=kfold, scoring=scoring)
print("Mean Estimated SVR", results.mean())

pickle.dump(SVR, open("models/modelSVR.pkl", "wb"))

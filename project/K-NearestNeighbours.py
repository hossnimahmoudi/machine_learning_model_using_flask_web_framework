from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
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


kfold = KFold(n_splits=10, random_state=None)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
KNN = KNeighborsRegressor()
KNN.fit(X_train, y_train)
#y_pred = KNN.predict(X_test)

#print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
scoring = 'neg_mean_squared_error'

results = cross_val_score(KNN, X, Y, cv=kfold, scoring=scoring)
print("Mean Squared Error KNN", results.mean())

pickle.dump(KNN, open("models/modelKNN.pkl", "wb"))

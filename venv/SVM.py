from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
import pickle

filename = 'files/pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
#print(dataframe)
array = dataframe.values

X = array[:, 0:8]
Y = array[:, 8]
num_folds = 10

kfold = KFold(n_splits=10, random_state=None)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
SVM = SVC(gamma='auto')
SVM.fit(X_train, y_train)
y_pred = SVM.predict(X_test)

print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))

results = cross_val_score(SVM, X, Y, cv=kfold)
print("Mean Estimated SVM", results.mean())

pickle.dump(SVM, open("models/modelSVM.pkl", "wb"))

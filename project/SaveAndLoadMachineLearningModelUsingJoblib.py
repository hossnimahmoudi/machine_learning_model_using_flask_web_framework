#save model using joblib
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from sklearn.externals.joblib import dump
# from sklearn.externals.joblib import load
from joblib import dump
from joblib import load

#Load dataset
file = 'files/pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(file, names=names)

#print(dataframe)
array = dataframe.values

X = array[:, 0:8]
Y = array[:, 8]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=7)

#Fit the model
model = LogisticRegression(solver='liblinear')
model.fit(X_train, Y_train)

#save the model to disk
filename = 'models/finalized_joblib_model.sav'
dump(model, open(filename, 'wb'))

# copy the file to some server or other computer and then ...

#load the model from the disk
loaded_model = load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print("Result = ", result)
from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from pickle import dump

#Load dataset
file = 'files/pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(file, names=names)

#print(dataframe)
array = dataframe.values

#splitting the array to input and output
X = array[:, 0:8]
Y = array[:, 8]

#DO REQUIRED SUMMARIZATIONS, EVALUATIONS AND OPTIMIZATIONS
model = LogisticRegression(solver='liblinear')
model.fit(X, Y)

#Save this model to disk for reuse
filename = 'models/final_pima_indian.sav'
dump(model, open(filename, 'wb'))

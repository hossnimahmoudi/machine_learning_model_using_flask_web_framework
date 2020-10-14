from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from pickle import dump

#Load dataset
file = 'files/iris.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataframe = read_csv(file, names=names)

#print(dataframe)
array = dataframe.values

#splitting the array to input and output
X = array[:, 0:4]
Y = array[:, 4]

#DO REQUIRED SUMMARIZATIONS, EVALUATIONS AND OPTIMIZATIONS
# class distribution
print(dataframe.groupby('class').size())

model = LogisticRegression(multi_class='multinomial', solver='newton-cg')
model.fit(X, Y)

#save this model to disk for reuse
filename = 'models/iris.sav'
dump(model, open(filename, 'wb'))
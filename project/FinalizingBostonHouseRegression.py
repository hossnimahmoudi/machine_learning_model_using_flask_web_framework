from pandas import read_csv
from sklearn.linear_model import LinearRegression
import pickle
from pickle import dump

filename = 'files/BostonHousing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = read_csv(filename, names=names)
#print(dataframe)
array = dataframe.values

X = array[:, 0:13]
Y = array[:, 13]

model = LinearRegression()
model.fit(X, Y)

#DO REQUIRED SUMMARIZATIONS, EVALUATIONS AND OPTIMIZATIONS
# class distribution
print(dataframe.groupby('MEDV').size())

#save this model to disk for reuse
filename = 'models/final_Boston.sav'
dump(model, open(filename, 'wb'))
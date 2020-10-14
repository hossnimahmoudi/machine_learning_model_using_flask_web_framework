from matplotlib import pyplot
from pandas import read_csv
filename = 'files/pima-indians-diabetes.csv'

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)

dataframe.plot(kind='density', subplots=True)
pyplot.show()
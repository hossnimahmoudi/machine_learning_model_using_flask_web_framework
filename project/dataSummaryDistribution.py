from pandas import read_csv
filename = 'files/pima-indians-diabetes.csv'

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)

class_counts = dataframe.groupby('class').size()
print("Class Count :", class_counts)
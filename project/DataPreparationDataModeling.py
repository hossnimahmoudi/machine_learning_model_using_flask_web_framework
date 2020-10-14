from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Load dataset
file = 'files/pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(file, names=names)

#print(dataframe)
array = dataframe.values

X = array[:, 0:8]
Y = array[:, 8]

estimators=[]
estimators.append(('standardize', StandardScaler()))
estimators.append(('LDA', LinearDiscriminantAnalysis()))
model = Pipeline(estimators)

#evaluate pipeline
kfold = KFold(n_splits=10, random_state=None)
results = cross_val_score(model, X, Y, cv=kfold)
print('Mean Estimated Accuracy Linear Discriminant Analysis using Pipeline', results.mean()*100)
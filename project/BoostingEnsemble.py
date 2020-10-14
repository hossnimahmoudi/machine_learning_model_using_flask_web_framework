from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

#Load dataset
file = 'files/pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(file, names=names)

#print(dataframe)
array = dataframe.values

X = array[:, 0:8]
Y = array[:, 8]
seed = 7
num_trees = 100

kfold = KFold(n_splits=10, random_state=seed)

#Ada Boost Model
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed,)
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy Ada Boost : %f" % (results.mean()*100))


#Gradiant Boosting Model
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy Gradient Boosting : %f" % (results.mean()*100))
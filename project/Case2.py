from pickle import load

#Name of the saved model from another computer
filename = 'models/iris.sav'

#Load the file
loaded_model = load(open(filename, 'rb'))

#define one new data instance for prediction
Xnew = [[7.2, 5.1, 6.2, 1.4]]

#make a prediction
ynew = loaded_model.predict(Xnew)

print("Input = %s , Predicted = %s " % (Xnew[0], ynew[0]))
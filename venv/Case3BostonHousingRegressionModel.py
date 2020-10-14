from pickle import load
import numpy as np

#Name of the saved model from another computer
filename = 'models/final_Boston.sav'

#Load the file
loaded_model = load(open(filename, 'rb'))

#define one new data instance for prediction
Xnew = [[0.01965, 80, 1.76, "0", 0.385, 6.23, 31.5, 9.0892, 1, 241, 18.2, 341.6, 12.93]]

#convert DataType of array to float64
Xnew = np.array(Xnew, dtype=np.float64)

#make a prediction
ynew = loaded_model.predict(Xnew)

print("Input = %s , Predicted = %s " % (Xnew[0], ynew[0]))

#The predicted must be multiplied by 1000 (Dollar)
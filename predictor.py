import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tkinter import *

heart_data = pd.read_csv("heart.csv")

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

model = LogisticRegression()
model.fit(X_train, Y_train)

def get_accuracy_scores():
   X_train_prediction = model.predict(X_train)
   training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
   print("Accuracy on training data: ", training_data_accuracy)

   X_test_prediction = model.predict(X_test)
   test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
   print("Accuracy on test data: ", test_data_accuracy)

def get_type(nb):
   try:
      to_return = int(nb)
   except ValueError:
      to_return = float(nb)
   return to_return

def predict():
   text = StringVar()
   result_label = Label(gui, textvariable=text)
   result_label.pack()
   if(get_type(sexEntry.get())!=0 and get_type(sexEntry.get())!=1) or (get_type(fbsEntry.get())!=0 and get_type(fbsEntry.get())!=1) or (get_type(exangEntry.get())!=0 and get_type(exangEntry.get())!=1):
      error_label = Label(gui, text="Error, please retry")
      error_label.pack()
   else:
      input_data = (
         get_type(ageEntry.get()),
         get_type(sexEntry.get()),
         get_type(cpEntry.get()),
         get_type(trestbpsEntry.get()),
         get_type(cholEntry.get()),
         get_type(fbsEntry.get()),
         get_type(restecgEntry.get()),
         get_type(thalachEntry.get()),
         get_type(exangEntry.get()),
         get_type(oldpeakEntry.get()),
         get_type(slopeEntry.get()),
         get_type(caEntry.get()),
         get_type(thalEntry.get())
      )
      input_data_nparray = np.asarray(input_data)
      input_data_reshaped = input_data_nparray.reshape(1, -1)
      prediction = model.predict(input_data_reshaped)
      if prediction[0]==0:
         text.set("The person doesn't have a heart disease")
      else:
         text.set("The person has a heart disease")
      result_label.config(text=text)

def clean():
   error_label.destroy()
   result_label.destroy()

gui = Tk()
gui.geometry("300x700")

ageLabel = Label(gui, text="Age")
ageEntry = Entry(gui, width=20)

sexLabel = Label(gui, text="Sex")
sexEntry = Entry(gui, width=20)

cpEntry = Entry(gui, width=20)
cpLabel = Label(gui, text="cp")

trestbpsEntry = Entry(gui, width=20)
trestbpsLabel = Label(gui, text="trestbps")

cholEntry = Entry(gui, width=20)
cholLabel = Label(gui, text="chol")

fbsEntry = Entry(gui, width=20)
fbsLabel = Label(gui, text="fbs")

restecgEntry = Entry(gui, width=20)
restecgLabel = Label(gui, text="restecg")

thalachEntry = Entry(gui, width=20)
thalachLabel = Label(gui, text="thalach")

exangEntry = Entry(gui, width=20)
exangLabel = Label(gui, text="exang")

oldpeakEntry = Entry(gui, width=20)
oldpeakLabel = Label(gui, text="oldpeak")

slopeEntry = Entry(gui, width=20)
slopeLabel = Label(gui, text="slope")

caEntry = Entry(gui, width=20)
caLabel = Label(gui, text="ca")

thalEntry = Entry(gui, width=20)
thalLabel = Label(gui, text="thal")

predict_button = Button(gui, text="Predict disease", command=predict)
clean_button = Button(gui, text="Clean", command=clean)

ageLabel.pack()
ageEntry.pack()

sexLabel.pack()
sexEntry.pack()

cpLabel.pack()
cpEntry.pack()

trestbpsLabel.pack()
trestbpsEntry.pack()

cholLabel.pack()
cholEntry.pack()

fbsLabel.pack()
fbsEntry.pack()

restecgLabel.pack()
restecgEntry.pack()

thalachLabel.pack()
thalachEntry.pack()

exangLabel.pack()
exangEntry.pack()

oldpeakLabel.pack()
oldpeakEntry.pack()

slopeLabel.pack()
slopeEntry.pack()

caLabel.pack()
caEntry.pack()

thalLabel.pack()
thalEntry.pack()

predict_button.pack()
clean_button.pack()

gui.mainloop()
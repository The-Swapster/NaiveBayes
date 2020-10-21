import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class StandardScaler(object):
    def __init__(self):
        pass

    def fit(self, X): # describes the fit function of StandardScaler
        self.mean_ = np.mean(X, axis=0) # gets the mean of the columns of the dataset
        self.scale_ = np.std(X - self.mean_, axis=0) # gets the standard deviation of the columns of the dataset
        return self

    def transform(self, X): # describes the transform function of StandardScaler
        return (X - self.mean_) / self.scale_ # formula for transforming each datapoint in the dataset

    def fit_transform(self, X): # describes the fit_transform function of StandardScaler
        return self.fit(X).transform(X) # fits and transforms data at the same time


class NaiveBayes:

    def fit(self, x, y): # describes the fit function of Naive Bayes
        n_samples, n_features = x.shape # get the shape of x
        self._classes = np.unique(y) # gets unique values in y
        n_classes = len(self._classes) # number of unique values is found
        
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64) # mean is claculated for each class
        self._var = np.zeros((n_classes, n_features), dtype=np.float64) # variance is claculated for each class
        self._priors = np.zeros(n_classes, dtype=np.float64) # prior probability is claculated for each class
        
        for idx,c in enumerate(self._classes):
            x_c = x[y==c] 
            self._mean[idx,:] = x_c.mean(axis=0)
            self._var[idx,:] = x_c.var(axis=0)
            self._priors[idx] = x_c.shape[0]/ float(n_samples)

    def predict(self, x):
        y_predict = [self._predict(i) for i in x] #get the predicted class for the data
        return y_predict

    def _predict(self, x):
        posteriors = []

        for idx in range(len(self._classes)):
            prior = np.log(self._priors[idx]) 
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
    
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x): # uses gaussian function to calculate the conditional probability
        mean = self._mean[class_idx] 
        var = self._var[class_idx]
        numerator = np.exp(- (x-mean)**2 / (2 * var)) 
        denominator = np.sqrt(2 * np.pi * var)
        return (numerator/denominator) 


def accuracy_score(y_test, y_pred, normalize=True):
    correct = sum(y_test == y_pred)
    return correct/len(y_test) if normalize else correct


df = pd.read_csv("heart.csv")
print(df)


s = StandardScaler()
df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal']] = s.fit_transform(df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal']])
print(s.mean_)
print(s.scale_)


x = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal']]
y = df['target']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state = 31)# spliting the dataset into training and testing
x_test = np.array(x_test) #converting testing dataset into numpy array
model = NaiveBayes() #creating an object for NaiveBayes class
model.fit(x_train,y_train) # fitting the model
y_pred = model.predict(x_test) # predicting the model
acc = accuracy_score(y_test, y_pred) # getting the accuracy of the model
print('Accuracy: ',acc)


from tkinter import *
import tkinter.font as tkfont

def result(a): # displays the popup window for class
    t=Tk()
    t.configure(background='LemonChiffon1')
    t.title('Result')
    f = tkfont.Font(family='Consolas', size=30)
    if a[0] == 1:
        text = 'Heart disease detected, with the accuracy of '+str(acc*100)
    elif a[0] == 0:
        text = 'Heart disease not detected'
    l1 = Label(t, text=text, compound=CENTER,font=f,background='LemonChiffon1')
    l1.grid(row=1, column=0)
    
t = Tk()
t.configure(background='LemonChiffon1')
f = tkfont.Font(family='Consolas', size=10)
t.title('Cardio Disease Detector')
v1 = IntVar(t)
v2 = IntVar(t)
v3 = StringVar(t)
v4 = IntVar(t)
v5 = IntVar(t)
v6 = IntVar(t)
v7 = StringVar(t)
v8 = IntVar(t)
v9 = IntVar(t)
v10 = StringVar(t)
v11 = StringVar(t)
v12 = IntVar(t)
v13 = StringVar(t)

# creating dictionary and list for drop-down menu
cpt = {0:'Typical Angina', 1:'Atypical Angina', 2:'Non-anginal', 3:'Asymptomatic'}
ecg = {0:'Normal', 1:'St-T wave abnormality', 2:'Probable or definite left ventricular hypertropy'}
st = {0:'Upsloping', 1:'Flat', 2:'Downward Sloping'}
fl = [0,1,2,3]
thal = {1:'Normal',2:'Fixed Defect',3:'Reversible Defect'}

# for adding the labels in the GUI
text1 = Label(t,font=f,justify=RIGHT, text="Age",background='LemonChiffon1')
text2 = Label(t,font=f,justify=RIGHT, text="Sex",background='LemonChiffon1')
text3 = Label(t,font=f,justify=RIGHT, text="Chest Pain Type",background='LemonChiffon1')
text4 = Label(t,font=f,justify=RIGHT, text="Resting Blood Pressure",background='LemonChiffon1')
text5 = Label(t,font=f,justify=RIGHT, text="Serum Cholestrol in mg/dl",background='LemonChiffon1')
text6 = Label(t,font=f,justify=RIGHT, text="Fasting blood Sugar > 120 mg/dl",background='LemonChiffon1')
text7 = Label(t,font=f,justify=RIGHT, text="Resting Electrocardiographic Results",background='LemonChiffon1')
text8 = Label(t,font=f,justify=RIGHT, text="Maximum Heart Rate Achieved",background='LemonChiffon1')
text9 = Label(t,font=f,justify=RIGHT, text="Exercise Induced Angina",background='LemonChiffon1')
text10 = Label(t,font=f,justify=RIGHT, text="ST depression induced by exercise relative to rest",background='LemonChiffon1')
text11 = Label(t,font=f,justify=RIGHT, text="The slope of the peak exercise ST segment",background='LemonChiffon1')
text12 = Label(t,font=f,justify=RIGHT, text="Number of major vessels (0-3) colored by Flourosopy",background='LemonChiffon1')
text13 = Label(t,font=f,justify=RIGHT, text="Thal",background='LemonChiffon1')

# adding the entry boxes or drop down menu or radio buttons
entry1 = Entry(t,font=f,textvariable=v1)
r1 = Radiobutton(t, text="Male", font=f, background='LemonChiffon1', variable=v2, value=1)
r2 = Radiobutton(t, text="Female", font=f, background='LemonChiffon1', variable=v2, value=0)
om1 = OptionMenu(t, v3, *cpt.values())
om1.config(font=f)
v3.set('Typical Angina')
entry2 = Entry(t,font=f,textvariable=v4)
entry3 = Entry(t,font=f,textvariable=v5)
r3 = Radiobutton(t, text="Yes", font=f, background='LemonChiffon1', variable=v6, value=1)
r4 = Radiobutton(t, text="No", font=f, background='LemonChiffon1', variable=v6, value=0)
om2 = OptionMenu(t, v7, *ecg.values())
om2.config(font=f)
v7.set('Normal')
entry4 = Entry(t,font=f,textvariable=v8)
r5 = Radiobutton(t, text="Yes", font=f, background='LemonChiffon1', variable=v9, value=1)
r6 = Radiobutton(t, text="No", font=f, background='LemonChiffon1', variable=v9, value=0)
entry5 = Entry(t,font=f,textvariable=v10)
om3 = OptionMenu(t, v11, *st.values())
om3.config(font=f)
v11.set('Upsloping')
om4 = OptionMenu(t, v12, *fl)
om4.config(font=f)
v12.set(fl[0])
om5 = OptionMenu(t, v13, *thal.values())
om5.config(font=f)
v13.set('Normal')

# aligning the elements in the GUI  
text1.grid(row=1, column=0) 
text2.grid(row=2, column=0)
text3.grid(row=3, column=0)
text4.grid(row=4, column=0)
text5.grid(row=5, column=0)
text6.grid(row=6, column=0)
text7.grid(row=7, column=0)
text8.grid(row=8, column=0)
text9.grid(row=9, column=0)
text10.grid(row=10, column=0)
text11.grid(row=11, column=0)
text12.grid(row=12, column=0)
text13.grid(row=13, column=0)
entry1.grid(row=1, column=2)
r1.grid(row=2,column=2)
r2.grid(row=2,column=3)
om1.grid(row=3,column=2)
entry2.grid(row=4, column=2)
entry3.grid(row=5, column=2)
r3.grid(row=6,column=2)
r4.grid(row=6,column=3)
om2.grid(row=7,column=2)
entry4.grid(row=8, column=2)
r5.grid(row=9,column=2)
r6.grid(row=9,column=3)
entry5.grid(row=10, column=2)
om3.grid(row=11,column=2)
om4.grid(row=12,column=2)
om5.grid(row=13,column=2)

# Get the key and value of the dictionaries
cpt_key=list(cpt.keys())
cpt_value=list(cpt.values())
ecg_key=list(ecg.keys())
ecg_value=list(ecg.values())
st_key=list(st.keys())
st_value=list(st.values())
thal_key=list(thal.keys())
thal_value=list(thal.values())

# adding a button in the GUI
b = Button(t, text='Predict',font=f, command=lambda: [result(model.predict(np.array(s.transform(pd.DataFrame([[float(v1.get()),float(v2.get()),float(cpt_key[cpt_value.index(v3.get())]),float(v4.get()),float(v5.get()),float(v6.get()),float(ecg_key[ecg_value.index(v7.get())]),float(v8.get()),float(v9.get()),float(v10.get()),float(st_key[st_value.index(v11.get())]),float(v12.get()),float(thal_key[thal_value.index(v13.get())])]],columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal']))))), print([[float(v1.get()),float(v2.get()),float(cpt_key[cpt_value.index(v3.get())]),float(v4.get()),float(v5.get()),float(v6.get()),float(ecg_key[ecg_value.index(v7.get())]),float(v8.get()),float(v9.get()),float(v10.get()),float(st_key[st_value.index(v11.get())]),float(v12.get()),float(thal_key[thal_value.index(v13.get())])]])])
b.grid(row=14, column=1)
t.mainloop()

"""
Created on Sat Nov  4 23:47:31 2023

@author: Brenda García, María José Merino
"""
#%% LIBRERÍAS
import pandas as pd
from Sammon import Sammon_Mapping as Sammon

#%% CARGAR ARCHIVO

df = pd.read_csv("seeds.txt", sep='\t', header=None, names=[
    "area A", "perimeter P", "compactness C", "length of kernel",
    "width of kernel", "asymmetry coefficient", "length of kernel groove", "class" ])


#%% MAPEO DE SAMMON

X = df.drop(columns=["class"])
Y = df["class"]

X_mapped, E = Sammon(X, 2, Y)


#%% CLASIFICACIÓN
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix


X_train, X_test, y_train, y_test = train_test_split(X_mapped, Y, test_size=0.3, random_state=5)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)


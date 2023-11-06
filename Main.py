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

#%% EVALUACIÓN
import seaborn as sns
import matplotlib.pyplot as plt

print('_' * 55)  
print('Resultados con los componentes principales:')
print(f'Precisión: {accuracy}')
print(f'Informe de clasificación:\n{classification_report}')
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicciones')
plt.ylabel('Etiquetas Reales')
plt.title('Matriz de Confusión')
plt.show()

#%% PROYECCIÓN FINAL

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.palettes import Category10
from bokeh.transform import factor_cmap

# Crear un DataFrame con los datos mapeados
df_mapped = pd.DataFrame(X_mapped, columns=['componente_1', 'componente_2'])
df_mapped['class'] = Y

class_labels = {1: 'Class 1', 2: 'Class 2', 3: 'Class 3'} 
df_mapped['class'] = df_mapped['class'].map(class_labels)


# Crear una figura Bokeh para la visualización
source = ColumnDataSource(df_mapped)

p = figure(title="Proyección Final", x_axis_label='Componente 1', y_axis_label='Componente 2')

color_map = factor_cmap('class', palette=Category10[10], factors=sorted(df_mapped['class'].unique()))

p.circle('componente_1', 'componente_2', size=8, source=source, legend_field='class', color=color_map)

show(p)

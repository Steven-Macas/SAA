"""Ejercicios de desarrollo. Dataframes y preproceso

1. Crea un dataframe con 1000 registros para tres datos con valores aleatorios generados
• x → entre 20 y 50

• y → entre 50 y 150

• z → entre 10000 y 40000"""
#Importamos las librerias necesarias
import numpy as np
import pandas as pd
import random
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#Creamos los 1000 registros con los rangos que nos piden para cada atributo
x = []
y = []
z = []
for a in range(1000):
  x.append(random.randint(20, 50))
for b in range(1000):
  y.append(random.randint(50, 150))
for c in range(1000):
  z.append(random.randint(10000, 40000))

#Creamos una matriz con los datos de los tres arrays de antes y lo convertimos a DataFrame
data = {
    'Edad':x,
    'Peso':y,
    'Salario':z
}
data = pd.DataFrame(data)

#Como los valores de cada columna se ve bien porque hay variedad 
#y las medias sale algo normal para el rango que tienen
#lo guardamos para volver a trabajar con ella más adelante
#data.to_csv('dataFrame.csv')

data = pd.read_csv('dataFrame.csv')

#2. Muestra los datos estadísticos de cada variable (media, desviación típica, máx, mín, etc.)

#Media de cada columna
mean_edad = data['Edad'].mean()
mean_peso = data['Peso'].mean()
mean_salario = data['Salario'].mean()
print('Media de Edad: ', mean_edad)
print('Media de Peso: ', mean_peso)
print('Media de Salario: ', mean_salario)

#Moda de cada columna
moda_peso = data['Peso'].mode()[0]
moda_edad = data['Edad'].mode()[0]
moda_salario = data['Salario'].mode()[0]
print('Moda de Edad:', moda_edad)
print('Moda de Peso:', moda_peso)
print('Moda de Salario:', moda_salario)

#Mediana de cada columna
median_edad = data['Edad'].median()
median_peso = data['Peso'].median()
median_salario = data['Salario'].median()
print('Mediana de Edad: ', median_edad)
print('Mediana de Peso: ', median_peso)
print('Mediana de Salario: ', median_salario)

#Valor máximo y mínimo
max_edad = data['Edad'].max()
max_peso = data['Peso'].max()
max_salario = data['Salario'].max()
min_edad = data['Edad'].min()
min_peso = data['Peso'].min()
min_salario = data['Salario'].min()
print('Máximo de Edad: ', max_edad, '.\n Mínimo de Edad: ', min_edad)
print('Máximo de Peso: ', max_peso, '.\n Mínimo de Peso: ', min_peso)
print('Máximo de Salario: ', max_salario, '.\n Mínimo de Salario: ', min_salario)

#Desviación de cada columna
desv_edad = data['Edad'].std()
desv_peso = data['Peso'].std()
desv_salario = data['Salario'].std()
print('Desviación de Edad: ', desv_edad)
print('Desviación de Peso: ', desv_peso)
print('Desviación de Salario: ', desv_salario)

#Estandarizacion
a = (55-median_peso)/desv_peso
b = (70-median_peso)/desv_peso
prob_peso = st.norm.cdf(b) - st.norm.cdf(a)
a2 = (20000-median_salario)/desv_salario
b2 = (30000-median_salario)/desv_salario
prob_salario = st.norm.cdf(b2) - st.norm.cdf(a2)
print('Probabilidad de que el Peso esté entre 55-70 es ', prob_peso)
print('Probabilidad de que el Salario esté entre 20000-30000 es ', prob_salario)
"""
4. Realiza dos funciones que reciban un dataframe y un nombre de columna y devuelva una tupla con los valores (XL, XN) correspondiente al intervalo al cual si no pertenece un valor puede ser considerado como outlier"""

def bandas(data,columna,k):
  #Calculamos los cuartiles, 1 y 3
  Q1  = np.quantile(data[columna], 0.25)
  Q3  = np.quantile(data[columna], 0.75)
  IQR = Q3 - Q1

  #Calculamos los extremos
  xL = Q1 - k * IQR
  xU = Q3 + k * IQR
  result = [xL,xU]
  return result

result = bandas(data,'Edad',1.5)
result

#5. Introduce deliberadamente dos valores fuera de los rangos en alguno de los atributos (x, y, z) y, usando el método jackknife detecte las observaciones influyentes para la media. Haz lo mismo para la mediana. Comenta los resultados obtenidos.
#añadimos dos valores fueras del rango
data.loc[1000] = [1001,1290,80,25000]
data.loc[1001] = [1002,1290,85,26000]

data.tail()

#calculamos la media y mediana con los dos valores que hemos introducido.

media = np.mean(data['Edad'])
mediana = np.median(data['Edad'])
print('Media: ', media, '. Mediana: ',mediana)

#Ahora ejecutamos la media quitando un dato cada vez

edad_arr= data['Edad'].to_numpy()

medias = np.zeros(len(edad_arr))
for i in range(len(edad_arr)):
  datos_sin_i = np.delete(edad_arr,i)
  medias[i]=np.mean(datos_sin_i)

#Calculamos los cuartiles, 1 y 3
Q1  = np.quantile(medias, 0.25)
Q3  = np.quantile(medias, 0.75)
IQR = Q3 - Q1
k=1
#Calculamos los extremos
xL = Q1 - k * IQR
xU = Q3 + k * IQR


for i in range(len(edad_arr)):
  if medias[i] < xL or medias[i] > xU:
    print(f"La media[{i}] = {medias[i]} es influyente")

#6. Utiliza MinMaxScaler para realizar un escalado del dataframe creado en el ejercicio 1. Muestra los resultados y comenta que ha sucedido.

scaler = MinMaxScaler()
print(scaler.fit(data))
print(scaler.data_max_)
print(scaler.transform(data))
# -*- coding: utf-8 -*-
"""Modelo 1 - Despliegue

- Cargamos el modelo
- Cargamos los datos futuros
- Preparar los datos futuros: dummies
- Aplicamos el modelo para la predicción
"""

#Cargamos librerías principales

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

#Cargamos el modelo
import pickle
filename = Path(__file__).with_name('modelo_1.pkl')
if not filename.exists():
    st.error('Falta modelo_1.pkl. Genera el archivo con python modelo_1.py --solo-exportar y colócalo junto a app.py.')
    st.stop()
with open(filename, 'rb') as archivo:
    modelo, variables, entradas, estados, error = pickle.load(archivo)


#Cargamos los datos futuros
#Los valores se ingresan desde la interfaz gráfica

#interfaz grafica

st.title('Predicción del estado de una carga')

#Mismos nombres y orden de las variables del modelo original
datos = []
for variable, valor in entradas.items():
    if isinstance(valor, list):
        valor = st.selectbox(variable, valor)
    elif isinstance(valor, int):
        valor = st.number_input(variable, value=valor, step=1)
    else:
        valor = st.number_input(variable, value=valor)
    datos.append(valor)


#Dataframe
data = pd.DataFrame([datos], columns=list(entradas)) #Dataframe con los mismos nombres de variables

#Se realiza la preparación de datos
data_preparada=data.copy()

#En despliegue drop_first= False
data_preparada = pd.get_dummies(data_preparada, columns=['Mode', 'Equipment', 'General_Pricing_Category', 'Origin_Zip'], drop_first=False, dtype=int)
data_preparada = pd.get_dummies(data_preparada, columns=['Roll_Over','Prebook','Stop_Type', 'FallOff'], drop_first=False, dtype=int)
data_preparada.head()

#Se adicionan las columnas faltantes
data_preparada=data_preparada.reindex(columns=variables,fill_value=0)
data_preparada.head()

#Este árbol no utiliza normalización
#En los despliegues no se llama fit

"""PREDICCIONES"""

#Hacemos la predicción con el Tree
Y_pred = modelo.predict(data_preparada)
print(Y_pred)

#Convertimos las dummies de Status a los nombres originales
predicciones = pd.DataFrame(np.asarray(Y_pred).reshape(len(data), -1), columns=estados[1:])
data['Prediccion']=predicciones.idxmax(axis=1)
data.loc[predicciones.sum(axis=1) == 0, 'Prediccion']=estados[0]
data.head()

st.write(data)

#Porcentaje de error de clasificación medido sobre el 30% de prueba
st.warning(f'El modelo tiene un error del {error:.2%} en el conjunto de prueba (30%).')

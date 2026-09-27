# -*- coding: utf-8 -*-
"""Modelo 1 - Despliegue

- Cargamos el modelo entrenado
- Cargamos los datos futuros desde Streamlit
- Preparar los datos futuros: dummies
- Aplicamos el modelo para la predicción
"""

#Cargamos librerías principales

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

#Cargamos el modelo
import pickle
filename = Path(__file__).with_name('modelo_1.pkl')
if not filename.exists():
    st.error('Coloca el archivo modelo_1.pkl del proyecto en la misma carpeta que app.py.')
    st.stop()
with open(filename, 'rb') as archivo:
    modelo, variables, entradas, estados, error = pickle.load(archivo)

#Variables de entrada del CSV, en el mismo orden; Status es la variable objetivo
columnas = ['Mode', 'Equipment', 'Miles', 'CarrierPay', 'CustomerPay',
            'Roll_Over', 'Prebook', 'General_Pricing_Category', 'Stop_Type',
            'Origin_Zip', 'Destination_Zip', 'Total_Drops', 'HasTracking', 'FallOff']

if list(entradas) != columnas or list(modelo.feature_names_in_) != list(variables):
    st.error('El modelo_1.pkl cargado no corresponde a las variables de este proyecto.')
    st.stop()

#Cargamos los datos futuros
#Los valores se ingresan desde la interfaz gráfica

#interfaz grafica

st.title('Predicción del estado de una carga')

Mode = st.selectbox('Mode', entradas['Mode'])
Equipment = st.selectbox('Equipment', entradas['Equipment'])
Miles = st.number_input('Miles', value=entradas['Miles'])
CarrierPay = st.number_input('CarrierPay', value=entradas['CarrierPay'])
CustomerPay = st.number_input('CustomerPay', value=entradas['CustomerPay'])
Roll_Over = st.selectbox('Roll_Over', entradas['Roll_Over'])
Prebook = st.selectbox('Prebook', entradas['Prebook'])
General_Pricing_Category = st.selectbox('General_Pricing_Category', entradas['General_Pricing_Category'])
Stop_Type = st.selectbox('Stop_Type', entradas['Stop_Type'])
Origin_Zip = st.selectbox('Origin_Zip', entradas['Origin_Zip'])
Destination_Zip = st.number_input('Destination_Zip', value=entradas['Destination_Zip'], step=1)
Total_Drops = st.number_input('Total_Drops', value=entradas['Total_Drops'], step=1)
HasTracking = st.selectbox('HasTracking', [0, 1], index=int(entradas['HasTracking']))
FallOff = st.selectbox('FallOff', entradas['FallOff'])


#Dataframe
datos = [[Mode, Equipment, Miles, CarrierPay, CustomerPay, Roll_Over, Prebook,
          General_Pricing_Category, Stop_Type, Origin_Zip, Destination_Zip,
          Total_Drops, HasTracking, FallOff]]
data = pd.DataFrame(datos, columns=columnas) #Dataframe con los mismos nombres de variables

#Se realiza la preparación de datos
data_preparada=data.copy()

#En despliegue drop_first= False
data_preparada = pd.get_dummies(data_preparada, columns=['Mode', 'Equipment', 'General_Pricing_Category', 'Origin_Zip'], drop_first=False, dtype=int)
data_preparada = pd.get_dummies(data_preparada, columns=['Roll_Over','Prebook','Stop_Type', 'FallOff'], drop_first=False, dtype=int)

#Se adicionan las columnas faltantes
data_preparada=data_preparada.reindex(columns=variables,fill_value=0)

"""PREDICCIONES"""

#Hacemos la predicción con el Tree ya entrenado
Y_pred = modelo.predict(data_preparada)

#Convertimos las dummies de Status a los nombres originales
predicciones = pd.DataFrame(np.asarray(Y_pred).reshape(len(data), -1), columns=estados[1:])
data['Prediccion']=predicciones.idxmax(axis=1)
data.loc[predicciones.sum(axis=1) == 0, 'Prediccion']=estados[0]

st.subheader('Resultado')
st.write(data)

#Error del conjunto de prueba guardado con el modelo
st.warning(f'El modelo tiene un error del {error:.2%} en el conjunto de prueba (30%).')


st.write(data)

#Porcentaje de error de clasificación medido sobre el 30% de prueba
st.warning(f'El modelo tiene un error del {error:.2%} en el conjunto de prueba (30%).')

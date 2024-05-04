# Código para ejecutar la app:
# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Cargar el modelo entrenado
with open('models/regression_model.pkl', 'rb') as file:
    model = pickle.load(file) # Tuvo que haber creado con la misma versión de s

# Encabezado de la app
st.write("""
# Predicción de precio de casas
Este es un estimador de precios de casas creado con un modelo de Machine Learning.
""")

# Sidebar
st.sidebar.header('Datos de entrada')

default_values = {'MedInc': 3.84, 
                  'HouseAge': 52.0, 
                  'AveRooms': 6.28, 
                  'AveBedrms': 1.08, 
                  'Population': 565, 
                  'AveOccup': 2.18, 
                  'Latitude': 37.85, 
                  'Longitude': -122.25}

# Función para obtener los datos de entrada creando un campo de entrada por cada característica
def user_input_features():    
    data = {} # Diccionario para almacenar los datos de entrada
    for key, value in default_values.items():
        input_value = st.sidebar.number_input(f'Enter {key}:', value=value)
        data[key] = input_value   
    df_features = pd.DataFrame(data, index=[0])
    return df_features

# Obtener los datos de entrada
df = user_input_features()

# Mostrar los datos de entrada
st.subheader('Datos de entrada')
st.dataframe(df, hide_index=True)

# Realizar la predicción
pred = model.predict(df.values)
pred_in_dollars = pred[0]*100000

# Mostrar la predicción
st.subheader('Predicción')
# Dividir la pantalla en dos columnas con proporción 40% y 60%
left_column, right_column = st.columns([0.4, 0.6])
# Mostrar la predicción en la columna izquierda
with left_column:
    st.write("El precio estimado de la casa es:")
    st.info(f"${pred_in_dollars:,.2f}")
# Columna derecha reservada para información adicional (vacía por ahora)
with right_column:
    pass
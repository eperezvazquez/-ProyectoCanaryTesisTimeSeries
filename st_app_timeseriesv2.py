##MODELO DE TIME SERIES
import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
import base64
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


st.title('📈Machine Learning y analítica aplicada al portafolio de proyectos')

"""
El objetivo general es poder generar un cuadro de mando integral, del área de Planificación estratégica en particular de los portafolios de AGESIC, donde permita tener indicadores a la hora de la toma de decisiones. 
En este caso hemos desarollado un modelo de series temporales (Time Series) que busca poder predecir el presupuesto futuo en un periodo de tiempo tanto para pesos como para dolares.

"""
st.image ('https://www.springboard.com/blog/wp-content/uploads/2022/02/data-scientist-without-a-degree-2048x1366.jpg')
st.subheader('Modelo de time series pesos')

"""
### Step 1: Importing Data
"""
st.sidebar.image('https://www.gub.uy/agencia-gobierno-electronico-sociedad-informacion-conocimiento/sites/agencia-gobierno-electronico-sociedad-informacion-conocimiento/files/catalogo/iso.png')
st.sidebar.subheader('¿Que es prophet?')
st.sidebar.write('Prophet “es un procedimiento para pronosticar datos de series de tiempo basado en un modelo aditivo en el que las tendencias no lineales se ajustan a la estacionalidad anual, semanal y diaria, más los efectos de las vacaciones. Funciona mejor con series de tiempo que tienen fuertes efectos estacionales y varias temporadas de datos históricos. Prophet es robusto ante los datos faltantes y los cambios en la tendencia, y por lo general maneja bien los valores atípicos.”Con esta herramienta, el equipo de ciencia de datos de Facebook buscaba lograr los siguientes objetivos:Lograr modelos de pronóstico rápidoz y precisos.Obtener pronósticos razonablemente correctos de manera automática.')
"""

"""
df_pagos = pd.read_csv("src/pagos_moneda_filtro_campos.csv", engine="python", sep=',', quotechar='"', error_bad_lines=False)
df_pagos_dolares = pd.read_csv("src/pagos_moneda_filtro_campos.csv", engine="python", sep=',', quotechar='"', error_bad_lines=False)
indexNames = df_pagos[df_pagos['pag_confirmar'].isnull()].index
df_pagos.drop(indexNames,inplace=True)
indexNames = df_pagos[df_pagos['pag_confirmar']==0].index
df_pagos.drop(indexNames,inplace=True)

# figure(figsize=(8, 6), dpi=80)
explode = (0.1, 0.1, 0)
fig1, ax1 = plt.subplots()
non_nombre_grf = [67,32,1]
nombres = ["Pesos","Dolares","Euros"]
colores = ["#191970","#FFD700","#1E90FF"]
desfase = (0,0,0)
ax1.pie(non_nombre_grf ,explode=explode, labels=nombres, autopct="%0.1f %%", colors=colores)
ax1.axis("equal")
st.pyplot(fig1)


df_pagos['pag_fecha_planificada'] = pd.to_datetime(df_pagos['pag_fecha_planificada'])
df_pagos['pag_fecha_real'] = pd.to_datetime(df_pagos['pag_fecha_real'])

indexNames = df_pagos[df_pagos['mon_pk']==2].index
df_pagos.drop(indexNames,inplace=True)

pagos_modelo_pesos=df_pagos.drop(['pag_pk','pag_fecha_planificada','pag_importe_planificado','pag_confirmar','mon_pk','mon_nombre'],axis=1)
pagos_modelo_pesos.shape #ver si va
indexNames = pagos_modelo_pesos[pagos_modelo_pesos['pag_importe_real']<=3019.0].index
pagos_modelo_pesos.drop(indexNames,inplace=True)
indexNames = pagos_modelo_pesos[pagos_modelo_pesos['pag_importe_real']>1200000].index
pagos_modelo_pesos.drop(indexNames,inplace=True)

pagos_modelo_pesos.sort_values(['pag_fecha_real', 'pag_importe_real'],ascending=False) 
pagos_modelo_pesos #ver si va
sp = pagos_modelo_pesos.rename(columns={'pag_fecha_real': 'ds','pag_importe_real': 'y'})
sp_sample = sp[(sp.ds.dt.year>2014)]

max_date = sp_sample['ds'].max()

def custom_forecast_plot():
    forecast_length = 30

    prior_df = sp[(sp.ds.dt.year>2014)]
    forecast_df = sp[(sp.ds.dt.year==2021) & (sp.ds.dt.month==1)]
    all_df = pd.concat([prior_df, forecast_df]).sort_values('ds')
    all_df.head()

    all_df_sample = all_df[-forecast_length*5:]
    forecast_sample = forecast[forecast['ds'].isin(all_df["ds"].values)].sort_values('ds')

    prior_vis_df = forecast_sample[-forecast_length*5:-forecast_length]
    forecast_vis_df = forecast_sample[-forecast_length:]

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)

    ax.plot(all_df_sample['ds'].dt.to_pydatetime(), all_df_sample["y"], '.k')

    ax.plot(prior_vis_df['ds'].dt.to_pydatetime(), prior_vis_df['yhat'], ls='-', c='#0072B2')
    ax.fill_between(prior_vis_df['ds'].dt.to_pydatetime(), prior_vis_df['yhat_lower'], prior_vis_df['yhat_upper'], color='#0072B2', alpha=0.2)

    ax.plot(forecast_vis_df['ds'].dt.to_pydatetime(), forecast_vis_df['yhat'], ls='-', c='#fc7d0b')
    ax.fill_between(forecast_vis_df['ds'].dt.to_pydatetime(), forecast_vis_df['yhat_lower'], forecast_vis_df['yhat_upper'], color='#fc7d0b', alpha=0.2)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)

    ax.set_title('S&P 500 30-Day Forecast')
    # plt.show(sns)
    st.pyplot(fig)


###

"""
### Step 2: Seleccionar horizonte de previsión
Tenga en cuenta que los pronósticos se vuelven menos precisos con horizontes de pronóstico más grandes."""

periods_input = st.number_input('¿Cuántos días le gustaría pronosticar a futuro?',
min_value = 1, max_value = 730)



"""
### Step 3: Visualizar datos de previsión
La siguiente imagen muestra valores pronosticados futuros. "Mean_predict" es el valor predicho y los límites superior e inferior son (de forma predeterminada) intervalos de confianza del 80 %.
"""

model1 = Prophet(interval_width=0.95, weekly_seasonality=False,seasonality_prior_scale=0.001,changepoint_prior_scale=0.9)
model1.add_seasonality(name='yearly', period=365, fourier_order=8)
model1.add_country_holidays(country_name='UY')
model1.fit(sp_sample)
# INICIO original
# m = Prophet(interval_width=0.95, weekly_seasonality=False, changepoint_prior_scale=0.9)
# m.add_seasonality(name='yearly', period=365, fourier_order=8)
# m.add_country_holidays(country_name='US')
# m.fit(data)
# FIN original
future = model1.make_future_dataframe(periods=periods_input)

forecast = model1.predict(future)   
#forecast = m.predict(future)
fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

fcst_filtered =  fcst[fcst['ds'] > max_date] 
fcst_filtered = fcst_filtered.rename(columns={'ds': 'Date','yhat': 'Mean_predict', 'yhat_lower': 'Low_prediction', 'yhat_upper': 'High_prediction'})
st.write(fcst_filtered)


"""
Las siguientes imágenes muestran una tendencia de valores de alto nivel, tendencias de días de la semana y tendencias anuales (si el conjunto de datos cubre varios años).
"""

fig2 = model1.plot_components(forecast)
st.write(fig2)  

"""
La siguiente imagen muestra los valores reales (puntos negros) y predichos (línea azul) a lo largo del tiempo. El área sombreada en azul representa los intervalos de confianza superior e inferior.
"""

fig1 = model1.plot(forecast)
st.write(fig1)

"""
La siguiente imágen muestra el modelo final ajustado en pesos.
"""
custom_forecast_plot()


metric_df = forecast.set_index('ds')[['yhat']].join(sp_sample.set_index('ds').y).reset_index()
metric_df.dropna(inplace=True)

"""
El r-cuadrado es cercano a 1 por lo que podemos concluir que el ajuste es bueno.
Un valor de r-cuadrado superior a 0,9 es sorprendente (y probablemente demasiado bueno para ser verdad, lo que me dice que es muy probable que estos datos estén sobreajustados).
El valor obtenido es:
"""
st.write(r2_score(metric_df.y, metric_df.yhat))
"""
MSE:
"""
st.write(mean_squared_error(metric_df.y, metric_df.yhat))
"""
Ese es un valor de MSE grande... y confirma mi sospecha de que estos datos están sobreajustados y es probable que no se mantengan en el futuro. Recuerde... para MSE, más cerca de cero es mejor.
Y finalmente, resultado MAE:
"""
st.write(mean_absolute_error(metric_df.y, metric_df.yhat))

"""
### Step 4: Download the Forecast Data
The below link allows you to download the newly created forecast to your computer for further analysis and use.
"""

csv_exp = fcst_filtered.to_csv(index=False)
    # When no file name is given, pandas returns the CSV as a string, nice.
b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'
st.markdown(href, unsafe_allow_html=True)

#--------------------------DOLARES--------------------

st.subheader('Modelo de time series dolares')

"""
### Step 1: Importing Data
"""

"""

"""
# df_pagos_dolares = pd.read_csv("pagos_moneda_filtro_campos.csv", engine="python", sep=',', quotechar='"', error_bad_lines=False)

indexNames = df_pagos_dolares[df_pagos_dolares['pag_confirmar'].isnull()].index
df_pagos_dolares.drop(indexNames,inplace=True)
indexNames = df_pagos_dolares[df_pagos_dolares['pag_confirmar']==0].index
df_pagos_dolares.drop(indexNames,inplace=True)

# figure(figsize=(8, 6), dpi=80)
#explode = (0.1, 0.1, 0)
#fig1, ax1 = plt.subplots()
#non_nombre_grf = [67,32,1]
#nombres = ["Pesos","Dolares","Euros"]
#colores = ["#EE6055","#60D394","#5574ee"]
#desfase = (0,0,0)
#ax1.pie(non_nombre_grf ,explode=explode, labels=nombres, autopct="%0.1f %%", colors=colores)
#ax1.axis("equal")
#st.pyplot(fig1)

df_pagos_dolares['pag_fecha_planificada'] = pd.to_datetime(df_pagos_dolares['pag_fecha_planificada'])
df_pagos_dolares['pag_fecha_real'] = pd.to_datetime(df_pagos_dolares['pag_fecha_real'])

indexNames = df_pagos_dolares[df_pagos_dolares['mon_pk']==1 & 3].index
df_pagos_dolares.drop(indexNames,inplace=True)

df_pagos_dolares=df_pagos_dolares.drop(['pag_pk','pag_fecha_planificada','pag_importe_planificado','pag_confirmar','mon_pk','mon_nombre'],axis=1)
df_pagos_dolares.shape #ver si va
indexNames = df_pagos_dolares[df_pagos_dolares['pag_importe_real']==0].index
df_pagos_dolares.drop(indexNames,inplace=True)
indexNames = df_pagos_dolares[df_pagos_dolares['pag_importe_real']>500000].index
df_pagos_dolares.drop(indexNames,inplace=True)
df_pagos_dolares['pag_fecha_real'] = pd.to_datetime(df_pagos_dolares['pag_fecha_real'])
df_pagos_dolares.sort_values(['pag_fecha_real', 'pag_importe_real'],ascending=False) 
df_pagos_dolares

sp_d = df_pagos_dolares.rename(columns={'pag_fecha_real': 'ds','pag_importe_real': 'y'})
sp_sample1 = sp_d[(sp_d.ds.dt.year>2014)]

max_date = sp_sample1['ds'].max()

def custom_forecast_plot_dol():
    forecast_length = 30

    prior_df = sp_d[(sp_d.ds.dt.year>2014)]
    forecast_df = sp_d[(sp_d.ds.dt.year==2021) & (sp_d.ds.dt.month==1)]
    all_df = pd.concat([prior_df, forecast_df]).sort_values('ds')
    all_df.head()

    all_df_sample = all_df[-forecast_length*5:]
    forecast_sample = forecast[forecast['ds'].isin(all_df["ds"].values)].sort_values('ds')

    prior_vis_df = forecast_sample[-forecast_length*5:-forecast_length]
    forecast_vis_df = forecast_sample[-forecast_length:]

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)

    ax.plot(all_df_sample['ds'].dt.to_pydatetime(), all_df_sample["y"], '.k')

    ax.plot(prior_vis_df['ds'].dt.to_pydatetime(), prior_vis_df['yhat'], ls='-', c='#0072B2')
    ax.fill_between(prior_vis_df['ds'].dt.to_pydatetime(), prior_vis_df['yhat_lower'], prior_vis_df['yhat_upper'], color='#0072B2', alpha=0.2)

    ax.plot(forecast_vis_df['ds'].dt.to_pydatetime(), forecast_vis_df['yhat'], ls='-', c='#fc7d0b')
    ax.fill_between(forecast_vis_df['ds'].dt.to_pydatetime(), forecast_vis_df['yhat_lower'], forecast_vis_df['yhat_upper'], color='#fc7d0b', alpha=0.2)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)

    ax.set_title('S&P 500 30-Day Forecast')
    # plt.show(sns)
    st.pyplot(fig)


###

"""
### Step 2: Seleccionar horizonte de previsión
Tenga en cuenta que los pronósticos se vuelven menos precisos con horizontes de pronóstico más grandes."""

periods_input_dol = st.number_input('¿Cuántos días le gustaría pronosticar a futuro en dólares?',
min_value = 1, max_value = 730)



"""
### Step 3: Visualizar datos de previsión
La siguiente imagen muestra valores pronosticados futuros. "Mean_predict" es el valor predicho y los límites superior e inferior son (de forma predeterminada) intervalos de confianza del 80 %.
"""

final_model_dolares = Prophet(interval_width=0.95, weekly_seasonality=False,seasonality_prior_scale=0.001,changepoint_prior_scale=0.9)
final_model_dolares .add_seasonality(name='yearly', period=365, fourier_order=8)
final_model_dolares .add_country_holidays(country_name='UY')
forecast = final_model_dolares.fit(sp_sample1).predict(future)
fig = final_model_dolares .plot(forecast)

# INICIO original
# m = Prophet(interval_width=0.95, weekly_seasonality=False, changepoint_prior_scale=0.9)
# m.add_seasonality(name='yearly', period=365, fourier_order=8)
# m.add_country_holidays(country_name='US')
# m.fit(data)
# FIN original
future = final_model_dolares.make_future_dataframe(periods=periods_input_dol)

forecast = final_model_dolares.predict(future)   
#forecast = m.predict(future)
fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

fcst_filtered =  fcst[fcst['ds'] > max_date] 
fcst_filtered = fcst_filtered.rename(columns={'ds': 'Date','yhat': 'Mean_predict', 'yhat_lower': 'Low_prediction', 'yhat_upper': 'High_prediction'})
st.write(fcst_filtered)



"""
Las siguientes imágenes muestran una tendencia de valores de alto nivel, tendencias de días de la semana y tendencias anuales (si el conjunto de datos cubre varios años).
"""

fig2 = final_model_dolares.plot_components(forecast)
st.write(fig2)  

"""
La siguiente imagen muestra los valores reales (puntos negros) y predichos (línea azul) a lo largo del tiempo. El área sombreada en azul representa los intervalos de confianza superior e inferior.
"""

fig1 = final_model_dolares.plot(forecast)
st.write(fig1)

"""
La siguiente imágen muestra el modelo final ajustado en dólares.
"""
custom_forecast_plot_dol()
# custom_forecast_plot()

metric_df = forecast.set_index('ds')[['yhat']].join(sp_sample1.set_index('ds').y).reset_index()
metric_df.dropna(inplace=True)

"""
El r-cuadrado es cercano a 1 por lo que podemos concluir que el ajuste es bueno.
Un valor de r-cuadrado superior a 0,9 es sorprendente (y probablemente demasiado bueno para ser verdad, lo que me dice que es muy probable que estos datos estén sobreajustados).
El valor obtenido es:
"""
st.write(r2_score(metric_df.y, metric_df.yhat))
"""
MSE:
"""
st.write(mean_squared_error(metric_df.y, metric_df.yhat))
"""
Ese es un valor de MSE grande... y confirma mi sospecha de que estos datos están sobreajustados y es probable que no se mantengan en el futuro. Recuerde... para MSE, más cerca de cero es mejor.
Y finalmente, resultado MAE:
"""
st.write(mean_absolute_error(metric_df.y, metric_df.yhat))



"""
Para los siguientes casos, haremos predicciones para cada empresa, utilizando también un modelo multivariable. Por ejemplo, modelos MLP Multivariante (https://www.youtube.com/watch?v=87c9D_41GWg)
"""

"""
### Step 4: Download the Forecast Data
The below link allows you to download the newly created forecast to your computer for further analysis and use.
"""

csv_exp = fcst_filtered.to_csv(index=False)
    # When no file name is given, pandas returns the CSV as a string, nice.
b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'
st.markdown(href, unsafe_allow_html=True)

st.image('http://i3campus.co/CONTENIDOS/es-cnbguatemala/content/images/a/a7/buz%25c3%25b3n_de_correo.png')
st.write('Por mas informacion nos puede escribir canarysoftware@gmail.com.')
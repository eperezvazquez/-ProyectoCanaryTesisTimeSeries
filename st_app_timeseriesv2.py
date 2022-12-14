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

"""
La siguiente imágen muestra en pesos la tendencia del pornostico
"""
custom_forecast_plot()


metric_df = forecast.set_index('ds')[['yhat']].join(sp_sample.set_index('ds').y).reset_index()
metric_df.dropna(inplace=True)

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
La imagen muestra el modelo final ajustado en pesos
"""

fig1 = model1.plot(forecast)
st.write(fig1)


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
"""
La siguiente imágen muestra el pronostico doloares.
"""
custom_forecast_plot_dol()
# custom_forecast_plot()

metric_df = forecast.set_index('ds')[['yhat']].join(sp_sample1.set_index('ds').y).reset_index()
metric_df.dropna(inplace=True)

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
Muestra el modelo ajustado en dolares:
"""

fig1 = final_model_dolares.plot(forecast)
st.write(fig1)


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

st.image('data:image/jpeg;base64,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')
st.write('Por mas informacion nos puede escribir canarysoftware@gmail.com.')

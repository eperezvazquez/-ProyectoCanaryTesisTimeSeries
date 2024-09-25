#PASAMOS A PRODUCCION EL MODELO TIME SERIES
#Aplicamos los imports primero
# linear algebra
import numpy as np 

#Pandas profile
#import pandas_profiling as pp # exploratory data analysis EDA

# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd 

# data visualization
import matplotlib.pyplot as plt # data visualisation
import matplotlib.dates as mdates
import seaborn as sns #data visualisation
import plotly.express as px #data visualisation
import plotly.graph_objects as go
from scipy.stats import chi2_contingency, norm # Calculo de chi2

# stocks related missing info
#import yfinance as yf

# ignoring the warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
#ranking the stocks
import matplotlib.pyplot as plt
import matplotlib.dates as dates
#import optuna
from wordcloud import WordCloud, STOPWORDS #es para la nube de palabras

# Evaluar si las elimino
import statsmodels.api as sm

#Timer series
import datetime

#!pip install fbprophet --quiet
import plotly.offline as py
py.init_notebook_mode()

#Guardar modelo
import pickle

#Aplicamos los from luego de los imports

# Evaluar si las elimino
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from matplotlib.pyplot import figure
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
from sklearn.metrics import mean_absolute_error

#ranking the stocks
from plotly.subplots import make_subplots

#Prophet Model Stuff
#!pip install fbprophet --quiet

from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.plot import plot_yearly
from prophet.plot import add_changepoints_to_plot
from pickle import FALSE


#PUESTA EN PRODUCCION DE TIME SERIES PESOS

df_pagos = pd.read_csv("src\pagos_moneda_filtro_campos2024.csv", engine="python", sep=',', quotechar='"', error_bad_lines=False)
#Eliminan las filas de esos registros
indexNames = df_pagos[df_pagos['pag_confirmar'].isnull()].index
# Delete these row indexes from dataFrame
df_pagos.drop(indexNames,inplace=True)
#Eliminan las filas de esos registros de los no confirmados.
indexNames = df_pagos[df_pagos['pag_confirmar']==0].index
# Delete these row indexes from dataFrame
df_pagos.drop(indexNames,inplace=True)
df_indice_recet= df_pagos.reset_index()

# Scatter plot comparing planned and real payments
plt.figure(figsize=(10, 6))
plt.scatter(df_pagos['pag_importe_planificado'], df_pagos['pag_importe_real'], alpha=0.5, color='purple')
plt.xlabel('Planned Payment Amount')
plt.ylabel('Real Payment Amount')
plt.title('Comparison of Planned vs. Real Payments')
plt.show()
# Box plot for planned vs. real payments
plt.figure(figsize=(10, 6))
df_pagos[['pag_importe_planificado', 'pag_importe_real']].plot(kind='box')
plt.title('Box Plot of Planned vs. Real Payments')
plt.show()

#Cambio de los tipo objetos a tipo fecha cambiar aca
df_pagos['pag_fecha_planificada'] = pd.to_datetime(df_pagos['pag_fecha_planificada'])
df_pagos['pag_fecha_real'] = pd.to_datetime(df_pagos['pag_fecha_real'])
df_pagos.head()
#Eliminan las filas de esos registros
indexNames = df_pagos[df_pagos['mon_pk']==2].index
# Delete these row indexes from dataFrame
df_pagos.drop(indexNames,inplace=True)
pagos_modelo_pesos=df_pagos.drop(['pag_pk','pag_fecha_planificada','pag_importe_planificado','pag_confirmar','mon_pk','mon_nombre'],axis=1)
pagos_modelo_pesos.shape
#Eliminan las filas de esos registros por debaje del 1% del valor medior dado que se suprimen 8% de registros es despeciable.
indexNames = pagos_modelo_pesos[pagos_modelo_pesos['pag_importe_real']<=3019.0].index
# Delete these row indexes from dataFrame
pagos_modelo_pesos.drop(indexNames,inplace=True)
#Eliminamos las filtas por encima de 12 millones dado que son los outlier del boxplot.
indexNames = pagos_modelo_pesos[pagos_modelo_pesos['pag_importe_real']>1200000].index
# Delete these row indexes from dataFrame
pagos_modelo_pesos.drop(indexNames,inplace=True)
pagos_modelo_pesos.sort_values(['pag_fecha_real', 'pag_importe_real'],ascending=False) 
pagos_modelo_pesos
sp = pagos_modelo_pesos.rename(columns={'pag_fecha_real': 'ds','pag_importe_real': 'y'})
sp_sample = sp[(sp.ds.dt.year>2014)]

# Create a figure and axis with a larger size
fig, ax = plt.subplots(figsize=(14, 8))

# Scatter plot with improved aesthetics
ax.scatter(x=pagos_modelo_pesos['pag_fecha_real'], y=pagos_modelo_pesos['pag_importe_real'], 
           color='blue', alpha=0.6, edgecolor='k', s=100)

# Set y-axis limit
plt.ylim(3031.0, 1000000.0)

# Filter dates to show from 2014 onwards
ax.set_xlim(pd.Timestamp('2014-01-01'), pagos_modelo_pesos['pag_fecha_real'].max())

# Format the date on the x-axis to show years only
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())

# Rotate the date labels for better readability
plt.xticks(rotation=45)

# Add labels and title
ax.set_xlabel('Real Payment Date (Year)', fontsize=14)
ax.set_ylabel('Real Payment Amount', fontsize=14)
ax.set_title('Scatter Plot of Real Payments Over Time (2014 Onwards)', fontsize=16)

# Add a grid
ax.grid(True, linestyle='--', alpha=0.7)

# Save the improved plot as a PNG with better quality
plt.savefig('diagrama-dispersion-2014-adelante.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

#modelo
model1 = Prophet(interval_width=0.95)
model1.add_country_holidays(country_name='UY')
model1.fit(sp_sample)

future = model1.make_future_dataframe(periods=30, freq="B")
forecast = model1.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig = model1.plot(forecast)
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
    plt.show(sns)
    
fig = model1.plot_components(forecast)
fig = model1.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), model1, forecast)

final_model_pesos = Prophet(
    interval_width=0.95, 
    weekly_seasonality=False,  # Activando la estacionalidad semanal
    seasonality_prior_scale=10,  # Incrementando la flexibilidad para capturar la estacionalidad
    changepoint_prior_scale=0.9  # Menos flexible con los cambios en la tendencia
)
final_model_pesos.add_seasonality(name='yearly', period=365, fourier_order=8)
final_model_pesos.add_country_holidays(country_name='UY')
forecast = final_model_pesos.fit(sp_sample).predict(future)
fig = final_model_pesos.plot(forecast)
#MAE
# Aseguramos que ambos DataFrames estén alineados por 'ds' (fecha)
forecast_aligned = forecast.set_index('ds').join(sp_sample.set_index('ds'), how='inner', lsuffix='_pred', rsuffix='_real')

# Alinear los DataFrames por las fechas 'ds' (fecha)
forecast_aligned = forecast.set_index('ds').join(sp_sample.set_index('ds'), how='inner', lsuffix='_pred', rsuffix='_real')

# Verificar las primeras filas para asegurarse de que la alineación es correcta
print(forecast_aligned[['y', 'yhat']].head())

# Eliminar posibles valores nulos después de la alineación
forecast_aligned = forecast_aligned.dropna(subset=['y', 'yhat'])

# Calcular el MAE
mae = mean_absolute_error(forecast_aligned['y'], forecast_aligned['yhat'])

print(f'MAE: {mae:.2f}')

# Se guarda el modelo de regresion que se pasa a produccion.
filename ='timeseriesprodpesos.sav'
pickle.dump(final_model_pesos, open(filename, 'wb'))

#PUESTA EN PRODUCCION DE TIME SERIES DOLARES

df_pagos_dolares = pd.read_csv("src\pagos_moneda_filtro_campos2024.csv", engine="python", sep=',', quotechar='"', error_bad_lines=False)
#Eliminan las filas de esos registros
indexNames = df_pagos_dolares[df_pagos_dolares['pag_confirmar'].isnull()].index
# Delete these row indexes from dataFrame
df_pagos_dolares.drop(indexNames,inplace=True)
#Eliminan las filas de esos registros de los no confirmados.
indexNames = df_pagos_dolares[df_pagos_dolares['pag_confirmar']==0].index
# Delete these row indexes from dataFrame
df_pagos_dolares.drop(indexNames,inplace=True)
#Eliminan las filas de esos registros
indexNames = df_pagos_dolares[df_pagos_dolares['mon_pk']==1 & 3].index
# Delete these row indexes from dataFrame
df_pagos_dolares.drop(indexNames,inplace=True)
df_pagos_dolares=df_pagos_dolares.drop(['pag_pk','pag_fecha_planificada','pag_importe_planificado','pag_confirmar','mon_pk','mon_nombre'],axis=1)
df_pagos_dolares.shape
#Eliminan las filas de esos registros por debaje del 1% del valor medior dado que se suprimen 8% de registros es despeciable.
indexNames = df_pagos_dolares[df_pagos_dolares['pag_importe_real']==0].index
# Delete these row indexes from dataFrame
df_pagos_dolares.drop(indexNames,inplace=True)
#Eliminan las filas de esos registros por debaje del 1% del valor medior dado que se suprimen 8% de registros es despeciable.
indexNames = df_pagos_dolares[df_pagos_dolares['pag_importe_real']>500000].index
# Delete these row indexes from dataFrame
df_pagos_dolares.drop(indexNames,inplace=True)
#Cambio de los tipo objetos a tipo fecha 
df_pagos_dolares['pag_fecha_real'] = pd.to_datetime(df_pagos_dolares['pag_fecha_real'])
df_pagos_dolares.head()
df_pagos_dolares.sort_values(['pag_fecha_real', 'pag_importe_real'],ascending=False) 
df_pagos_dolares
sp = df_pagos_dolares.rename(columns={'pag_fecha_real': 'ds','pag_importe_real': 'y'})
sp_sample1 = sp[(sp.ds.dt.year>2014)]

# Rename columns for easier plotting
sp = df_pagos_dolares.rename(columns={'pag_fecha_real': 'ds', 'pag_importe_real': 'y'})

# Filter data for years greater than 2014
sp_sample1 = sp[sp['ds'].dt.year > 2014]

# Create the figure and axes with a larger size
fig, ax = plt.subplots(figsize=(14, 8))

# Draw scatter plot with improved aesthetics
ax.scatter(x=sp_sample1['ds'], y=sp_sample1['y'], color='purple', alpha=0.7, edgecolor='k', s=80)

# Set y-axis limit
plt.ylim(3031.0, 400000.0)

# Format the date on the x-axis to show years only
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())

# Rotate the date labels for better readability
plt.xticks(rotation=45)

# Add labels and title
ax.set_xlabel('Real Payment Date (Year)', fontsize=14)
ax.set_ylabel('Real Payment Amount (USD)', fontsize=14)
ax.set_title('Scatter Plot of Real Payments in Dollars Over Time (2015 Onwards)', fontsize=16)

# Add a grid
ax.grid(True, linestyle='--', alpha=0.7)

# Save the improved plot as a PNG with better quality
plt.savefig('diagrama-dispersion-dolares-2015-onwards.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

model11 = Prophet(interval_width=0.95)
model11.add_country_holidays(country_name='UY')
model11.fit(sp_sample1)

future = model11.make_future_dataframe(periods=30, freq="B")
forecast = model11.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig = model11.plot(forecast)

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
    plt.show(sns)
    
final_model_dolares = Prophet(interval_width=0.95, weekly_seasonality=False,seasonality_prior_scale=0.001,changepoint_prior_scale=0.9)
final_model_dolares .add_seasonality(name='yearly', period=365, fourier_order=8)
final_model_dolares .add_country_holidays(country_name='UY')
forecast = final_model_dolares.fit(sp_sample1).predict(future)
fig = final_model_dolares .plot(forecast)

# Se guarda el modelo de regresion que se pasa a produccion.
filename ='timeseriesprodDOLARESproduccion.sav'
pickle.dump(final_model_dolares, open(filename, 'wb'))

# Heroku uses the last version of python, but it conflicts with 
# some dependencies. Low your version by adding a runtime.txt file
# https://stackoverflow.com/questions/71712258/
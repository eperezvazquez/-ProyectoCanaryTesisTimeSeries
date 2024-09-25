# MODELO DE TIME SERIES
!pip install runtime.txt
!pip install streamlit
!pip install requirements.txt
!pip install seaborn
!pip install statsmodels
!pip install yfinance
!pip install optuna
!pip install --upgrade pip
!pip install Cython
!pip install fbprophet --quiet
!python -m pip install prophet
!capture
!pip install fbprophet
!pip install prophet
!pip install pystan==2.19.1.1 prophet
!pip install --upgrade plotl
!pip install pmdarima
!pip install pandas-profiling
!pip install ipywidgets
!pip install wordcloud
!pip install figure
!python3 -m pip install scikit-learn
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

st.title('📈 Machine Learning and Analytics Applied to the Project Portfolio')

"""
The main objective is to create a comprehensive dashboard for the Strategic Planning area, specifically for the AGESIC portfolios, to have indicators available for decision-making.
In this case, we have developed a time series model that aims to predict the future budget over a period of time, both in pesos and in dollars.
"""
st.image('https://www.springboard.com/blog/wp-content/uploads/2022/02/data-scientist-without-a-degree-2048x1366.jpg')
st.subheader('Time Series Model in Pesos')

"""
### Step 1: Importing Data
"""
st.sidebar.image('https://www.gub.uy/agencia-gobierno-electronico-sociedad-informacion-conocimiento/sites/agencia-gobierno-electronico-sociedad-informacion-conocimiento/files/catalogo/iso.png')
st.sidebar.subheader('What is Prophet?')
st.sidebar.write('Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and generally handles outliers well. With this tool, the Facebook data science team aimed to achieve quick and accurate forecasting models and obtain reasonably accurate forecasts automatically.')

df_pagos = pd.read_csv("pagos_moneda_filtro_campos2024.csv", engine="python", sep=',', quotechar='"', error_bad_lines=False)
df_pagos_dolares = pd.read_csv("pagos_moneda_filtro_campos2024.csv", engine="python", sep=',', quotechar='"', error_bad_lines=False)
indexNames = df_pagos[df_pagos['pag_confirmar'].isnull()].index
df_pagos.drop(indexNames, inplace=True)
indexNames = df_pagos[df_pagos['pag_confirmar'] == 0].index
df_pagos.drop(indexNames, inplace=True)

# Updated data with correct values
data = {
    'Currency': ['Pesos', 'Dollars', 'Indexed Units', 'Euros', 'Adjustable Units'],
    'Count': [15337, 7999, 315, 7, 1],
    'Percentage': [64.83, 33.81, 1.33, 0.03, 0.00]
}

# Creating a bar plot
plt.figure(figsize=(10, 6))
bars = plt.barh(data['Currency'], data['Percentage'], color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#673AB7'])

# Adding labels and title
plt.xlabel('Percentage (%)')
plt.ylabel('Currency')
plt.title('Percentage Distribution of Different Currencies')
plt.xlim(0, 70)

# Adding data labels to each bar
for bar in bars:
    plt.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.2f}%', va='center', ha='left', fontsize=12)

# Display the plot in Streamlit
st.pyplot(plt)

# Additional Streamlit components
st.subheader('Time Series Model in Pesos')

df_pagos['pag_fecha_planificada'] = pd.to_datetime(df_pagos['pag_fecha_planificada'])
df_pagos['pag_fecha_real'] = pd.to_datetime(df_pagos['pag_fecha_real'])

indexNames = df_pagos[df_pagos['mon_pk'] == 2].index
df_pagos.drop(indexNames, inplace=True)

pagos_modelo_pesos = df_pagos.drop(['pag_pk', 'pag_fecha_planificada', 'pag_importe_planificado', 'pag_confirmar', 'mon_pk', 'mon_nombre'], axis=1)
pagos_modelo_pesos.shape
indexNames = pagos_modelo_pesos[pagos_modelo_pesos['pag_importe_real'] <= 3019.0].index
pagos_modelo_pesos.drop(indexNames, inplace=True)
indexNames = pagos_modelo_pesos[pagos_modelo_pesos['pag_importe_real'] > 1200000].index
pagos_modelo_pesos.drop(indexNames, inplace=True)

pagos_modelo_pesos.sort_values(['pag_fecha_real', 'pag_importe_real'], ascending=False)
pagos_modelo_pesos
sp = pagos_modelo_pesos.rename(columns={'pag_fecha_real': 'ds', 'pag_importe_real': 'y'})
sp_sample = sp[(sp.ds.dt.year > 2014)]

max_date = sp_sample['ds'].max()

def custom_forecast_plot():
    forecast_length = 30

    prior_df = sp[(sp.ds.dt.year > 2014)]
    forecast_df = sp[(sp.ds.dt.year == 2021) & (sp.ds.dt.month == 1)]
    all_df = pd.concat([prior_df, forecast_df]).sort_values('ds')
    all_df.head()

    all_df_sample = all_df[-forecast_length * 5:]
    forecast_sample = forecast[forecast['ds'].isin(all_df["ds"].values)].sort_values('ds')

    prior_vis_df = forecast_sample[-forecast_length * 5:-forecast_length]
    forecast_vis_df = forecast_sample[-forecast_length:]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    ax.plot(all_df_sample['ds'].dt.to_pydatetime(), all_df_sample["y"], '.k')

    ax.plot(prior_vis_df['ds'].dt.to_pydatetime(), prior_vis_df['yhat'], ls='-', c='#0072B2')
    ax.fill_between(prior_vis_df['ds'].dt.to_pydatetime(), prior_vis_df['yhat_lower'], prior_vis_df['yhat_upper'], color='#0072B2', alpha=0.2)

    ax.plot(forecast_vis_df['ds'].dt.to_pydatetime(), forecast_vis_df['yhat'], ls='-', c='#fc7d0b')
    ax.fill_between(forecast_vis_df['ds'].dt.to_pydatetime(), forecast_vis_df['yhat_lower'], forecast_vis_df['yhat_upper'], color='#fc7d0b', alpha=0.2)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)

    ax.set_title('S&P 500 30-Day Forecast')
    st.pyplot(fig)

"""
### Step 2: Select Forecast Horizon
Please note that forecasts become less accurate with longer forecast horizons.
"""

periods_input = st.number_input('How many days would you like to forecast into the future?',
min_value=1, max_value=730)

"""
### Step 3: Visualize Forecast Data
The following image shows predicted future values. "Mean_predict" is the predicted value, and the upper and lower limits are (by default) 80% confidence intervals.
"""

model1 = Prophet(interval_width=0.95,
    weekly_seasonality=False,  # Enabling weekly seasonality
    seasonality_prior_scale=10,  # Increasing flexibility to capture seasonality
    changepoint_prior_scale=0.9)  # Less flexible with trend changes
model1.add_seasonality(name='yearly', period=365, fourier_order=8)
model1.add_country_holidays(country_name='UY')
model1.fit(sp_sample)

future = model1.make_future_dataframe(periods=periods_input)
forecast = model1.predict(future)
fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

fcst_filtered = fcst[fcst['ds'] > max_date]
fcst_filtered = fcst_filtered.rename(columns={'ds': 'Date', 'yhat': 'Mean_predict', 'yhat_lower': 'Low_prediction', 'yhat_upper': 'High_prediction'})
st.write(fcst_filtered)

"""
The following images show a high-level trend, weekday trends, and annual trends (if the dataset covers multiple years).
"""

fig2 = model1.plot_components(forecast)
st.write(fig2)

"""
The following image shows the actual values (black dots) and predicted values (blue line) over time. The blue shaded area represents the upper and lower confidence intervals.
"""

fig1 = model1.plot(forecast)
st.write(fig1)

"""
The following image shows the final adjusted model in pesos.
"""
custom_forecast_plot()

metric_df = forecast.set_index('ds')[['yhat']].join(sp_sample.set_index('ds').y).reset_index()
metric_df.dropna(inplace=True)

"""
R-squared is close to 1, so we can conclude that the fit is good.
An R-squared value above 0.9 is surprising (and probably too good to be true, suggesting that these data are likely overfitted).
The value obtained is:
"""
st.write(r2_score(metric_df.y, metric_df.yhat))
"""
MSE:
"""
st.write(mean_squared_error(metric_df.y, metric_df.yhat))
"""
This is a large MSE value... and it confirms my suspicion that these data are overfitted and likely won't hold in the future. Remember... for MSE, closer to zero is better.
And finally, the MAE result:
"""
st.write(mean_absolute_error(metric_df.y, metric_df.yhat))

"""
### Step 4: Download the Forecast Data
The below link allows you to download the newly created forecast to your computer for further analysis and use.
"""

csv_exp = fcst_filtered.to_csv(index=False)
b64 = base64.b64encode(csv_exp.encode()).decode()
href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'
st.markdown(href, unsafe_allow_html=True)

#--------------------------DOLLARS--------------------

st.subheader('Time Series Model in Dollars')

"""
### Step 1: Importing Data
"""

indexNames = df_pagos_dolares[df_pagos_dolares['pag_confirmar'].isnull()].index
df_pagos_dolares.drop(indexNames, inplace=True)
indexNames = df_pagos_dolares[df_pagos_dolares['pag_confirmar'] == 0].index
df_pagos_dolares.drop(indexNames, inplace=True)

df_pagos_dolares['pag_fecha_planificada'] = pd.to_datetime(df_pagos_dolares['pag_fecha_planificada'])
df_pagos_dolares['pag_fecha_real'] = pd.to_datetime(df_pagos_dolares['pag_fecha_real'])

indexNames = df_pagos_dolares[df_pagos_dolares['mon_pk'] == 1 & 3].index
df_pagos_dolares.drop(indexNames, inplace=True)

df_pagos_dolares = df_pagos_dolares.drop(['pag_pk', 'pag_fecha_planificada', 'pag_importe_planificado', 'pag_confirmar', 'mon_pk', 'mon_nombre'], axis=1)
df_pagos_dolares.shape
indexNames = df_pagos_dolares[df_pagos_dolares['pag_importe_real'] == 0].index
df_pagos_dolares.drop(indexNames, inplace=True)
indexNames = df_pagos_dolares[df_pagos_dolares['pag_importe_real'] > 500000].index
df_pagos_dolares.drop(indexNames, inplace=True)
df_pagos_dolares['pag_fecha_real'] = pd.to_datetime(df_pagos_dolares['pag_fecha_real'])
df_pagos_dolares.sort_values(['pag_fecha_real', 'pag_importe_real'], ascending=False)
df_pagos_dolares

sp_d = df_pagos_dolares.rename(columns={'pag_fecha_real': 'ds', 'pag_importe_real': 'y'})
sp_sample1 = sp_d[(sp_d.ds.dt.year > 2014)]

max_date = sp_sample1['ds'].max()

def custom_forecast_plot_dol():
    forecast_length = 30

    prior_df = sp_d[(sp_d.ds.dt.year > 2014)]
    forecast_df = sp_d[(sp_d.ds.dt.year == 2021) & (sp_d.ds.dt.month == 1)]
    all_df = pd.concat([prior_df, forecast_df]).sort_values('ds')
    all_df.head()

    all_df_sample = all_df[-forecast_length * 5:]
    forecast_sample = forecast[forecast['ds'].isin(all_df["ds"].values)].sort_values('ds')

    prior_vis_df = forecast_sample[-forecast_length * 5:-forecast_length]
    forecast_vis_df = forecast_sample[-forecast_length:]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    ax.plot(all_df_sample['ds'].dt.to_pydatetime(), all_df_sample["y"], '.k')

    ax.plot(prior_vis_df['ds'].dt.to_pydatetime(), prior_vis_df['yhat'], ls='-', c='#0072B2')
    ax.fill_between(prior_vis_df['ds'].dt.to_pydatetime(), prior_vis_df['yhat_lower'], prior_vis_df['yhat_upper'], color='#0072B2', alpha=0.2)

    ax.plot(forecast_vis_df['ds'].dt.to_pydatetime(), forecast_vis_df['yhat'], ls='-', c='#fc7d0b')
    ax.fill_between(forecast_vis_df['ds'].dt.to_pydatetime(), forecast_vis_df['yhat_lower'], forecast_vis_df['yhat_upper'], color='#fc7d0b', alpha=0.2)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)

    ax.set_title('S&P 500 30-Day Forecast')
    st.pyplot(fig)

"""
### Step 2: Select Forecast Horizon
Please note that forecasts become less accurate with longer forecast horizons.
"""

periods_input_dol = st.number_input('How many days would you like to forecast into the future in dollars?',
min_value=1, max_value=730)

"""
### Step 3: Visualize Forecast Data
The following image shows predicted future values. "Mean_predict" is the predicted value, and the upper and lower limits are (by default) 80% confidence intervals.
"""

final_model_dolares = Prophet(interval_width=0.95, weekly_seasonality=False, seasonality_prior_scale=0.001, changepoint_prior_scale=0.9)
final_model_dolares.add_seasonality(name='yearly', period=365, fourier_order=8)
final_model_dolares.add_country_holidays(country_name='UY')
forecast = final_model_dolares.fit(sp_sample1).predict(future)
fig = final_model_dolares.plot(forecast)

future = final_model_dolares.make_future_dataframe(periods=periods_input_dol)
forecast = final_model_dolares.predict(future)
fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

fcst_filtered = fcst[fcst['ds'] > max_date]
fcst_filtered = fcst_filtered.rename(columns={'ds': 'Date', 'yhat': 'Mean_predict', 'yhat_lower': 'Low_prediction', 'yhat_upper': 'High_prediction'})
st.write(fcst_filtered)

"""
The following images show a high-level trend, weekday trends, and annual trends (if the dataset covers multiple years).
"""

fig2 = final_model_dolares.plot_components(forecast)
st.write(fig2)

"""
The following image shows the actual values (black dots) and predicted values (blue line) over time. The blue shaded area represents the upper and lower confidence intervals.
"""

fig1 = final_model_dolares.plot(forecast)
st.write(fig1)

"""
The following image shows the final adjusted model in dollars.
"""
custom_forecast_plot_dol()

metric_df = forecast.set_index('ds')[['yhat']].join(sp_sample1.set_index('ds').y).reset_index()
metric_df.dropna(inplace=True)

"""
R-squared is close to 1, so we can conclude that the fit is good.
An R-squared value above 0.9 is surprising (and probably too good to be true, suggesting that these data are likely overfitted).
The value obtained is:
"""
st.write(r2_score(metric_df.y, metric_df.yhat))
"""
MSE:
"""
st.write(mean_squared_error(metric_df.y, metric_df.yhat))
"""
This is a large MSE value... and it confirms my suspicion that these data are overfitted and likely won't hold in the future. Remember... for MSE, closer to zero is better.
And finally, the MAE result:
"""
st.write(mean_absolute_error(metric_df.y, metric_df.yhat))

"""
For the following cases, we will make predictions for each company, also using a multivariable model. For example, Multivariate MLP models (https://www.youtube.com/watch?v=87c9D_41GWg)
"""

"""
### Step 4: Download the Forecast Data
The below link allows you to download the newly created forecast to your computer for further analysis and use.
"""

csv_exp = fcst_filtered.to_csv(index=False)
b64 = base64.b64encode(csv_exp.encode()).decode()
href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'
st.markdown(href, unsafe_allow_html=True)

st.image('http://i3campus.co/CONTENIDOS/es-cnbguatemala/content/images/a/a7/buz%25c3%25b3n_de_correo.png')
st.write('Por mas informacion nos puede escribir canarysoftware@gmail.com.')

#For run streamlit desde termianl
# 1.Estar en la carpeta de streamlit cd... por ejemplo en este caso es cd src
# luego de estar en la carpetar ejecutar el comando: streamlit run st_app_timeseriesv2.py
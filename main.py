from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
import streamlit as st
import pandas as pd
import numpy as np

def ln_regressor(df):
    x = df['x'].values.reshape(-1,1)
    y = df['y'].values.reshape(-1,1)
    linear_regressor = LinearRegression()
    return linear_regressor.fit(x, y)


st.markdown("<h1 style='text-align: center;'><img src='https://umg.edu.gt/privacidad/img/logo.png'/></h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: grey;'>Regresion Linear</h1>", unsafe_allow_html=True)
st.divider()
st.subheader('Suba sus datos')
file = st.file_uploader(label='Cargue su archivo .csv', type='csv', help='Archivo csv con formato (x,y)')

if file is not None:
    df = pd.read_csv(file)
    df = df.sort_values('x')

    st.caption('Previsualizacion de los datos:')
    st.dataframe(df.head())

    st.divider()
    st.subheader('Grafico de datos')
    fig = px.line(df, x='x', y='y', markers=True)
    st.plotly_chart(fig)

    st.divider()
    st.subheader('Formula')
    model = ln_regressor(df)
    m = model.coef_[0][0]
    c = model.intercept_[0]
    label = r'$x = %0.4f*y %+0.4f$'%(m,c)
    st.header(label)

    st.divider()
    st.subheader('Regresion Linear 7u7')    
    x = df['x']
    y = df['y']
    x_range = np.linspace(x.min(), x.max(), 100).reshape(-1,1)
    y_range = model.predict(x_range)

    fig = go.Figure([
        go.Scatter(x = x, y = y, name='Current values'),    
        go.Scatter(x = x_range.squeeze(), y = y_range.squeeze(), name = 'Regresion Linear')
    ])
    st.plotly_chart(fig)

    st.divider()
    st.subheader('Resultados de la regresion')
    ols = sm.OLS(x, y)
    ols_result = ols.fit()
    ols_summary = ols_result.summary()
    st.write(ols_summary)










# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 22:10:21 2021

@author: Sandeep Menon
"""
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import cross_val_score
warnings.filterwarnings('ignore')
import numpy as np
import altair as alt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error,accuracy_score,r2_score
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_model import ARIMA
import math


# defining functions for plotting and prediction
def predictions_plot(test,predicted):
    plt.figure(figsize=(16,10))
    plt.plot(test, color='yellow',label='Real FB Stock Price')
    plt.plot(predicted, color='black',label='Predicted FB Stock Price')
    plt.title('Facebook Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Facebook Stock Price')
    plt.legend()
    plt.show()



def LR(df): 
    df.drop(['Date'],axis=1,inplace=True)
    
    X = df.loc[:,df.columns!='Close/Last']
    y=df['Close/Last']
    y = pd.DataFrame(y)
    ss = StandardScaler()
    X_scaled= ss.fit_transform(X)
    X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=43)
    
    reg=LinearRegression()
    reg.fit(X_train,y_train)
    
    y_pred=reg.predict(X_test)
    
    Y = y_test.to_numpy()
    st.markdown("")
    st.write("Predictor variables: Open Price , Highest Price, Lowest Price and Volume")
    st.markdown("")
    st.write("Target Variable: Close Price")
    st.markdown("")
    st.write("""#### After Performing Regression:""")
    st.markdown("")
    
    
    col1,col2 = st.columns(2)    
    col1.success('Actual Values')
    col1.write(Y)
    
    col2.success('Predicted Values')
    col2.write(y_pred)
    st.markdown("")
    res = pd.DataFrame({'Actual': Y.flatten(), 'Predicted': y_pred.flatten()}, columns=['Actual', 'Predicted'])
    columns = st.multiselect(label='Select Actual or Predicted column or both', options=res.columns,default=["Predicted"])
    st.line_chart(res[columns])
    
    
    st.markdown("")
    st.write(""" ### Model Evaluation:""") 
    st.markdown('')
    MAE = mean_absolute_error(y_test,y_pred)
    st.write("Mean Absolute Error:",MAE)
    rscore =cross_val_score(reg,X_scaled,y, cv=10, scoring="r2").mean()
    st.write("R2 Score: ",rscore)
    mse = mean_squared_error(y_test,y_pred)
    st.write("Mean Squared Error: ",mse)
    RSME = np.sqrt(mean_squared_error(y_test,y_pred))
    st.write("Root Mean Squared Error:",RSME)
    st.markdown("")
    st.write(""" ### Try Predicting:""") 
    st.markdown("")
    #vol = st.text_input("Volume")
    col3,col4= st.columns(2)
    vol = col3.text_input("Volume",0) 

    opn = col4.text_input("Open",0)
    
    col5,col6 =st.columns(2)
   
    hgh = col5.text_input("High",0)
    lw = col6.text_input("Low",0)
    st.markdown("")
    a=reg.predict(ss.transform([[vol,opn,hgh,lw,4,8,2021]]))
    st.markdown("")
    
    if st.button('Predict'):
        st.markdown("")
        st.write ("""#### Predicted Close Value:  """,round(a[0][0],2))
    
def LS(df):
    df.drop(['Date'],axis=1,inplace=True)
    
    X = df.iloc[:,1:5]
    y=df['Close/Last']
    

    # scaling the feature dataset
    scaler_x = MinMaxScaler (feature_range=(0, 1))
    x1 = np.array(X).reshape((len(X),4))
    x1 = scaler_x.fit_transform(x1)

    # scaling the target variable
    scaler_y =MinMaxScaler (feature_range=(0, 1))
    y1 = np.array(y).reshape((len(y), 1))
    y1 = scaler_y.fit_transform(y1)

    X_train1,X_test1,y_train1,y_test1=train_test_split(x1,y1,test_size=0.2,random_state=43)
    
    # converting into 3d for lstm
    X_train =X_train1.reshape(X_train1.shape[0],X_train1.shape[1] , 1)
    X_test = X_test1.reshape(X_test1.shape[0],X_test1.shape[1] , 1)
    
    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(4,1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.fit(X_train,y_train1,validation_data=(X_test,y_test1),epochs=100,batch_size=64,verbose=1)
    
    #prediction and check performance metrics
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    
    y4 = scaler_y.inverse_transform(y_test1)
    y5 = scaler_y.inverse_transform(test_predict)
    
    col1,col2 = st.columns(2)    
    col1.success('Actual Values')
    col1.write(y4)
    
    col2.success('Predicted Values')
    col2.write(y5)
    st.markdown("")
    
    res = pd.DataFrame({'Actual': y4.flatten(), 'Predicted': y5.flatten()}, columns=['Actual', 'Predicted'])
    columns = st.multiselect(label='Select Actual or Predicted column or both', options=res.columns,default=["Predicted"])
    st.line_chart(res[columns])
    
   
    st.markdown("")
    st.write(""" ### Model Evaluation:""") 
    st.markdown('')
    MAE = mean_absolute_error(y4,y5)
    st.write("Mean Absolute Error:",MAE)
    RSME = np.sqrt(mean_squared_error(y4,y5))
    st.write("Root Mean Squared Error:",RSME)
    rscore = r2_score(y_test1,model.predict(X_test))
    st.write("R2 Score: ",rscore)
    st.markdown("")
    

def AR(df):
    df1 = df[['Close/Last']].copy()
    st.write(df1)
    # Check if price series is stationary


    result = adfuller(df['Close/Last'].dropna())
    st.write(f"ADF Statistic: {result[0]}")
    st.write(f"p-value: {result[1]}")
    
    st.line_chart(df['Close/Last'])
    
    # ARIMA Model
    model = ARIMA(df['Close/Last'], order=(6, 1, 3))
    result = model.fit(disp=0)
    
    n = int(len(df) * 0.8)
    train = df['Close/Last'][:n]
    test = df['Close/Last'][n:]
    
    step = 30

    model = ARIMA(train, order=(6, 1, 3))
    result = model.fit(disp=0)
    
    # Forecast
    fc, se, conf = result.forecast(step)
    
    fc = pd.Series(fc, index=test[:step].index)
    lower = pd.Series(conf[:, 0], index=test[:step].index)
    upper = pd.Series(conf[:, 1], index=test[:step].index)
    
    
    col1,col2 = st.columns(2)    
    col1.success('Actual Values')
    col1.write(test[:step])
    
    col2.success('Predicted Values')
    col2.write(fc)
    st.markdown("")
    
    Y = test[:step]
    Y = Y.to_numpy()
    
    pred1 = fc.to_numpy()
    res = pd.DataFrame({'Actual': Y.flatten(), 'Predicted': pred1.flatten()}, columns=['Actual', 'Predicted'])
    
    st.write('Actual vs Predicted')
    st.line_chart(res)
    
    st.markdown("")
    st.write(""" ### Model Evaluation:""") 
    st.markdown('')
    
    mse = mean_squared_error(test[:step], fc)
    st.write('Mean Squared Error:  ', mse)
    mae = mean_absolute_error(test[:step], fc)
    st.write('Mean Absolute Error: ',mae)
    rmse = math.sqrt(mean_squared_error(test[:step], fc))
    st.write('Root Mean Squared Error: ',rmse)
    mape = np.mean(np.abs(fc - test[:step])/np.abs(test[:step]))
    st.write('Mean Absolute Percentage Error: ',mape)
    
    
    
    
def graphs(df):
    st.line_chart(df['Open'])
    st.line_chart(df['High'])
    st.line_chart(df['Low'])
    
st.title("Stock Price Predictor")
st.markdown("")
comp_name = st.selectbox("Select Company Dataset", ("Facebook","Apple","Amazon","Tesla","Cisco","Microsoft","Starbucks","Qualcomm"))
st.write(" ## Company : ", comp_name)

def dataset(company_name):
    rg ="C:\\Users\\sande\\Technocolabs_ML\\" + comp_name +'.csv'
    return pd.read_csv(rg)

df = dataset(comp_name)
st.markdown("")
st.subheader('Raw Data')
st.write(df)


df['Date'] = pd.to_datetime(df['Date'])
df['Open'] = df['Open'].str.replace('$','').astype(float)
df['High'] = df['High'].str.replace('$','').astype(float)
df['Close/Last'] = df['Close/Last'].str.replace('$','').astype(float)
df['Low'] = df['Low'].str.replace('$','').astype(float)
df['Day']=pd.to_datetime(df['Date'], format="%Y/%m/%d").dt.day
df['Month']=pd.to_datetime(df['Date'], format="%Y/%m/%d").dt.month
df['Year']=pd.to_datetime(df['Date'], format="%Y/%m/%d").dt.year
st.markdown("")
st.subheader('Pre-Processed Data')
st.write(df)
st.markdown("")
st.write(""" #### Graphs:""")
st.markdown("")
graphs(df)
st.markdown("")
mod_name = st.selectbox("Select Model", ("Linear Regression","LSTM","Arima"))
st.write(" ## Model : ", mod_name)

if mod_name == 'Linear Regression':
    LR(df)
elif mod_name == 'LSTM':
   LS(df)
else:
   AR(df)


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math
from numpy import array
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import yfinance as yf
from tensorflow.keras.models import load_model
import os


def create_dataset(dataset, time_step):
    dataX = []
    dataY = []
    for i in range(len(dataset) - time_step - 1):
        temp1 = dataset[i:(i + time_step), 0]
        temp2 = dataset[i + time_step, 0]
        dataX.append(temp1)
        dataY.append(temp2)

    return np.array(dataX), np.array(dataY)


def show_anal(form_val):
    ticker_names = {'CIPLA': 'CIPLA.NS', 'MARUTI': 'MARUTI.NS', 'TCS': 'TCS.NS', 'HEROMOTOCO': 'HEROMOTOCO.NS', 'RELIANCE': 'RELIANCE.NS',
                    'ITC': 'ITC.NS', 'ONGC': 'ONGC.NS', 'KOTAK': 'KOTAKBANK.NS', 'INDUSINDBK': 'INDUSINDBK.NS', 'HDFCLIFE': 'HDFCLIFE.NS',
                    'ULTRACEMCO': 'ULTRACEMCO.NS', 'MM': 'MM.NS', 'WIPRO': 'WIPRO.NS', 'NTPC': 'NTPC.NS', 'COALINDIA': 'COALINDIA.NS',
                    'BAJFINANCE': 'BAJFINANCE.NS', 'ICICIBANK': 'ICICIBANK.NS', 'BRITANNIA': 'BRITANNIA.NS', 'SHREECEM': 'SHREECEM.NS',
                    'LT': 'LT.NS', 'TECHM': 'TECHM.NS', 'GRASIM': 'GRASIM.NS', 'NESTLEIND': 'NESTLEIND.NS', 'BAJAJ-AUTO': 'BAJAJ-AUTO.NS',
                    'BHARTIARTL': 'BHARTIARTL.NS', 'GAIL': 'GAIL.NS', 'TITAN': 'TITAN.NS', 'HINDALCO': 'HINDALCO.NS', 'BAJAJFINSV': 'BAJAJFINSV.NS',
                    'TATASTEEL': 'TATASTEEL.NS'}
    df_tick = ticker_names.get(form_val)
    print(df_tick)
    df = yf.download(df_tick)
    df = df.drop('Adj Close', axis=1)
    df = df.fillna(method='ffill')
    close_data = df['Close'].astype('float64')
    plt.plot(close_data)
    plt.xlabel('Time')
    plt.ylabel('Price')
    save_results_to = 'H:/Projects/se project working/se project working/Stock-Market-Predictor/Webpage/maincode/static/'
    plt.savefig(save_results_to + 'anal.png', dpi=300)


def predict_share(form_val):
    ticker_names = {'CIPLA': 'CIPLA.NS', 'MARUTI': 'MARUTI.NS', 'TCS': 'TCS.NS', 'HEROMOTOCO': 'HEROMOTOCO.NS', 'RELIANCE': 'RELIANCE.NS',
                    'ITC': 'ITC.NS', 'ONGC': 'ONGC.NS', 'KOTAK': 'KOTAKBANK.NS', 'INDUSINDBK': 'INDUSINDBK.NS', 'HDFCLIFE': 'HDFCLIFE.NS',
                    'ULTRACEMCO': 'ULTRACEMCO.NS', 'MM': 'MM.NS', 'WIPRO': 'WIPRO.NS', 'NTPC': 'NTPC.NS', 'COALINDIA': 'COALINDIA.NS',
                    'BAJFINANCE': 'BAJFINANCE.NS', 'ICICIBANK': 'ICICIBANK.NS', 'BRITANNIA': 'BRITANNIA.NS', 'SHREECEM': 'SHREECEM.NS',
                    'LT': 'LT.NS', 'TECHM': 'TECHM.NS', 'GRASIM': 'GRASIM.NS', 'NESTLEIND': 'NESTLEIND.NS', 'BAJAJ-AUTO': 'BAJAJ-AUTO.NS',
                    'BHARTIARTL': 'BHARTIARTL.NS', 'GAIL': 'GAIL.NS', 'TITAN': 'TITAN.NS', 'HINDALCO': 'HINDALCO.NS', 'BAJAJFINSV': 'BAJAJFINSV.NS',
                    'TATASTEEL': 'TATASTEEL.NS'}
    model_names = {'CIPLA': 'CIPLA.h5', 'MARUTI': 'MARUTI.h5', 'TCS': 'TCS.h5', 'HEROMOTOCO': 'HEROMOTOCO.h5', 'RELIANCE': 'RELIANCE.h5',
                   'ITC': 'ITC.h5', 'ONGC': 'ONGC.h5', 'KOTAK': 'KOTAKBANK.h5', 'INDUSINDBK': 'INDUSINDBK.h5', 'HDFCLIFE': 'HDFCLIFE.h5',
                   'ULTRACEMCO': 'ULTRACEMCO.h5', 'MM': 'MM.h5', 'WIPRO': 'WIPRO.h5', 'NTPC': 'NTPC.h5', 'COALINDIA': 'COALINDIA.h5',
                   'BAJFINANCE': 'BAJAJFINANCE.h5', 'ICICIBANK': 'ICICIBANK.h5', 'BRITANNIA': 'BRITANNIA.h5', 'SHREECEM': 'SHREECEM.h5',
                   'LT': 'LT.h5', 'TECHM': 'TECHM.h5', 'GRASIM': 'GRASIM.h5', 'NESTLEIND': 'NESTLEIND.h5', 'BAJAJ-AUTO': 'BAJAJ-AUTO.h5',
                   'BHARTIARTL': 'BHARTIARTL.h5', 'GAIL': 'GAIL.h5', 'TITAN': 'TITAN.h5', 'HINDALCO': 'HINDALCO.h5', 'BAJAJFINSV': 'BAJAJFINSV.h5',
                   'TATASTEEL': 'TATASTEEL.h5'}
    df_tick = ticker_names.get(form_val)
    model_name = model_names.get(form_val)
    print(df_tick, model_name)
    df = yf.download(df_tick)
    # print(df)
    model = load_model(model_name)
    df = df.fillna(method='ffill')
    close_data = df['Close'].astype('float64')
    df = df.drop('Adj Close', axis=1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_data = scaler.fit_transform(np.array(close_data).reshape(-1, 1))

    train_size = int(len(close_data)*0.80)
    test_size = len(close_data) - train_size
    train_data = close_data[:train_size]
    test_data = close_data[train_size:]

    time_step = 500
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    abc = len(test_data)-500
    x_input = test_data[abc:].reshape(1, -1)

    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    lst_output = []
    n_steps = 500
    i = 0
    while(i < 30):

        if(len(temp_input) > 500):
            x_input = np.array(temp_input[1:])
            #print("{} day input {}".format(i, x_input))
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            #print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            i = i + 1

        else:
            x_input = x_input.reshape(1, n_steps, 1)
            yhat = model.predict(x_input, verbose=0)
            # print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i = i + 1
    day_new = np.arange(1, 501)
    day_pred = np.arange(501, 531)
    abc2 = len(close_data)-500
    plt.plot(day_new, scaler.inverse_transform(close_data[abc2:]))
    plt.plot(day_pred, scaler.inverse_transform(lst_output))

    df = close_data.tolist()
    df.extend(lst_output)
    plt.plot(df[len(close_data)-100:])

    plt.ylabel('Price')
    # plt.show()

    # plt.show()
    save_results_to = 'H:/Projects/se project working/se project working/Stock-Market-Predictor/Webpage/maincode/static/'
    plt.savefig(save_results_to + 'test.png', dpi=300)

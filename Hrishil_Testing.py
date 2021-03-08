import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

df = pd.read_csv("stockdata.csv")
df = df[2:]
close_data = df['Close.6'].astype('float64')

plt.plot(close_data)

scaler = MinMaxScaler(feature_range=(0, 1))
close_data = scaler.fit_transform(np.array(close_data).reshape(-1, 1))

train_size = int(len(close_data)*0.80)
test_size = len(close_data) - train_size
train_data = close_data[:train_size]
test_data = close_data[train_size:]


def create_dataset(dataset, time_step):
    dataX = []
    dataY = []
    for i in range(len(dataset) - time_step - 1):
        temp1 = dataset[i:(i + time_step), 0]
        temp2 = dataset[i + time_step, 0]
        dataX.append(temp1)
        dataY.append(temp2)

    return np.array(dataX), np.array(dataY)


time_step = 500
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[0], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)




# print(close_data)

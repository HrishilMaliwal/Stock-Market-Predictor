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
def preprocess(df):
    df.fillna(method='ffill', inplace=True)
    return df

df = pd.read_csv("df1.csv")
df=preprocess(df)
close_data = df['Close'].astype('float64')
df=df.drop('Adj Close',axis=1)

plt.plot(close_data)
plt.show()

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

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64,verbose=1)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# math.sqrt(mean_squared_error(y_train,train_predict))
# math.sqrt(mean_squared_error(y_test,test_predict))

look_back = 500
trainPredictPlot = np.empty_like(close_data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

testPredictPlot = np.empty_like(close_data)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict

plt.plot(scaler.inverse_transform(close_data))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

x_input = test_data['''XYZ''':].reshape(1, -1)

temp_input = list(x_input)
temp_input = temp_input[0].tolist()

lst_output = []
n_steps = 500
i = 0
while(i < 30):

    if(len(temp_input) > 100):
        x_input = np.array(temp_input[1:])
        print("{} day input {}".format(i, x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        lst_output.extend(yhat.tolist())
        i = i + 1
    
    else:
        x_input = x_input.reshape(1, n_steps, 1)
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i = i + 1

print(lst_output)


day_new = np.arange(1, 101)
day_pred = np.arange(101, 131)

plt.plot(day_new, scaler.inverse_transform(close_data['''XYZ''':]))
plt.plot(day_pred, scaler.inverse_transform(lst_output))

df = close_data.tolist()
df.extend(lst_output)
plt.plot(df['''XYZ''':])

df = scaler.inverse_transform(df).tolist()

plt.plot(df)



# print(close_data)

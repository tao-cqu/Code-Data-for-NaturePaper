# coding=gbk
import pandas as pd
import joblib
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense,GRU,Reshape
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

xl = pd.ExcelFile()
scaler_x=joblib.load()
scaler_y = joblib.load()
model=joblib.load()
writer_lstm = pd.ExcelWriter()
writer_gru = pd.ExcelWriter()
ls=pd.read_excel()
count=0
for a in ls.iloc[:,0]:
    count+=1
    cell_values = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        cell_value = df.iloc[a-1,4:9].values
        cell_values.append(cell_value)

    ts_data = pd.DataFrame(cell_values)
    n = len(ts_data)
    ts_index = pd.date_range(start='1984-12-31', periods=n, freq='YE-DEC')
    ts = pd.DataFrame(ts_data.values, index=ts_index)
    X = []
    y = []
    window_size = 5
    pred_length = window_size

    for i in range(len(ts_data) - window_size - pred_length + 1):
        X.append(ts_data[i:i + window_size])
        y.append(ts_data[i + window_size:i + window_size + pred_length].values.flatten())

    # Convert X and y to numpy array
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))
    y = y.reshape((y.shape[0], pred_length, ts_data.shape[1]))

    X_train = X
    y_train = y

    # Reshape the data to 2D
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    y_train_2d = y_train.reshape(-1, y_train.shape[-1])
    # Initialize a new StandardScaler instance
    scaler = StandardScaler().fit(X_train_2d)

    # Fit the scaler to the training data and transform it
    X_train_2d = scaler.transform(X_train_2d)
    y_train_2d = scaler.transform(y_train_2d)
    # Reshape the data back to 3D
    X_train = X_train_2d.reshape(X_train.shape)
    y_train = y_train_2d.reshape(y_train.shape)
    # 预测集
    y_test = y_train[-1].reshape(1, pred_length, -1)

    optimizer_1 = Adam(clipnorm=1.0)


    # lstm
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, activation='relu', input_shape=(window_size, ts_data.shape[1])))
    model_lstm.add(Dense(pred_length * ts_data.shape[1]))
    model_lstm.add(Reshape((pred_length, ts_data.shape[1])))
    model_lstm.compile(optimizer=optimizer_1, loss='mse')


    model_lstm.fit(X_train, y_train, epochs=200, verbose=0)

    yhat_lstm = model_lstm.predict(y_test)
    yhat_lstm = np.array(yhat_lstm)

    for i in range(19):
        new_input = yhat_lstm[-1:,:, :]
        new_input = new_input.reshape((1, pred_length, ts_data.shape[1]))
        new_output = model_lstm.predict(new_input)
        yhat_lstm = np.concatenate((yhat_lstm, new_output))

    yhat_2d_lstm = yhat_lstm.reshape(-1, yhat_lstm.shape[-1])
    yhat_2d_lstm = scaler.inverse_transform(yhat_2d_lstm)
    result_lstm = pd.DataFrame(yhat_2d_lstm)
    result_lstm.iloc[:, 0] = result_lstm.iloc[:, 0]
    # gru
    model_gru = Sequential()
    model_gru.add(GRU(50, activation='relu', input_shape=(window_size, ts_data.shape[1])))
    model_gru.add(Dense(pred_length * ts_data.shape[1]))
    model_gru.add(Reshape((pred_length, ts_data.shape[1])))

    optimizer_2 = Adam(clipnorm=1.0)

    model_gru.compile(optimizer=optimizer_2, loss='mse',)


    model_gru.fit(X_train, y_train, epochs=200, verbose=0)

    yhat_gru = model_gru.predict(y_test)
    yhat_gru = np.array(yhat_gru)

    for i in range(19):
        new_input = yhat_gru[-1,:, :]
        new_input = new_input.reshape((1, pred_length, ts_data.shape[1]))
        new_output = model_gru.predict(new_input)
        yhat_gru = np.concatenate((yhat_gru, new_output))

    yhat_2d_gru = yhat_gru.reshape(-1, yhat_gru.shape[-1])
    yhat_2d_gru = scaler.inverse_transform(yhat_2d_gru)
    result_gru = pd.DataFrame(yhat_2d_gru)
    result_gru.iloc[:, 0] = result_gru.iloc[:, 0]

    X_scaled_gru = result_gru
    X_scaled_lstm = result_lstm

    output_gru = model.predict(X_scaled_gru)
    output_gru = pd.DataFrame(scaler_y.inverse_transform(output_gru))
    output_lstm = model.predict(X_scaled_lstm)
    output_lstm = pd.DataFrame(scaler_y.inverse_transform(output_lstm))


    result_gru.columns = ['Flux','Population','GDP','Precipitation','Temperature']
    result_lstm.columns = ['Flux','Population','GDP','Precipitation','Temperature']

    result_gru['M_weight'] = output_gru
    result_gru['MPs_Flux(tons)'] = result_gru['M_weight'] * result_gru['Flux'] * 10 ** (-18)
    result_lstm['M_weight'] = output_lstm
    result_lstm['MPs_Flux(tons)'] = result_lstm['M_weight'] * result_lstm['Flux'] * 10 ** (-18)

    sheet_name = ls.iloc[count - 1, 1].replace('/', '_')

writer_gru.close()
writer_lstm.close()
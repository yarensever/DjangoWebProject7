from datetime import datetime
from django.shortcuts import render
from django.http import HttpRequest
from math import sqrt
from datetime import datetime
from numpy import concatenate
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, Bidirectional, GRU
from keras.layers.recurrent import LSTM
from sklearn.utils import shuffle
import requests
import json
scaler  = MinMaxScaler(feature_range=(0, 1))

#first of all, create one year data
def EPIAS_API():
    down = './test.json'
    url  = 'https://seffaflik.epias.com.tr/transparency/service/market/day-ahead-mcp?endDate=2020-12-28&startDate=2019-12-30'
    outpath=down
    generatedURL=url
    response = requests.get(generatedURL)
    if response.status_code == 200:
        with open(outpath, "wb") as out:
            for chunk in response.iter_content(chunk_size=128):
                out.write(chunk)
    with open(down) as json_file:
        data = json.load(json_file)
    body=data.get('body')
    gen=body.get('dayAheadMCPList')
    df=pd.DataFrame(gen)
    return(df)

def split_dataset(df,look_back):
    df             = df[0:(len(df)-look_back)]   #ilk 26256 saat = 1094 gün
    #normalize data
    values  = df.values.reshape(-1,1)
    values  = values.astype('float32')
    dataset = scaler.fit_transform(values)
    print("Dataset length is:", len(dataset))
     
    #%80 Train %20 Test
    split_by    = 0.79999
    train_size  = int(len(dataset) * split_by)
    test_size   = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    return train, test, dataset

def create_dataset(dataset, look_back):
        data_X, data_Y = [], []
        for i in range(len(dataset) - look_back -1):
            a = dataset[i:(i + look_back), 0]
            data_X.append(a)
            data_Y.append(dataset[i + look_back, 0])
        return(np.array(data_X), np.array(data_Y))

def creating_train_test(train,test,look_back):
    train_X, train_Y = create_dataset(train, look_back)
    test_X, test_Y   = create_dataset(test,  look_back)

    train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
    test_X  = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
    return train_X, train_Y, test_X, test_Y

#LSTM MODEL
def fitting_model(train_X, train_Y,test_X, test_Y, look_back):
        model = Sequential()
        model.add(LSTM(5, input_shape = (1, look_back)))
        model.add(Dense(1))
        model.compile(loss = 'mae', optimizer = 'adam')
        hist= model.fit(train_X, train_Y, epochs = 10, batch_size = 30, verbose = 0, validation_data=(test_X, test_Y), shuffle=False)
        return(model)

def prediction(model, x,y):
    #inverse transform
    predicted = scaler.inverse_transform(model.predict(x))
    actual     = scaler.inverse_transform([y])
    rmse = math.sqrt(mean_squared_error(actual[0], predicted[:, 0]))
    return(rmse, predicted)

def results(model, train_X, train_Y, test_X, test_Y):
    rmse_train, train_predict = prediction(model, train_X, train_Y)
    rmse_test, test_predict   = prediction(model, test_X, test_Y)
    #print("Training data score: %.2f RMSE" % rmse_train)
    #print("Test data score: %.2f RMSE" % rmse_test)
    return train_predict, test_predict

def learning(dataset, train_predict, test_predict, look_back):
    train_predict_plot = np.empty_like(dataset)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict

    test_predict_plot = np.empty_like(dataset)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict) + (look_back * 2) + 1:len(dataset)-1 , :] = test_predict
    result = test_predict_plot[(len(dataset)-look_back)-1:len(dataset)-1].flatten()
    return result


def output(df, look_back):
    train, test, dataset = split_dataset(df,look_back)
    train_X, train_Y, test_X, test_Y = creating_train_test(train,test,look_back)
    #Fit the first model.
    model = fitting_model(train_X, train_Y, test_X, test_Y,look_back)
    train_predict, test_predict = results(model, train_X, train_Y, test_X, test_Y)
    result = learning(dataset,train_predict,test_predict,look_back)
    return result

def home(request):
    df = EPIAS_API()
    df_eur = df['priceEur']
    df_usd = df['priceUsd']
    df_tl  = df['price']

    look_back = 24
    mcp_eur = output(df_eur, look_back).tolist()
    mcp_tl = output(df_tl, look_back).tolist()
    mcp_usd = output(df_usd, look_back).tolist()
    Saat = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

    return render(
        request,
        'app/index.html',
        {
            'mcp_eur':mcp_eur,
            'mcp_tl':mcp_tl,
            'mcp_usd':mcp_usd,
            'Saat': Saat
        }
    )


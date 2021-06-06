from datetime import datetime
from django.shortcuts import render
from django.http import HttpRequest
from math import sqrt
from numpy import concatenate
import numpy as np
import pandas as pd
import math
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import LabelEncoder
#from sklearn.metrics import mean_squared_error
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation
#from keras.layers import LSTM, Bidirectional, GRU
#from keras.layers.recurrent import LSTM
#from sklearn.utils import shuffle
#import requests
#import json
from datetime import datetime, timedelta,date
#from evds import evdsAPI
#import psycopg2
#from .models import FileModel
#scaler  = MinMaxScaler(feature_range=(0, 1))
#connection = psycopg2.connect(user="developer@ys-host.postgres.database.azure.com",
#                                  password="Seniorproject123",
#                                  host="ys-host.postgres.database.azure.com",
#                                  port="5432",
#                                  database="postgres")

##writes data to the table
#def write(connection,f):
#        with connection:
#                cur = connection.cursor()
#                next(f)
#                cur.copy_from(f, "public.day_ahead_market" , sep=',')
#                connection.commit()  


##API function gets today's data and returns df
#def EPIAS_API(today):
#    down = './test.json'
#    url  = 'https://seffaflik.epias.com.tr/transparency/service/market/day-ahead-mcp?endDate='+ str(today) +'&startDate=' + str(today) + ''
#    outpath=down
#    generatedURL=url
#    response = requests.get(generatedURL)
#    if response.status_code == 200:
#        with open(outpath, "wb") as out:
#            for chunk in response.iter_content(chunk_size=128):
#                out.write(chunk)
#    with open(down) as json_file:
#        data = json.load(json_file)
#    body=data.get('body')
#    gen=body.get('dayAheadMCPList')
#    df=pd.DataFrame(gen)
#    df.to_csv("daily", index=False)
#    f = open('daily', 'r')
#    return f


#def GET_CURR(date):
#    evds = evdsAPI('0BvAGmYlzp')
#    df = evds.get_data(['TP.DK.EUR.S.YTL', 'TP.DK.EUR.C'], startdate=date, enddate=date)
#    euro_tl=float(df['TP_DK_EUR_S_YTL'].values)
#    euro_usd=float(df['TP_DK_EUR_C'].values)
#    return euro_tl, euro_usd


#def getData(size):
#    down = './test.json'
#    url  = 'https://seffaflik.epias.com.tr/transparency/service/market/day-ahead-mcp?endDate=2020-12-28&startDate=2019-12-30'
#    outpath=down
#    generatedURL=url
#    response = requests.get(generatedURL)
#    if response.status_code == 200:
#        with open(outpath, "wb") as out:
#            for chunk in response.iter_content(chunk_size=128):
#                out.write(chunk)
#    with open(down) as json_file:
#        data = json.load(json_file)
#    body=data.get('body')
#    gen=body.get('dayAheadMCPList')
#    df=pd.DataFrame(gen)
#    return(df['priceEur'])


#def split_dataset(df,look_back):
#    df             = df[0:(len(df)-look_back)]   #ilk 26256 saat = 1094 gün
#    df = pd.DataFrame(df)
#    #normalize data
#    values  = df.values.reshape(-1,1)
#    values  = values.astype('float32')
#    dataset = scaler.fit_transform(values)
#    print("Dataset length is:", len(dataset))
     
#    #%80 Train %20 Test
#    split_by    = 0.79999
#    train_size  = int(len(dataset) * split_by)
#    test_size   = len(dataset) - train_size
#    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
#    return train,test,dataset

#def create_dataset(dataset, look_back):
#        data_X, data_Y = [], []
#        for i in range(len(dataset) - look_back -1):
#            a = dataset[i:(i + look_back), 0]
#            data_X.append(a)
#            data_Y.append(dataset[i + look_back, 0])
#        return(np.array(data_X), np.array(data_Y))

#def creating_train_test(train,test,look_back):
#    train_X, train_Y = create_dataset(train, look_back)
#    test_X, test_Y   = create_dataset(test,  look_back)

#    train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
#    test_X  = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
#    return train_X, train_Y, test_X, test_Y

##LSTM MODEL
#def fitting_model(train_X, train_Y,test_X, test_Y, look_back):
#        model = Sequential()
#        model.add(LSTM(5, input_shape = (1, look_back)))
#        model.add(Dense(1))
#        model.compile(loss = 'mae', optimizer = 'adam')
#        hist= model.fit(train_X, train_Y, epochs = 10, batch_size = 30, verbose = 0, validation_data=(test_X, test_Y), shuffle=False)
#        return(model)

#def prediction(model, x,y):
#    #inverse transform
#    predicted = scaler.inverse_transform(model.predict(x))
#    actual     = scaler.inverse_transform([y])
#    rmse = math.sqrt(mean_squared_error(actual[0], predicted[:, 0]))
#    return(rmse, predicted)

#def results(model, train_X, train_Y, test_X, test_Y):
#    rmse_train, train_predict = prediction(model, train_X, train_Y)
#    rmse_test, test_predict   = prediction(model, test_X, test_Y)
#    print("Training data score: %.2f RMSE" % rmse_train)
#    print("Test data score: %.2f RMSE" % rmse_test)
#    return train_predict, test_predict

#def learning(dataset, train_predict, test_predict, look_back):
#    test_predict_plot = np.empty_like(dataset)
#    test_predict_plot[:, :] = np.nan
#    test_predict_plot[len(train_predict) + (look_back * 2) + 1:len(dataset)-1 , :] = test_predict
#    result = test_predict_plot[(len(dataset)-look_back)-1:len(dataset)-1].flatten()
#    return result

#def output(df, look_back):
#    train, test, dataset = split_dataset(df,look_back)
#    train_X, train_Y, test_X, test_Y = creating_train_test(train,test,look_back)
#    #Fit the first model.
#    model = fitting_model(train_X, train_Y, test_X, test_Y,look_back)
#    train_predict, test_predict = results(model, train_X, train_Y, test_X, test_Y)
#    result = learning(dataset,train_predict,test_predict,look_back)
#    return result

#def ar(df):
#    data = pd.DataFrame(df).values
    
#    data_arr = np.reshape(data, (24,24)) #creates 24x24 matrix

#    #creates 24x4 matrix only includes PTF at t-24, t-48, t-168, t-336 
#    y_all= [];
#    for i in range(24):
#        y_all.append([data_arr[-1][i], data_arr[-2][i], data_arr[-7][i], data_arr[-14][i]])
#    y_all = np.reshape(y_all,(24,4))

#    #AR coefficients
#    beta = [[0.1861393 ],
#     [0.11570386],
#     [0.33828201],
#     [0.35176419]]

#    #forecasting
#    forecasted_result = np.matmul(y_all, beta).tolist()
#    return forecasted_result


def home(request):
    today = datetime.today() + timedelta(days=1)
    date = datetime.strftime(today,"%d-%m-%Y")
    #look_back = 24
    #euro_tl, euro_usd = GET_CURR(date)

    #lstm_eur = output(getData(8808-168), look_back).tolist()
    #lstm_tl  =[float(i)*euro_tl for i in lstm_eur]
    #lstm_usd =[float(i)*euro_usd for i in lstm_eur]

    #ar_eur = ar(df_eur_ar)
    #ar_eur = list(np.concatenate(ar_eur).flat)
    #ar_tl = [float(i)*euro_tl for i in ar_eur]
    #ar_usd = [float(i)*euro_usd for i in ar_eur]
    mr_eur = [34.19234899, 33.31135675, 32.04882816, 31.48935097, 31.98780587,31.16855209,31.41239895,34.29366887,31.83414352,29.06576025,33.0051231,32.53998862,32.2921479,32.70794282,34.21267163,31.92205053,33.27367704,34.13111617,
              33.90711492,34.88112931,35.5437873,36.84678749,36.99814832, 34.62661438]
   
    pr_eur = [33.22035595, 33.7456801 , 32.40736245 ,32.69828024, 32.49639888 ,32.07119787,
                28.36133762 ,29.77796826 ,33.9074097 , 33.70556755 ,34.84621175 ,36.4394152,
                33.76014699, 34.76491597, 35.67669415, 34.50172161 ,35.47626673 ,36.81794607,
                37.78520169, 40.55386256, 41.89982896, 41.82792678, 38.6702254 , 35.92591061]

    mult_lstm_eur = [34.1155 ,  33.886517 ,33.777683 ,33.23288 , 32.869976 ,32.712982, 32.15533,
                     32.00008  ,32.559105, 34.54532 , 35.492363, 35.710125, 35.649544 ,34.47031,
                     34.58501,  35.310616, 35.090435 ,35.256374, 35.19935 , 34.5394 ,  34.559464,
                     35.62798 , 35.445393, 34.725155] 
     
    lstm_eur = [34.58519, 34.090454 ,33.221344, 33.153435, 32.699093, 32.424393 ,32.144745,
                 32.73788 , 31.283497, 31.2897 ,  32.715984 ,33.11534 , 33.55743 , 33.486015,
                 33.46278,  33.689835 ,33.75273 , 33.5936 ,  34.09149 , 34.18594 , 34.27697,
                 34.796738, 36.162926, 35.87821 ]
    
    Hour = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

    return render(
        request,
        'app/index.html',
        {
            'mr_eur': mr_eur,
            'pr_eur': pr_eur,
            'mult_lstm_eur': mult_lstm_eur,
            'lstm_eur': lstm_eur,
            'Hour': Hour,
            'date': date
        }
    )


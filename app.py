from pandas import DataFrame
from pandas import Series
from pandas import concat
import pandas as pd
import json
import re

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy
from flask import Flask, render_template, url_for, jsonify
# แปลงอนุกรมเวลาให้เปHนSupervised Learning 

def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	#print(df)
	return df


# # การทำ  Differencing  หรือ การทำข้อมูลให้เป้นค่าคงที่
def difference(dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
                value = dataset[i] - dataset[i - interval]
                diff.append(value)
        return Series(diff)

#  invese ค่าที่ Differencing 
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

#  normaliz ข้อมูล ให้อยุ่ในช่วง 1 -1
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# inverse ค่าสำหรับการทำนาย
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# ใช้เทนนิค LSTM กับค่า Test
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=1, verbose=0, shuffle=False)
		model.reset_states()
	return model

#one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=1)
	#print(yhat)
	return yhat[0,0]

# โหลดข้อมูล
series = pd.read_csv('PRSA_data_2010.1.1-2014.12.31.csv')
series=series.dropna(axis=0)

# เปลี่ยนข้อมูลให้เป้นค่าคงที่
raw_values = series.values
diff_values = difference(raw_values, 1)

#  เปลี่ยนข้อมูลให้เป้น supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

#  เเบ่งข้อมูล train test
train, test = supervised_values[0:-168], supervised_values[-168:]
train1, test1 = supervised_values[0:-120], supervised_values[-120:]
# เเปลงขนาดข้อมูล
scaler, train_scaled, test_scaled = scale(train, test)

# กำหนดค่าให้โมเดล
lstm_model = fit_lstm(train_scaled, 1, 50, 1)

train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

# walk-forward validation 

predictions = list()


day =0
mymax =0
mymaxs =0
#testx = json.dumps(test)
sev = []
app = Flask(__name__)
for i in range(len(test_scaled)):
	# make one-step forecast
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        yhat = forecast_lstm(lstm_model, 1, X)
	# invert scaling
        yhat = invert_scale(scaler, X, yhat)
	# invert differencing
        yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
	# store forecast
        predictions.append(yhat)
        raw = raw_values[len(train) + i + 1]
       # print('hour=%d, Predicted=%f, Expected=%f' % (i+1, yhat, raw))
        sev.append(yhat)
        mymax=yhat
        mymaxs=(max(mymax))
        day = day+yhat
        #print("averaget",sev)
               
        
        

        
#print("averaget",sev[2])
xx ='my'
per = day/24
test =int(sev[23])
test1=int(sev[47])
test2=int(sev[71])
test3=int(sev[95])
test4=int(sev[119])
test5=int(sev[143])
test6=int(sev[167])
mymaxs =int(mymax)


myj=test
myj1 =test1
myj2 =test2
myj3 =test3
myj4 =test4
myj5 =test5
myj6 =test6
mys =json.dumps(myj)
mys1 =json.dumps(myj1)
mys2 =json.dumps(myj2)
mys3 =json.dumps(myj3)
mys4 =json.dumps(myj4)
mys5 =json.dumps(myj5)
mys6 =json.dumps(myj6)
        
        

data = {
                "one" : mys,
                 "two" : mys1,
                 "three" : mys2,
                 "four" : mys3,
                 "five" : mys4,
                 "six" : mys5,
                 "seven" : mys6
                
        }

data1 = [
        
           
            mys1
            
        
        
    ]
# report performance
rmse = sqrt(mean_squared_error(raw_values[-168:], predictions))
#print(' RMSE: %.3f' % rmse)
# line plot of observed vs predicted










@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api", methods=['GET'])
def get_api():
    return jsonify(data)  # Return web frameworks information



if __name__ == "__main__":
    app.run(host="0.0.0.0")


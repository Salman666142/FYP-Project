import os
import sys
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import plotly.graph_objs as go
from plotly.offline import plot
import time
import json
import plotly
from chart_studio import plotly as py
from sklearn.metrics import mean_squared_error


def train(df):
    #     print(df)

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_axis(df['Date'], inplace=True)
    df.dropna(inplace=True)

    data = df['Value'].values
    data = data.reshape((-1, 1))

    split_percent = 0.80
    split = int(split_percent * len(data))

    train = data[:split]
    test = data[split:]

    print(len(train))
    print(len(test))

    date_train = df['Date'][:split]
    date_test = df['Date'][split:]

    look_back = 45

    # train_generator = TimeseriesGenerator(train, train, length=look_back, batch_size=20)
    # test_generator = TimeseriesGenerator(test, test, length=look_back, batch_size=1)

    train_generator_close = TimeseriesGenerator(train, train, length=look_back, batch_size=20)
    test_generator_close = TimeseriesGenerator(test, test, length=look_back, batch_size=1)

    model = Sequential()
    model.add(
        LSTM(15,
             activation='relu',
             input_shape=(look_back, 1))
    )
    model.add(Dropout(0.3))
    model.add(Dense(1))

    opt = keras.optimizers.RMSprop(learning_rate=0.01)
    model.compile(
        optimizer=opt,
        loss='msle'
    )

    #     checkpointer = ModelCheckpoint(verbose=1, save_best_only=True)
    #     callbacks = [
    #         tf.keras.callbacks.EarlyStopping(patience=3, monitor='loss'),
    #         tf.keras.callbacks.TensorBoard(log_dir='logs')]

    model.fit_generator(train_generator_close, epochs=2, verbose=1, validation_data=test_generator_close)
    prediction = model.predict_generator(test_generator_close)
    train = train.reshape((-1))
    test = test.reshape((-1))
    prediction = prediction.reshape((-1))
    trace1 = go.Scatter(
        x=date_train,
        y=train,
        mode='lines',
        name='Data'
    )
    trace2 = go.Scatter(
        x=date_test,
        y=prediction,
        mode='lines',
        name='Prediction'
    )
    trace3 = go.Scatter(
        x=date_test,
        y=test,
        mode='lines',
        name='Ground Truth'
    )

    data = [trace1, trace2, trace3]
    return model, data


lst = []
graphs = []


def predict(df, loaded_model):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_axis(df['Date'], inplace=True)
    df.dropna(inplace=True)

    data = df['Value'].values
    data = data.reshape((-1, 1))

    split_percent = 0.80
    split = int(split_percent * len(data))

    train = data[:split]
    test = data[split:]

    #     print(len(train))
    #     print(len(test))

    date_train = df['Date'][:split]
    date_test = df['Date'][split:]

    look_back = 30
    #     loaded_model = load_model(name + '.h5')
    data = data.reshape((-1))

    def predict(num_prediction, model):
        prediction_list = data[-look_back:]

        for _ in range(num_prediction):
            y = prediction_list[-look_back:]
            y = y.reshape((1, look_back, 1))
            inn = loaded_model.predict(y)[0][0]
            prediction_list = np.append(prediction_list, inn)
        prediction_list = prediction_list[look_back - 1:]

        return prediction_list

    num_prediction = 1
    forecast_close = predict(num_prediction, loaded_model)

    def predict_dates(num_prediction):
        last_date = df['Date'].values[-1]
        prediction_dates = pd.date_range(last_date, periods=num_prediction + 1).tolist()
        return prediction_dates

    forecast_dates = predict_dates(num_prediction)

    trace1 = go.Scatter(
        x=forecast_dates,
        y=forecast_close,
        mode='lines',
        name='Value'
    )

    data = [trace1]
    graphs.append(data)
    # fig = go.Figure(data=[trace1], layout=layout)
    # fig.show()
    # lst.clear()
    for i in forecast_close[1:]:
        print(i)
        lst.append(i)


def suggestion(df, pred_list):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_axis(df['Date'], inplace=True)
    df.dropna(inplace=True)

    data = df['Value']
    #     data = data.reshape((-1, 1))
    counter = 0
    for values in pred_list:
        print(values, '------------------')
        if values > data[-1]:
            counter += 1
    chances = (counter / len(pred_list)) * 100
    # print("Your chances for success are", chances, "%")

    if chances < 30.0:
        return_string = f"Your chances are {chances}%, which is considered low. Its strongly recommended not to " \
                        f"invest on it at this point of time. The last value of stock is {data[-1]}. Your " \
                        f"predicted values are the predicted values : {pred_list} "

    elif 30.0 <= chances < 50.0:
        return_string = f"Your chances are {chances}%, which is considered slightly low. I suggest you not to " \
                        f"invest at this point of time. The last value of stock is {data[-1]}. Your predicted " \
                        f"values are the predicted values : {pred_list} "

    elif 50.0 <= chances < 75.0:
        return_string = f"Your chances are {chances}%! Good amount of chances are there. You might want to " \
                        f"give it a try. The last value of stock is {data[-1]}. Your predicted values are the " \
                        f"predicted values : {pred_list} "

    else:
        return_string = f"Your chances are {chances}%! Very high chance is there. Suggesting you not to miss " \
                        f"this chance to win big!. The last value of stock is {data[-1]}. Your predicted " \
                        f"values are the predicted values : {pred_list} "
    lst.clear()
    return return_string


# % tb
from flask import Flask, make_response, request, render_template
import io
from io import StringIO
import csv
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Dropout, GRU
import tensorflow
import os
import random

app = Flask(__name__)

my_pic = os.path.join('static', 'pics')
app.config['UPLOAD_FOLDER'] = my_pic
ALLOWED_EXTENSION = {'csv'}

current_user = []

hot_stocks = ["Apple", "Nike", "Intel", "Visa"]
rising_stocks = ["Travelers", "Cisco", "3M", "Dow"]
dropping_stocks = ["Walmart", "Johnson & Johnson", "AMGEN", "United Health"]
tip = ["Investing should be a luxury not a necessity. A golden rule for investment is the money that you want to "
       "invest, you should be prepared to lose it in worst case. It shouldn't be a money that you would really need it "
       "for other necessitates in life. Do manage you money wisely and plan further ahead before investing. Good "
       "Luck! ", "Do ensure u pick the right stock,short term or long term. Remember: Buying a share of a company’s "
                 "stock makes you a part owner of that business. Profits and losses will be shared. So choose your "
                 "stock wisely. Good Luck!"]
chosen_number = random.randint(0, 1)
chosen_tip = tip[chosen_number]

main_page_colour = "yellow"
error_page_colour = "red"
prediction_page_colour = "aqua"


def allowed_file(filename):
    print("..............")
    print(filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION)


def pred(df, a):
    # convert an array of values into a dataset matrix
    for i in range(int(a)):
        #         print('Prediction number: ' + str(i + 1))
        model, trainData = train(df)
        predict(df, model)
    #         print("Current prediction:", lst[-1])

    #     print(lst)
    return suggestion(df, lst), graphs, trainData


@app.route('/')
def form():
    stock_market = os.path.join(app.config['UPLOAD_FOLDER'], 'stock market.jpg')
    return render_template('main.html', user_image=stock_market, hot_stocks=hot_stocks, rising_stocks=rising_stocks,
                           dropping_stocks=dropping_stocks, tip=chosen_tip, color=main_page_colour)


@app.route('/button', methods=["GET", "POST"])
def button():
    if request.method == "POST":
        stock_market = os.path.join(app.config['UPLOAD_FOLDER'], 'stock market.jpg')

        return render_template("wait.html", wait='Training', user_image=stock_market, color=main_page_colour)


#     return render_template("wait.html", ButtonPressed = ButtonPressed)


@app.route('/txt', methods=["POST", "GET"])
def txt():
    ff = request.form['feed']
    print(current_user)
    if len(current_user) > 0:
        n = current_user[-1]
    else:
        n = "USER"
    print("CUrrent_USER----->", n)
    # n = str(n)
    print(ff)
    f = open("Feedback.txt", "a")
    f.write(f"Name: {n}, Message: {ff}" + '\n')
    f.close()
    stock_market = os.path.join(app.config['UPLOAD_FOLDER'], 'stock market.jpg')

    return render_template('main.html', user_image=stock_market, hot_stocks=hot_stocks, rising_stocks=rising_stocks,
                           dropping_stocks=dropping_stocks, tip=chosen_tip, color=main_page_colour)


@app.route('/transform', methods=["POST"])
def transform_view():
    missing = os.path.join(app.config['UPLOAD_FOLDER'], 'missing.jpg')
    low_value = os.path.join(app.config['UPLOAD_FOLDER'], 'low value.jpg')
    no_value = os.path.join(app.config['UPLOAD_FOLDER'], 'no value.jpg')
    no_name = os.path.join(app.config['UPLOAD_FOLDER'], 'no name.jpg')
    error = os.path.join(app.config['UPLOAD_FOLDER'], 'error.jpg')
    prediction = os.path.join(app.config['UPLOAD_FOLDER'], 'prediction.jpg')

    try:
        f = request.files['data_file']
        a = request.form['loop']
        n = request.form['name']
        # print(f)
        current_user.append(n)
        if a.strip() != '' and a.strip() is not None:
            a = int(a)
        else:
            a = 0
        if not f:
            return render_template("wait.html", wait='No files inputted ', name=n, user_image=missing, a=a,
                                   color=error_page_colour)
        elif int(a) == 0:
            return render_template("wait.html", wait="Please give an iteration value!! it must not be left blank!!",
                                   name=n, user_image=error, a=a, color=error_page_colour)
        elif int(a) < 0:
            return render_template("wait.html", wait=' Please give a valid iteration value which would be 1 or more!',
                                   name=n, user_image=low_value, a=a, color=error_page_colour)
        elif a is None:
            return render_template("wait.html", wait=' Please give an iteration value!!',
                                   name=n, user_image=no_value, a=a, color=error_page_colour)
        elif not n:
            return render_template("wait.html", wait=' No name given!', user_image=no_name, a=a,
                                   color=error_page_colour)
        else:
            # allowed_file(f)
            # current_user.append(n)
            file = request.files.get('data_file')
            print(file)
            #         print(file[0])
            df = pd.read_csv(file)

            lst, graphs, trainData = pred(df, a)

            df['Value'][-14:].tolist()
            graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

            # train graph
            trainDataJson = json.dumps(trainData, cls=plotly.utils.PlotlyJSONEncoder)

            return render_template("wait.html", wait=lst, name=n, user_image=prediction, graphJSON=graphJSON, a=a,
                                   trainDataJson=trainDataJson, current_user=current_user, color=prediction_page_colour)
    except UnicodeDecodeError and KeyError:
        exception_string = "Invalid file format!! please give a valid file format!!"
        return render_template("wait.html", wait=exception_string, name=' ', user_image=error, a=a
                               , color=error_page_colour)
    except TypeError:
        exception_string = "Please give an iteration value!! it must not be left blank!!"
        return render_template("wait.html", wait=exception_string, name=' ', user_image=error, a=a
                               , color=error_page_colour)


if __name__ == "__main__":
    app.run(debug=True, port=5000)

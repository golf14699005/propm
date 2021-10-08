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


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api", methods=['GET'])
def get_api():
    return jsonify(data)  # Return web frameworks information



if __name__ == "__main__":
    app.run(host="0.0.0.0")


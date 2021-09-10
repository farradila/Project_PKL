from flask import Flask, render_template, request, make_response, redirect, session, url_for
import sqlite3
from datetime import datetime
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import os
import time
import pandas as pd
import numpy as np

from werkzeug.utils import secure_filename
from flask_uploads import UploadSet,configure_uploads,IMAGES,DATA,ALL
import re

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.secret_key = 'yoursecretkey'

# Retrieve LAST data from database
def getLastData():
    conn = sqlite3.connect('sensorsData.db')
    curs = conn.cursor()
    for row in curs.execute("SELECT * FROM DHT_data ORDER BY timestamp DESC LIMIT 1"):
        time = str(row[0])
        temp = row[1]
        hum = row[2]
        co2 = row[3]
    # conn.close()
    return time, temp, hum, co2


# Get 'x' samples of historical data
def getHistData(numSamples):
    conn = sqlite3.connect('sensorsData.db')
    curs = conn.cursor()
    curs.execute("SELECT * FROM DHT_data ORDER BY timestamp DESC LIMIT " + str(numSamples))
    data = curs.fetchall()
    dates = []
    temps = []
    hums = []
    co2 =[]

    for row in reversed(data):
        dates.append(row[0])
        temps.append(row[1])
        hums.append(row[2])
        co2.append(row[3])
        temps, hums , co2 = testeData(temps, hums, co2)
    return dates, temps, hums, co2

def getHistDataSmoke(numSamples):
    conn = sqlite3.connect('sensorsData.db')
    curs = conn.cursor()
    curs.execute("SELECT * FROM smoke_data ORDER BY timestamp DESC LIMIT " + str(numSamples))
    data = curs.fetchall()
    dates = []
    co2 = []

    for row in reversed(data):
        dates.append(row[1])
        co2.append(row[2])
    return dates, co2

# Test data for cleanning possible "out of range" values
def testeData(temps, hums, co2):
    n = len(temps)
    for i in range(0, n - 1):
        if (temps[i] < -10 or temps[i] > 50):
            temps[i] = temps[i - 2]
        if (hums[i] < 0 or hums[i] > 100):
            hums[i] = temps[i - 2]
        if (co2[i] < 0 or co2[i] > 100):
            co2[i] = temps[i - 2]
    return temps, hums, co2


# Get Max number of rows (table size)
def maxRowsTable():
    conn = sqlite3.connect('sensorsData.db')
    curs = conn.cursor()
    for row in curs.execute("select COUNT(temp) from  DHT_data"):
        maxNumberRows = row[0]
    return maxNumberRows


# Get sample frequency in minutes
def freqSample():
    times, temps, hums, co2 = getHistData(2)
    fmt = '%Y-%m-%d %H:%M:%S'
    tstamp0 = datetime.strptime(times[0], fmt)
    tstamp1 = datetime.strptime(times[1], fmt)
    freq = tstamp1 - tstamp0
    freq = int(round(freq.total_seconds() / 60))
    return (freq)

# define and initialize global variables
global numSamples
numSamples = maxRowsTable()
if (numSamples > 101):
    numSamples = 100

global freqSamples
freqSamples = freqSample()

global rangeTime
rangeTime = 100

@app.route('/', methods=['GET', 'POST'])
def login():
    msg = ''
    if (request.method == "POST"):
        username = request.form["username"]
        password = request.form["password"]
        conn = sqlite3.connect("sensorsData.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = '" + username + "' and password = '" + password + "'")
        r = c.fetchall()
        for i in r:
            if (username == i[0] and password == i[1]):
                session["logedin"] = True
                session["username"] = username
                return redirect(url_for("home"))
            else:
                msg = 'please enter valid username and password'

    return render_template('login.html', msg=msg)

@app.route('/index')
def index():
    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the home page
        return render_template('index3.html', username=session['username'])
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = None
    if (request.method == "POST"):
        if (request.form["username"] != "" and request.form["password"] != ""):
            username = request.form["username"]
            password = request.form["password"]
            conn = sqlite3.connect("sensorsData.db")
            c = conn.cursor()
            c.execute("INSERT INTO users VALUES ('" + username + "', '" + password + "')")
            msg = "Your account is created"
            conn.commit()
            conn.close()
        else:
            msg = "Something wents wrong"

    return render_template('register.html', msg=msg)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route('/home')
def home():
    return render_template('index3.html')

@app.route("/monitoring")
def monitoring():
    time, temp, hum, co2 = getLastData()
    templateData = {
        'time': time,
        'temp': temp,
        'hum': hum,
        'co2': co2,
        'freq': freqSamples,
        'rangeTime': rangeTime
    }
    return render_template('monitoring.html', **templateData)

@app.route("/grafik")
def grafik():
    time, temp, hum, co2 = getLastData()
    templateData = {
        'time': time,
        'temp': temp,
        'hum': hum,
        'co2': co2,
        'freq': freqSamples,
        'rangeTime': rangeTime
    }
    return render_template('grafik.html', **templateData)

@app.route('/grafik', methods=['POST'])
def my_form_post():
    global numSamples
    global freqSamples
    global rangeTime
    rangeTime = int(request.form['rangeTime'])
    if (rangeTime < freqSamples):
        rangeTime = freqSamples + 1
    numSamples = rangeTime //freqSamples
    numMaxSamples = maxRowsTable()

    if (numSamples > numMaxSamples):
        numSamples = (numMaxSamples -1)
    time, temp, hum, co2 = getLastData()

    templateData = {
        'time'		: time,
        'temp'		: temp,
        'hum'		: hum,
        'freq'		: freqSamples,
        'rangeTime'	: rangeTime
    }
    return render_template('grafik.html', **templateData)

@app.route('/plot/temp')
def plot_temp():
    times, temps, hums, co2 = getHistData(numSamples)
    ys = temps
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title("Temperature [°C]")
    axis.set_xlabel("Samples")
    axis.grid(True)
    xs = range(numSamples)
    axis.plot(xs, ys)
    canvas = FigureCanvas(fig)
    output = io.BytesIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    return response

@app.route('/plot/hum')
def plot_hum():
    times, temps, hums, co2 = getHistData(numSamples)
    ys = hums
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title("Humidity [%]")
    axis.set_xlabel("Samples")
    axis.grid(True)
    xs = range(numSamples)
    axis.plot(xs, ys)
    canvas = FigureCanvas(fig)
    output = io.BytesIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    return response

@app.route('/plot/co2')
def plot_co2():
    dates, co2 = getHistDataSmoke(numSamples)
    ys = co2
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title("Level")
    axis.set_xlabel("Samples")
    axis.grid(True)
    xs = range(len(co2))
    axis.plot(xs, ys)
    canvas = FigureCanvas(fig)
    output = io.BytesIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    return response

files = UploadSet('files', ALL)
app.config['UPLOADED_FILES_DEST'] = 'static/uploadsDB'
configure_uploads(app, files)

@app.route("/table")
def table():
    return render_template('table.html')

@app.route("/dataupload", methods=['GET', 'POST'])
def dataupload():
    if request.method == 'POST' and 'csv_data' in request.files:
        file = request.files['csv_data']
        filename = secure_filename(file.filename)
        file.save(os.path.join('static/uploadsDB', filename))

        date = str(datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"))

        # EDA function
        df = pd.read_csv(os.path.join('static/uploadsDB',filename))
        df_size = df.size
        df_shape = df.shape
        df_columns = list(df.columns)
        df_targetname = df[df.columns[-1]].name

    return render_template('details.html', filename=filename, date=date,
                           df_size=df_size,
                           df_shape=df_shape,
                           df_columns=df_columns,
                           df_targetname=df_targetname,
                           dfplot=df)

@app.route("/prediksi")
def prediksi():
    return render_template('prediksi.html')

@app.route("/result", methods = ['POST', 'GET'])
def result():
    if request.method == 'POST' and 'csv_data' in request.files:
        file = request.files['csv_data']
        teks = pd.read_csv(file, header=0, delimiter=',', encoding='utf-8')
        df = pd.DataFrame(teks)
        label = []
        co = []
        for i in range(len(df)):
            label.append(df.values[i][0])
            co.append(df.values[i][4])

        times, temps, hums, co2 = getHistData(len(label))

        xTarget = df.drop(['tanggal', 'critical', 'categori', 'lokasi_spku'], axis=1)

        yTarget = df['categori']

        labelencoder_class = LabelEncoder()
        Y = labelencoder_class.fit_transform(yTarget)
        tfidf_transformer = OneHotEncoder()
        X = tfidf_transformer.fit_transform(xTarget)

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

        clf = SVC()
        clf = clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        result = accuracy_score(y_pred, y_test)
        kategori = "Sedang" if result > 0.5 else "Tidak Sehat"

        return render_template("prediksi2.html", result=result, kategori=kategori, label=label, co=co, co2=co2)

if __name__ == "__main__":
    app.run(debug=True)
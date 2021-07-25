from flask import Flask, render_template, request, make_response, redirect
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

# ML Packages
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

app = Flask(__name__)

# Retrieve LAST data from database
def getLastData():
    conn = sqlite3.connect('sensorsData.db')
    curs = conn.cursor()
    for row in curs.execute("SELECT * FROM DHT_data ORDER BY timestamp DESC LIMIT 1"):
        time = str(row[0])
        temp = row[1]
        hum = row[2]
    # conn.close()
    return time, temp, hum


# Get 'x' samples of historical data
def getHistData(numSamples):
    conn = sqlite3.connect('sensorsData.db')
    curs = conn.cursor()
    curs.execute("SELECT * FROM DHT_data ORDER BY timestamp DESC LIMIT " + str(numSamples))
    data = curs.fetchall()
    dates = []
    temps = []
    hums = []

    for row in reversed(data):
        dates.append(row[0])
        temps.append(row[1])
        hums.append(row[2])
        temps, hums = testeData(temps, hums)
    return dates, temps, hums


# Test data for cleanning possible "out of range" values
def testeData(temps, hums):
    n = len(temps)
    for i in range(0, n - 1):
        if (temps[i] < -10 or temps[i] > 50):
            temps[i] = temps[i - 2]
        if (hums[i] < 0 or hums[i] > 100):
            hums[i] = temps[i - 2]
    return temps, hums


# Get Max number of rows (table size)
def maxRowsTable():
    conn = sqlite3.connect('sensorsData.db')
    curs = conn.cursor()
    for row in curs.execute("select COUNT(temp) from  DHT_data"):
        maxNumberRows = row[0]
    return maxNumberRows


# Get sample frequency in minutes
def freqSample():
    times, temps, hums = getHistData(2)
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

@app.route("/")
def index():
    return render_template('index3.html')

@app.route("/monitoring")
def monitoring():
    time, temp, hum = getLastData()
    templateData = {
        'time': time,
        'temp': temp,
        'hum': hum,
        'freq': freqSamples,
        'rangeTime': rangeTime
    }
    return render_template('monitoring.html', **templateData)

@app.route("/grafik")
def grafik():
    time, temp, hum = getLastData()
    templateData = {
        'time': time,
        'temp': temp,
        'hum': hum,
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
    time, temp, hum = getLastData()

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
    times, temps, hums = getHistData(numSamples)
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
    times, temps, hums = getHistData(numSamples)
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

@app.route("/table")
def table():
    conn = sqlite3.connect('sensorsData.db')
    conn.row_factory = sqlite3.Row
    curs = conn.cursor()
    curs.execute("select * from data_pencemaran")
    rows = curs.fetchall();

    return render_template("table.html",rows = rows)

@app.route("/delete", methods=['POST','GET'])
def delete(id):
    conn = sqlite3.connect('sensorsData.db')
    curs = conn.cursor()
    curs.execute("DELETE FROM data_pencemaran WHERE id=%s", (id,))
    return render_template("table.html")

db = sqlite3(app)

files = UploadSet('files',ALL)
app.config['UPLOADED_FILES_DEST'] = 'static/uploadsDB'
configure_uploads(app,files)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///static/uploadsDB/filestorage.db'

class FileContents(db.Model):
	id = db.Column(db.Integer,primary_key=True)
	name = db.Column(db.String(300))
	modeldata = db.Column(db.String(300))
	data = db.Column(db.LargeBinary)

@app.route("/grafik2", methods=['GET','POST'])
def dataupload():
    if request.method == 'POST' and 'csv_data' in request.files:
        file = request.files['csv_data']
        filename = secure_filename(file.filename)

        file.save(os.path.join('static/uploadsDB', filename))
        fullfile = os.path.join('static/uploadsDB', filename)

        # For Time
        date = str(datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"))

        # EDA function
        df = pd.read_csv(os.path.join('static/uploadsDB', filename))
        df_size = df.size
        df_shape = df.shape
        df_columns = list(df.columns)
        df_targetname = df[df.columns[-1]].name
        df_featurenames = df_columns[0:-1]  # select all columns till last column
        df_Xfeatures = df.iloc[:, 0:-1]
        df_Ylabels = df[df.columns[-1]]  # Select the last column as target
        # same as above df_Ylabels = df.iloc[:,-1]

        # Model Building
        X = df_Xfeatures
        Y = df_Ylabels
        seed = 7
        # prepare models
        models = []
        models.append(('LR', LogisticRegression()))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC()))
        # evaluate each model in turn

        results = []
        names = []
        allmodels = []
        scoring = 'accuracy'
        for name, model in models:
            kfold = model_selection.KFold(n_splits=10, shuffle=True)
            cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            allmodels.append(msg)
            model_results = results
            model_names = names

    return render_template('grafik2.html',filename=filename,date=date,
		df_size=df_size,
		df_shape=df_shape,
		df_columns =df_columns,
		df_targetname =df_targetname,
		model_results = allmodels,
		model_names = names,
		fullfile = fullfile,
		dfplot = df)

if __name__ == "__main__":
    app.run(debug=True)
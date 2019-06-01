from keras import models
import numpy as np
import  h5py
from keras.layers import Dense, Dropout
from numpy import typename
from keras import activations
from keras import Sequential
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from keras.engine.input_layer  import Input
import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sn
from keras.layers.advanced_activations import PReLU
import datetime as dt



from keras import Sequential
from sklearn.model_selection import train_test_split
import  h5py
import pandas as pd
import tensorflow as tf

import keras
from sklearn.preprocessing import MinMaxScaler
import seaborn as sn
from keras.layers.advanced_activations import PReLU
import datetime as dtt
from keras.models import model_from_json
import tkinter
from keras import backend as K
from keras.models import load_model


seed_value= 30

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.set_random_seed(seed_value)

import numpy as np
import tensorflow as tf
import random as rn

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# creating the keras model
# input_layer = keras.engine.input_layer.Input(shape = (32))

# model save counter
count = 0

datasets = pd.read_csv('dataset_new.csv')
# reshaping the network layer to have a shape of (1,None)
x_input_layer = datasets.Year.values.reshape(20, 1)
y_output = datasets.drop(['Year'], axis=1)
model = Sequential()

# split the data
x_train, x_test, y_train, y_test = train_test_split(x_input_layer, y_output, test_size=0.30)

# scaling the feature
scalerX = MinMaxScaler()
x_train = scalerX.fit_transform(x_train)
x_test = scalerX.transform(x_test)

# scaling the feature
scalerY = MinMaxScaler()
y_train = scalerY.fit_transform(y_train)
y_test = scalerY.transform(y_test)














"------------------MAIN PROGRAM GOES HERE-------------------------"


# Seed value

def train_model():
    # the input_shape = (1,) ==> Each set of the input to be passed into the network
    # the main nueral network model for the prediction
    # the root mean square
    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

    model.add(Dense(40, kernel_initializer='uniform', input_shape=(1,), activation="relu", bias_initializer="zeros"))
    model.add(Dense(40, activation="relu", kernel_initializer="uniform", bias_initializer="zeros"))
    model.add(Dense(40, activation="relu", kernel_initializer="uniform", bias_initializer="zeros"))
    model.add(Dense(40, activation="relu", kernel_initializer="uniform", bias_initializer="zeros"))
    model.add(Dense(40, activation="relu", kernel_initializer="uniform", bias_initializer="zeros"))
    model.add(Dense(40, activation="relu", kernel_initializer="uniform", bias_initializer="zeros"))
    model.add(Dense(40, activation="relu", kernel_initializer="uniform", bias_initializer="zeros"))

    # adding another dense layer of 64 input nuerons
    model.add(Dense(units=3, activation="linear", kernel_initializer="uniform", bias_initializer="zeros"))

    # compiling the model setting the paramenters to train the model
    model.compile(loss=['mse'], metrics=['mse', 'mae', 'mape', 'cosine', 'accuracy', 'squared_hinge', 'cosine','msle',root_mean_squared_error],
                  optimizer=keras.optimizers.Adam
                  (lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))

    model.fit(x_train, y_train, batch_size=10, epochs=3000, verbose=1)

    model.summary()
    model.save_weights("network_weights.h5".format(dt.datetime.now()))
    model.save("model.h5")

    # getting the metrics results from here...
    print("----------METRICS EVALUATION RESULTS---------------------------")
    print("-----------------------MEAN SQUARED ERROR---------------------")
    print(model.history.history['mean_squared_error'])
    print("------------------------MEAN SQUARED LOGARITHMIC ERROR------------------")
    print(model.history.history['mean_squared_logarithmic_error'])
    print("------------------------TRAINING ACCURACY------------------")
    print(model.history.history['acc'])
    print("------------------------MEAN ABSOLUTE ERROR-------------------")
    print(model.history.history['mean_absolute_error'])
    print("------------------------MEAN ABSOLUTE PERCENTAGE ERROR----------------")
    print(model.history.history['mean_absolute_percentage_error'])
    print("------------------------SQAURED HINGE-----------------------------------")
    print(model.history.history['squared_hinge'])
    print("-----------------------CONSINE PROXIMITY--------------------------")
    print(model.history.history['cosine_proximity_1'])
    print("<---------------- ROOT MEAN SQUARED ERROR ----------------->")
    # print(model.history.history['root_mean_squared_error'])

def model_initialise():

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

    global  model
    model = load_model('model.h5');
    print(model)
#
# def predict_unemployment_rates(features_values):
#     features_scaler = MinMaxScaler()
#     predict_input = features_scaler.fit_transform(features_values);
#     print("Yes it was clicked")
#     print(scalerY.inverse_transform( model.predict(predict_input)) )


def predict_unemployment_rate(features_values) :
    global predicted_result
    predicted_result = ""
    count = 0 ;
    rate_string = ""
    # predicted = model.predict(features)
    # print("PREDICTION == ", predicted)

    features_scaler = MinMaxScaler()
    predict_input = features_scaler.fit_transform(features_values);
    print("Yes it was clicked")
    predicted = scalerY.inverse_transform(model.predict(predict_input))
    print(scalerY.inverse_transform( model.predict(predict_input)) )


    for preval in predicted :
        for rate in preval:
            rate_string +="{0}".format(rate) +"\t"
        result ="{0}\t{1}".format(features_values[count][0] , rate_string) +"\n\n"
        rate_string = "";
        predicted_result +=result

        count += 1
    # print(predicted_result) ;

    prediction_results.config(text= predicted_result)
    print("THIS IS THE PREDICTED RESULT ====> ",predicted_result)


def evaluate_prediction_model ():
    model.evaluate(x_test, y_test)

def display_graph ():
    # ploting the history of the losses and the accuracy
    from matplotlib import pylab as plt
    plt.figure(figsize=(15, 10))
    plt.xlabel('Losses', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    # plt.plot(model.history.epoch, np.array(model.history.history['acc']),label='Train Accuracy')
    plt.plot(model.history.history['loss'], model.history.history['acc'], label='Losses and Accuracy')

    plt.legend()

    # ploting the history of the losses and the accuracy
    from matplotlib import pylab as plt
    plt.figure(figsize=(15, 10))
    plt.xlabel('Losses', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    # plt.plot(model.history.epoch, np.array(model.history.history['acc']),label='Train Accuracy')
    plt.scatter(model.history.history['loss'], model.history.history['acc'], label='Losses and Accuracy')

    plt.legend()

    # ploting the history of the losses and the accuracy
    from matplotlib import pylab as plt
    plt.figure(figsize=(15, 10))
    plt.xlabel('Losses', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.plot(model.history.history['loss'], model.history.history['acc'], label='Losses and Accuracy')

    plt.legend()


    scalerY.inverse_transform(model.predict(x_train))

    scalerX.inverse_transform(x_train)

    # ploting the history of the losses and the accuracy
    from matplotlib import pylab as plt
    plt.figure(figsize=(15, 10))
    plt.title("Accuracy Vs Epoch of the Neural Network", fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.plot(model.history.epoch, model.history.history['acc'])
    plt.legend()

    sn.jointplot(x=model.history.epoch, y=model.history.history['loss'])

def load_network_weights():
    # load the weight of the nwetwork
    model.load_weights("network_weights2018-12-04 16:43:26.388594.h5");
    print("Predict X_Train" , scalerY.inverse_transform(model.predict(x_train)) )
    print("Predict X_Test" , scalerY.inverse_transform(model.predict(x_test)) )
    print( "Evaluate X_Train and Y_Train",model.evaluate(x_train, y_train))
    print( "Evaluate X_test and Y_test",model.evaluate(x_test, y_test))



unemployemnt_rate  = 0 ;
unemployemnt_rate_of_females = 0 ;
unemployemnt_rate_of_male = 0 ;

root = tkinter.Tk()
frame = tkinter.Frame(root)

label_notice = tkinter.Label(root , text = "Please enter a single year or multiple year separated with a comma delimeter")
label_notice.pack()

input_value = tkinter.StringVar()
entry = tkinter.Entry(root , width = 200  , textvariable = input_value, font ="Helvetical 44 bold" )
entry.pack()





value_label = tkinter.Label(root , text = "" ,  width = 200) ;
value_label.pack()


label_enter = tkinter.Label(root, fg="dark green", text=" Enter Year : ")
label_enter.pack()
#

label_message = tkinter.Label(root , text = "Message : ")
label_message.pack()

label_enter.pack()

# predic_btn.pack()
root.title("UNEMPLOYMENT PREDICTION IN NIGERIA")



def validate_and_predict():
    try:
        features = [[float(val)] for val in input_value.get().split(",")]
        print([[value] for value in input_value.get().split(",")])
        print(features)
        for ft in features :
            if (ft[0] // 1000) > 0 :
                print(ft[0] / 1000)
                print("value =>", ft, " accepted")
                label_message.config(text="Prediction was successful...", fg="green");
                continue
            else :
                label_message.config(text="The year to predict should not be less than 1000", fg="red");
                return Exception()

        #predict if no error occurs [ 12.28609276,  56.2775383 ,  43.82889175],
        predict_unemployment_rate(features)

    except Exception:
        print("Error occured while casting value to number")
        label_message.config(text = "Wrong input characters...please enter correct values", fg = "red");

btn = tkinter.Button(root , width = 200 , text = "Predict", command = validate_and_predict )
btn.pack()

label_result = tkinter.Label(root , text = "---------Results-------")
label_result.pack()

label_result = tkinter.Label(root , text = "Unemployment Rate : {0} | Unemployment Rate Of Male : {1} | Unemployment Rate Of Female : {2}".format(unemployemnt_rate ,unemployemnt_rate_of_male, unemployemnt_rate_of_females))
label_result.pack()

prediction_results = tkinter.Label(root , text = "")
prediction_results.pack()




#display the GUI for the User to input the prediction value
if __name__ == "__main__" :
    train_model()
    model_initialise()
    # load_network_weights()

    root.geometry('600x600')
    root.mainloop()


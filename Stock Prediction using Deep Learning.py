#!/usr/bin/env python
# coding: utf-8

# # <center> STOCK PREDICTION USING DEEP LEARNING </center>
# 
# ## <center> ABOUT THE DATA </center>
# 
# ### We have five datasets consisting of the stock prices of five companies (Facebook, Amazon, Apple, Netflix and Google) from 2012 till March 2021.
# 
# ### The datasets have 6 columns - Date, Open, High, Low, Close, Adj Close and Volume
# 
#  - Date      : Trading date of the stock
#  - Open      : Stock’s opening price 
#  - High      : Highest stock price on a particular trading day
#  - Low       : Lowest stock price on a particular trading day
#  - Close     : Stock's closing price
#  - Adj Close : Ending or closing price of the stock which was changed to contain any corporations’ actions and distribution that is occurred during trade time of the day
#  - Volume    : Number of stocks traded on a particular day.
#  

# ### IMPORTING LIBRARIES

# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM,Dropout
from sklearn.metrics import accuracy_score
from tensorflow.keras import datasets, layers,models
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Activation,SimpleRNN
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from pykalman import KalmanFilter
import PySimpleGUI as sg
import plotly.graph_objects as go
from datetime import datetime
tf.random.set_seed(7)


# ### READING AND ANALYSING THE DATA

# In[32]:


# To plot the Kalman filtered price fluctuation
def price_fluct(company):
    
        kf = KalmanFilter(transition_matrices = [1], observation_matrices = [1], initial_state_mean = 0, 
                         initial_state_covariance = 1, observation_covariance = 1,transition_covariance = 0.0001)
        mean, cov = kf.filter(company['Adj Close'].values)
        mean, std = mean.squeeze(), np.std(cov.squeeze())
        plt.figure(figsize=(12,6))
        plt.plot(company['Adj Close'].values - mean, 'red', lw=1.5)
        plt.xticks(range(0,company.shape[0],500),company['Date'].loc[::500],rotation=45)
        plt.title("Kalman filtered price fluctuation")
        plt.ylabel("Deviation from the mean ($)")
        plt.xlabel("Days")

# Interactive candlestick visualization of the company's stock data 
def candlestick_viz(company):
    
    fig = go.Figure(data=[go.Candlestick(x=Facebook['Date'],open=Facebook['Open'],high=Facebook['High'],
                low=Facebook['Low'], close=Facebook['Close'])])

    fig.show()
    


# In[33]:


Facebook = pd.read_csv(r"C:\Users\Om Bhandwalkar\Downloads\META (1).csv")
Facebook


# In[34]:


price_fluct(Facebook)


# ![image.png](attachment:image.png)

# In[35]:


candlestick_viz(Facebook)


# In[36]:


Amazon = pd.read_csv(r"C:\Users\Om Bhandwalkar\Downloads\AMZN.csv")
Amazon


# In[37]:


price_fluct(Amazon)


# In[38]:


candlestick_viz(Amazon)


# In[39]:


Apple = pd.read_csv(r"C:\Users\Om Bhandwalkar\Downloads\AAPL.csv")
Apple


# In[40]:


price_fluct(Apple)


# In[41]:


candlestick_viz(Apple)


# In[42]:


Netflix = pd.read_csv(r"C:\Users\Om Bhandwalkar\Downloads\NFLX.csv")
Netflix


# In[43]:


price_fluct(Netflix)


# In[44]:


candlestick_viz(Netflix)


# In[45]:


Google = pd.read_csv(r"C:\Users\Om Bhandwalkar\Downloads\GOOG.csv")
Google


# In[46]:


price_fluct(Google)


# In[47]:


candlestick_viz(Google)


# In[48]:


# Plotting the raw closing prices of all the companies from 2012 till March 2021
plt.plot(range(Facebook.shape[0]),(Facebook['Close']))
plt.plot(range(Amazon.shape[0]),(Amazon['Close']))
plt.plot(range(Apple.shape[0]),(Apple['Close']))
plt.plot(range(Netflix.shape[0]),(Netflix['Close']))
plt.plot(range(Google.shape[0]),(Google['Close']))

plt.xticks(range(0,Apple.shape[0],500),Apple['Date'].loc[::500],rotation=45)

plt.legend(["Facebook","Amazon","Apple","Netflix","Google"])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Closing Price',fontsize=18)


# ### DEEP LEARNING MODELS

# <img src="data:image/png;base64,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" alt="image">

# ### TRAINING AND TESTING THE MODEL

# In[49]:


class stock_predict_DL:
    
    def __init__(self,comp_df):
        # reseved method in python classes (Constructor)
        # We are taking only the Open prices for predicting 
        data = comp_df.filter(['Open'])
        dataset = data.values
        # We take 90% of the data for training and 10% for testing 
        training_data_len = int(np.ceil( len(dataset) * 0.90 ))
        # We are scaling the open prices to the range(0,1)
        self.scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = self.scaler.fit_transform(dataset)
        # Taking the first 90% of the dataset for training 
        train_data = scaled_data[0:int(training_data_len), :]
        # Split the data into self.X_train and self.y_train data sets
        self.X_train = []
        self.y_train = []
        
        # We are taking predicting the open price of a given day based on the trend in the previous 60 days
        for i in range(60, len(train_data)):
            self.X_train.append(train_data[i-60:i, 0])
            self.y_train.append(train_data[i, 0])

        # Convert the self.X_train and self.y_train to numpy arrays 
        self.X_train, self.y_train = np.array(self.X_train), np.array(self.y_train)

        # Create the testing data set
        # Create a new array containing scaled values from index 1543 to 2002 
        test_data = scaled_data[training_data_len - 60: , :]
        # Create the data sets self.X_test and self.y_test
        self.X_test = []
        # Rmaining 10% of the data needs to be given for testing 
        self.y_test = dataset[training_data_len:, :]
        for i in range(60, len(test_data)):
            self.X_test.append(test_data[i-60:i, 0])

        # Convert the data to a numpy array
        self.X_test = np.array(self.X_test)
        test_dates = comp_df['Date'].values
        self.testd = test_dates[training_data_len:] # stores the test dates
        # List to store the R2 scores of all the models to get the best model at the end
        self.model_score = []
        
    def LSTM_model(self):
        
        print("Long Short-Term Memory (LSTM)")
        # Reshape the data
        Xtrain = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        # Reshape the data
        Xtest = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1 ))
        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape= (Xtrain.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(32))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        # We are adding dropout to reduce overfitting 
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        # Train the model
        model.fit(Xtrain, self.y_train, batch_size=1, epochs= 1)
         # Get the models predicted price values 
        predictions = model.predict(Xtest)
        # We need to inverse transform the scaled data to compare it with our unscaled y_test data
        predictions = self.scaler.inverse_transform(predictions)
        print("R2 SCORE")
        print(metrics.r2_score(self.y_test, predictions))
        self.model_score.append(["LSTM",metrics.r2_score(self.y_test, predictions)])
        # Mean squared logarithmic error (MSLE) can be interpreted as a measure of the
        # ratio between the true and predicted values.
        print("MSLE")
        print(metrics.mean_squared_log_error(self.y_test, predictions))
        plt.plot(predictions)
        plt.plot(self.y_test)
        plt.legend(["Predicted","Observed"])
        plt.xticks(range(0,len(self.y_test),50),self.testd,rotation=45)
        plt.xlabel('Date',fontsize=18)
        plt.ylabel('Price',fontsize=18)
        plt.title("LSTM")
        
    def basic_ann_model(self):
        
        print("Basic Artificial Neural Network (ANN)")
        classifier = Sequential()
        classifier.add(Dense(units = 128, activation = 'relu', input_dim = self.X_train.shape[1]))
        classifier.add(Dropout(0.2))
        classifier.add(Dense(units = 64))
        classifier.add(Dropout(0.2))
        classifier.add(Dense(units = 1))
        # We are adding dropout to reduce overfitting
        # adam is one of the best optimzier for DL as it uses stochastic gradient method
        # Mean Square Error (MSE) is the most commonly used regression loss function.
        # MSE is the sum of squared distances between our target variable and predicted values.
        classifier.compile(optimizer = 'adam', loss = 'mean_squared_error')
        classifier.fit(self.X_train, self.y_train, batch_size = 32, epochs = 10)
        # Predicting the prices
        prediction = classifier.predict(self.X_test)
        y_pred = self.scaler.inverse_transform(prediction)
        print("R2 SCORE")
        print(metrics.r2_score(self.y_test, y_pred))
        # Appending the R2 score
        self.model_score.append(["ANN",metrics.r2_score(self.y_test, y_pred)])
        print("MSLE")
        print(metrics.mean_squared_log_error(self.y_test, y_pred))
        plt.plot(y_pred)
        plt.plot(self.y_test)
        plt.legend(["Predicted","Observed"])
        plt.xticks(range(0,len(self.y_test),50),self.testd,rotation=45)
        plt.xlabel('Date',fontsize=18)
        plt.ylabel('Price',fontsize=18)
        plt.title("ANN")
    
        
    def autoen_model(self):
        
        print("Autoencoder")
        # No of encoding dimensions
        encoding_dim = 32
        input_dim = self.X_train.shape[1]
        input_layer = Input(shape=(input_dim, ))
        # Encoder
        encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(1e-5))(input_layer)
        encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
        # Decoder
        decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
        decoder = Dense(1, activation='relu')(decoder)
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        nb_epoch = 10
        b_size = 32
        # Fitting and compiling the train data using adam (stochastic gradient) optimiser and mse loss
        autoencoder.compile(optimizer='adam',loss='mean_squared_error')
        autoencoder.fit(self.X_train, self.y_train,epochs=nb_epoch,batch_size = b_size,shuffle=True)
        predictions = autoencoder.predict(self.X_test)
        predictions = self.scaler.inverse_transform(predictions)
        print("R2 SCORE")
        print(metrics.r2_score(self.y_test, predictions))
        self.model_score.append(["Autoencoder",metrics.r2_score(self.y_test, predictions)])
        print("MSLE")
        print(metrics.mean_squared_log_error(self.y_test, predictions))
        plt.plot(predictions)
        plt.plot(self.y_test)
        plt.legend(["Predicted","Observed"])
        plt.xticks(range(0,len(self.y_test),50),self.testd,rotation=45)
        plt.xlabel('Date',fontsize=18)
        plt.ylabel('Price',fontsize=18)
        plt.title("AUTOENCODER")
        
    def Mlp_model(self):
        
        print("Multilayer perceptron (MLP)")
        # We are using MLPRegressor as the problem at hand is a regression problem
        regr = MLPRegressor(hidden_layer_sizes = 100, alpha = 0.01,solver = 'lbfgs',shuffle=True)
        regr.fit(self.X_train, self.y_train)
        # predicting the price
        y_pred = regr.predict(self.X_test)
        y_pred = y_pred.reshape(len(y_pred),1)
        y_pred = self.scaler.inverse_transform(y_pred)
        print("R2 SCORE")
        print(metrics.r2_score(self.y_test, y_pred))
        # Appending the model score and printing the mean squared log error
        self.model_score.append(["MLP",metrics.r2_score(self.y_test, y_pred)])
        print("MSLE")
        print(metrics.mean_squared_log_error(self.y_test, y_pred))
        plt.plot(y_pred)
        plt.plot(self.y_test)
        plt.legend(["Predicted","Observed"])
        plt.xticks(range(0,len(self.y_test),50),self.testd,rotation=45)
        plt.xlabel('Date',fontsize=18)
        plt.ylabel('Price',fontsize=18)
        plt.title("MLP")
        
 
    def rnn_model(self):
        
        print("Recurrent neural network (RNN)")
        # Reshape the data
        Xtrain = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        # Reshape the data
        Xtest = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1 ))
        model = Sequential()
        model.add(SimpleRNN(units=4, input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(Xtrain, self.y_train, epochs=10, batch_size=1)
        # predicting the opening prices
        prediction = model.predict(Xtest)
        y_pred = self.scaler.inverse_transform(prediction)
        print("R2 SCORE")
        # Appending the R2 score
        print(metrics.r2_score(self.y_test, y_pred))
        self.model_score.append(["RNN",metrics.r2_score(self.y_test, y_pred)])
        print("MSLE")
        print(metrics.mean_squared_log_error(self.y_test, y_pred))
        plt.plot(y_pred)
        plt.plot(self.y_test)
        plt.legend(["Predicted","Observed"])
        plt.xticks(range(0,len(self.y_test),50),self.testd,rotation=45)
        plt.xlabel('Date',fontsize=18)
        plt.ylabel('Price',fontsize=18)
        plt.title("RNN")
        
    def best_model(self):
        #function to find the best model based on the accuracies of the models trained
        print(self.model_score)
        Dict = {item[0]: item[1:][0] for item in self.model_score}
        keys = list(Dict.keys()) # creating a list of Dict keys
        vals = list(Dict.values()) # creating a list of Dict values
        print("The best model is ",keys[vals.index(max(vals))]) # getting the model with the highest accuracy
        print("Accuracy of ",keys[vals.index(max(vals))],'is',max(vals)) #getting the accuracy of the best model
      


# In[50]:


import PySimpleGUI as sg


# ## Choosing the data and Calling the model

# In[51]:


# GUI
# builds a window where we can browse for our data file
sg.theme("DarkTeal2")
layout = [[sg.T("")], [sg.Text("Choose a file (.csv): "), sg.Input(), sg.FileBrowse(key="-IN-")], [sg.Button("Select")]]

# Building Window
window = sg.Window('Stock Dataset Browser', layout, size=(600, 150))

# If the user presses the select button, then store the file path
# After selecting it, it closes the popup
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == "Exit":
        break
    elif event == "Select":
        print(values["-IN-"])
        filepath = values["-IN-"]
        window.close()


# In[52]:


folder = 'C:/Users/Om Bhandwalkar/Downloads/AAPL.csv'
# reading the csv file that was selected and printing the file name 
data = pd.read_csv(filepath)
comp_name = filepath.replace(folder,"")
comp_name = comp_name.replace('.csv',"")
print("Company " + comp_name + "'s stocks chosen")


# In[53]:


# creating an object company_stock for the class stock_predict_DL
company_stock = stock_predict_DL(data)


# In[68]:


company_stock.LSTM_model()


# In[31]:


company_stock.autoen_model()


# In[54]:


company_stock.Mlp_model()


# In[71]:


company_stock.basic_ann_model()


# In[55]:


company_stock.rnn_model()


# In[56]:


company_stock.best_model()


# In[ ]:





# Guess Length of Theoretical UFC Round - LSTM
by Megan Tran

## Table of Contents
* [Purpose of Program](#Purpose-of-program)
* [Screenshots](#screenshots)
* [Technologies](#technologies)
* [Setup](#setup)
* [Using the Program](#Using-the-Program)
* [Credits](#Credits)

## Purpose of Program

This program was created to predict the length of the next three rounds of a fight using previous randomly generated times for each round using the Long Short Term Memory (LSTM) model. 

## Screenshots

<img width="360" alt="image" src="https://github.com/Sonicdaheghod/Long-Short-Term-Memory-LSTM-_Practice/assets/68253811/10d59077-2498-44eb-8cec-50be04152140">

Pandas dataframe to show randomly generated times for each UFC fight game - three rounds each game.

<img width="720" alt="image" src="https://github.com/Sonicdaheghod/Long-Short-Term-Memory-LSTM-_Practice/assets/68253811/5f4af440-844a-4fde-802e-513c38653bbf">

Training the LSTM model.

<img width="720" alt="image" src="https://github.com/Sonicdaheghod/Long-Short-Term-Memory-LSTM-_Practice/assets/68253811/685bd00b-d19d-4fc8-8cbb-c40b1d7cc507">

Prediction of the times for each round of a future UFC fight. 

## Technologies
Languages/ Technologies used:

* Jupyter Notebook

* Python3

## Setup

Download the necessary packages:
```
pip install scikit-learn
pip install pandas
pip install keras
pip install numpy
```
Check to see if version of Python/Python3 (if on jupyter Notebook) is used, this is to ensure packages work properly. Python 3.8 or better is recommended.

```
import sys
sys.version
```

Import the following packages and libraries:

```
# Data processing
import pandas as pd
import numpy as np

#Helps create the LSTM model
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout

```
  
## Using the Program

1) Data Preparation
<img width="300" alt="image" src="https://github.com/Sonicdaheghod/Long-Short-Term-Memory-LSTM-_Practice/assets/68253811/b773d998-4cff-48ae-b22b-b5834c617d34">

* columns are for round numbers, rows are for UFC fight game number
* Data time values randomly generated using ``` np.random.uniform ```
  
2) Training LSTM Model

First load the model, which is made up of four layers where data is processed and prevents overfitting for the model.

```
#import libraries

from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout
#import numpy as np

batch_size = 55
model = Sequential()
model.add(Bidirectional(LSTM(240,
            input_shape=(num_games, features_games),
            return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(240,
            input_shape=(num_games, features_games),
            return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(240,
            input_shape=(num_games, features_games),
            return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(240,
            input_shape=(num_games, features_games),
            return_sequences=False)))
model.add(Dense(59))
model.add(Dense(features_games))
model.compile(loss="mse", optimizer="rmsprop", metrics=["accuracy"])

```
Then run this code to pass through sets of data from our randomly generated dataset into model multiple times for training.
```
model.fit(fight_train, fight_label, 
         batch_size = 55, epochs = 250)

```
3) Evaluating Model
   
* This is done when we view the accuracy and loss when training the LSTM model.
* Use code ``` model.compile(loss="mse", optimizer="rmsprop", metrics=["accuracy"]) ```
<img width="600" alt="image" src="https://github.com/Sonicdaheghod/Long-Short-Term-Memory-LSTM-_Practice/assets/68253811/65f993a1-3c6f-473d-a78d-834ff66ce654">


4) Making Predictions of Future UFC Fight Durations

Use test data not used for training and put it through the trained LSTM model to predict duration of next UFC fight rounds.

```
#using times for last few games, randomly generated times 
predict_nextUFC = np.random.uniform(min_float,max_float, size=(5,columns))
scaled_predict= scaler.transform(predict_nextUFC)

#Prediction
scaled_predictfight_output = model.predict(np.array([scaled_predict]))
print(scaler.inverse_transform(scaled_predictfight_output).astype(float)[0].round(2))

```

## Credits

* Tutorial referenced: [Predict Lottery Numbers using Artificial Intelligent Neural Network in Kera, Python by Arnold Dalby](https://youtu.be/vN_EuIfD42g)



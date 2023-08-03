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

## Credits

* Tutorial referenced: [Predict Lottery Numbers using Artificial Intelligent Neural Network in Kera, Python by Arnold Dalby](https://youtu.be/vN_EuIfD42g)



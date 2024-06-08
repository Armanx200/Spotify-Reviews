import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# Import Data
df = pd.read_csv('Dataset.csv')

# Data INFO
print(df.head())
print(df.describe())
print(df.info())
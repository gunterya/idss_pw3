"""
Created on Thu May 17 11:04:15 2018

@author: polizzi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns

df = pd.read_csv('/Users/isabellepolizzi/Desktop/UPC/IDSS/PW3/idss_pw3/data/processed_cleveland_data.csv')
df = df.replace('?',0)

# Normalize data between 0 and 1
x = df.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
data_scaled = pd.DataFrame(x_scaled, columns=df.columns)
data_scaled.head()

# Data boxplot
data_scaled.boxplot()
plt.show()

# Data distribution
data_scaled.plot(kind='kde', subplots=True)
plt.xlim(-0.1,1,1)
plt.show()

# Histogram
bins = int(np.ceil(np.log2(data_scaled.values.shape[0]) + 1)) # Sturge's rule for the number of bins
data_scaled.plot(kind='hist', bins=bins, subplots=True)
plt.show()

# Heatmap
sns.heatmap(data_scaled.corr(method='pearson'))
plt.show()
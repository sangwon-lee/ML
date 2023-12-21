from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
data = load_iris()
data.keys()

print(data.data.shape, data.target.shape)
print(data.target_names, data.feature_names)

print(data.DESCR)

iris = pd.DataFrame(data.data, columns=data.feature_names)
iris.head()

np.unique(data.target, return_counts=True)

print(data.target_names.shape, data.target_names[data.target].shape)

np.hstack((data.target.reshape(-1,1), data.target_names[data.target].reshape(-1,1)))

iris.columns = ['sl', 'sw', 'pl', 'pw']
iris['Species'] = data.target_names[data.target]
iris.head()
iris.isna().sum()

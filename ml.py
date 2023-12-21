from sklearn.datasets import load_iris
data = load_iris()
data.keys()

print(data.data.shape, data.target.shape)
print(data.target_names, data.feature_names)

print(data.DESCR)

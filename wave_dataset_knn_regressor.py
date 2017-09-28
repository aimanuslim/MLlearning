from sklearn.neighbors import KNeighborsRegressor
import mglearn
from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_wave(n_samples=40)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# instantiate the model, set the number of neighbors to consider to 3
reg = KNeighborsRegressor(n_neighbors=3)

# Fit the model using the training data and training targets
reg.fit(X_train, y_train)

KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=3, p=2, weights='uniform')

print(reg.predict(X_test))


print(reg.score(X_test, y_test))

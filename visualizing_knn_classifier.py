import matplotlib.pyplot as plt
import mglearn
from sklearn.neighbors import KNeighborsClassifier

X, y = mglearn.datasets.make_forge()
fig, axes = plt.subplots(1,3,figsize=(10,3))

for n_neighbors, ax in zip([1,3, 9], axes):
	clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
	mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
	ax.scatter(X[:, 0], X[:, 1], c=y, s=60, cmap=mglearn.cm2)
	ax.set_title("%d neighbor(s)" % n_neighbors)


plt.show()

from sklearn.datasets import load_boston, load_breast_cancer
import mglearn

cancer = load_breast_cancer
boston = load_boston
X, y = mglearn.datasets.load_extended_boston()
print(X.shape)



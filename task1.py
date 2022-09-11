from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import numpy as np


class RFFPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=1000, new_dim=50, use_PCA=True, classifier='logreg'):
        """
        Implements pipeline, which consists of PCA decomposition,
        Random Fourier Features approximation and linear classification model.

        n_features, int: amount of synthetic random features generated with RFF approximation.

        new_dim, int: PCA output size.

        use_PCA, bool: whether to include PCA preprocessing.

        classifier, string: either 'svm' or 'logreg', a linear classification model to use on top of pipeline.

        Feel free to edit this template for your preferences.
        """
        self.n_features = n_features
        self.use_PCA = use_PCA
        self.new_dim = new_dim
        self.classifier = classifier
        if use_PCA:
            self.PCA = PCA(n_components=new_dim)
        if classifier == 'logreg':
            self.model = LogisticRegression(multi_class='ovr')
        elif classifier == 'svm':
            self.model = LinearSVC(multi_class='ovr')

    def fit(self, X, y):
        """
        Fit all parts of algorithm (PCA, RFF, Classification) to training set.
        """
        if self.use_PCA:
            X = self.PCA.fit_transform(X)
        first_indexes = np.random.choice(X.shape[0], 1500000)
        second_indexes = np.random.choice(X.shape[0], 1500000)
        mask = first_indexes != second_indexes
        first_indexes = first_indexes[mask]
        second_indexes = second_indexes[mask]
        self.sigma_sqr = np.median(np.sum((X[first_indexes] - X[second_indexes]) ** 2, axis=1))
        self.W = np.random.normal(0.0, 1 / self.sigma_sqr, (X.shape[1], self.n_features))
        self.b = np.random.uniform(-np.pi, np.pi, (1, self.n_features))
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X = (X - self.mean[np.newaxis, :]) / self.std[np.newaxis, :]
        X = np.cos(X @ self.W + self.b)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Apply pipeline to obtain scores for input data.
        """
        if self.classifier == 'svm':
            raise AttributeError('SVM do not predict probability')
        if self.use_PCA:
            X = self.PCA.transform(X)
        X = (X - self.mean[np.newaxis, :]) / self.std[np.newaxis, :]
        X = np.cos(X @ self.W + self.b)
        return self.model.predict_proba(X)

    def predict(self, X):
        """
        Apply pipeline to obtain discrete predictions for input data.
        """
        if self.use_PCA:
            X = self.PCA.transform(X)
        X = (X - self.mean[np.newaxis, :]) / self.std[np.newaxis, :]
        X = np.cos(X @ self.W + self.b)
        return self.model.predict(X)

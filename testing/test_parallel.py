import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from neuraxle.base import Identity
from neuraxle.hyperparams.distributions import RandInt, LogUniform, Boolean
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.parallel import SaverParallelTransform
from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import NumpyTranspose
from neuraxle.steps.sklearn import SKLearnWrapper
from neuraxle.union import AddFeatures, ModelStacking


class HeavyIdentity(Identity):
    def __init__(self, heavy):
        Identity.__init__(self)
        self.heavy = heavy


def test_boston_housing(tmpdir):
    boston = load_boston()
    X, y = shuffle(boston.data, boston.target, random_state=13)
    X = X.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
    test_data_inputs = np.repeat(X_train, 1000, axis=0)

    boston_housing_pipeline = Pipeline([
        HeavyIdentity(heavy=np.repeat(X_train, 100000, axis=0)),
        AddFeatures([
            SKLearnWrapper(PCA(n_components=13), HyperparameterSpace({"n_components": RandInt(1, 3)})),
            SKLearnWrapper(FastICA(n_components=13), HyperparameterSpace({"n_components": RandInt(1, 3)})),
        ], n_jobs=1),
        ModelStacking([
            SKLearnWrapper(
                GradientBoostingRegressor(),
                HyperparameterSpace({"n_estimators": RandInt(50, 600), "max_depth": RandInt(1, 10),
                                     "learning_rate": LogUniform(0.07, 0.7)})
            ),
            SKLearnWrapper(KMeans(), HyperparameterSpace({"n_clusters": RandInt(5, 10)})),
        ], joiner=NumpyTranspose(), judge=SKLearnWrapper(Ridge(), HyperparameterSpace(
            {"alpha": LogUniform(0.7, 1.4), "fit_intercept": Boolean()})), n_jobs=1)
    ])

    boston_housing_pipeline = boston_housing_pipeline.fit(X_train, y_train)

    a = time.time()
    # outputs = boston_housing_pipeline.transform(test_data_inputs)
    b = time.time()
    diff = b - a
    print('\n')
    print('without parallel')
    print(diff)

    p = Pipeline([
        SaverParallelTransform(
            boston_housing_pipeline,
            mount_path='cache',
            n_jobs=12,
            batch_size=100
        )
    ], cache_folder=tmpdir)

    a = time.time()
    actual_outputs = p.transform(np.repeat(X_train, 10, axis=0))
    b = time.time()
    diff = b - a
    print('\n')
    print('with saver parallel transform')
    print(diff)


def test_saver_parallel_transform(tmpdir):
    boston = load_boston()
    X, y = shuffle(boston.data, boston.target, random_state=13)
    X = X.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
    sk_learn_pca = SKLearnWrapper(PCA(n_components=2), HyperparameterSpace({"n_components": RandInt(1, 3)}))
    sk_learn_pca = sk_learn_pca.fit(X_train, y_train)
    expected_outputs = sk_learn_pca.transform(X_train)

    p = Pipeline([
        SaverParallelTransform(
            sk_learn_pca,
            n_jobs=10,
            batch_size=10
        )
    ], cache_folder=tmpdir)

    actual_outputs = p.transform(X_train)

    assert np.array_equal(expected_outputs, actual_outputs)

import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor
from keras_preprocessing.sequence import pad_sequences
import keras
from util import TEXT_COLUMNS, TARGET_COLUMNS
from regression_model import x_numerical_test, x_numerical_train, y_train_delay, y_train_length, y_test_delay, y_test_length
from sklearn.decomposition import PCA

def test_dummy_regressor() -> None:
    strat = 'mean'
    dummy_regr_delay = DummyRegressor(strategy=strat)
    dummy_regr_length = DummyRegressor(strategy=strat)
    dummy_regr_delay.fit(x_numerical_train, y_train_delay.values)
    dummy_regr_length.fit(x_numerical_train, y_train_length.values)
    result_delay = dummy_regr_delay.score(x_numerical_test, y_test_delay.values)
    result_length = dummy_regr_length.score(x_numerical_test, y_test_length.values)
    print(f'R^2 value for dummy regressor using {strat} strategy is\nDelay: {result_delay}\nLength: {result_length}')

def principal_component_analysis() -> None:
    '''
    ADAPTED FROM: https://www.datacamp.com/tutorial/principal-component-analysis-in-python
    - The idea is to reduce the amount of dimensions/features to capture the variance in the data
    '''
    print(f'Number of numerical features originally: {np.shape(x_numerical_train)[-1]}')
    pca = PCA(copy=True, iterated_power='auto', n_components=0.95, random_state=None, svd_solver='auto', tol=0.0, whiten=False)
    pca.fit(x_numerical_train)
    print(f'Number of numerical features after PCA: {pca.n_components_}')
    return None

def main():
    principal_component_analysis()

if __name__ == '__main__':
    main()


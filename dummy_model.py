import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor
from keras_preprocessing.sequence import pad_sequences
import keras
from util import TEXT_COLUMNS, TARGET_COLUMNS
from model import x_numerical_test, x_numerical_train, y_train_delay, y_train_length, y_test_delay, y_test_length

def main():
    strat = 'mean'
    dummy_regr_delay = DummyRegressor(strategy=strat)
    dummy_regr_length = DummyRegressor(strategy=strat)
    dummy_regr_delay.fit(x_numerical_train, y_train_delay.values)
    dummy_regr_length.fit(x_numerical_train, y_train_length.values)
    result_delay = dummy_regr_delay.score(x_numerical_test, y_test_delay.values)
    result_length = dummy_regr_length.score(x_numerical_test, y_test_length.values)
    print(f'R^2 value for dummy regressor using {strat} strategy is\nDelay: {result_delay}\nLength: {result_length}')

if __name__ == '__main__':
    main()


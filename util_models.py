import pandas as pd
import numpy as np
from math import sqrt
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from util import TEXT_COLUMNS, TARGET_COLUMNS
from regression_model import x_numerical_test, x_numerical_train, y_train_delay, y_train_length, y_test_delay, y_test_length
from sklearn.decomposition import PCA
import xgboost as xgb

def test_other_models() -> None:
    models = {
        'Dummy Regressor': DummyRegressor(strategy='mean'),
        'Random Forest': RandomForestRegressor(),
        'Linear Regression': LinearRegression(), 
        'K-Nearest Neighbors': KNeighborsRegressor(),
        'XGBOOST': xgb.XGBRegressor()
    }
    for name, model in models.items():
        # Fit model
        delay = model.fit(x_numerical_train, y_train_delay.values)
        length = model.fit(x_numerical_train, y_train_length.values)
        # Get pred values
        delay_pred = delay.predict(x_numerical_test)
        length_pred = length.predict(x_numerical_test)
        # Get r2 values
        delay_r2 = r2_score(y_test_delay.values, delay_pred)
        length_r2 = r2_score(y_test_length.values, length_pred)
        # Get MSE and RMSE
        delay_mse = mean_squared_error(y_test_delay, delay_pred)
        delay_rmse = sqrt(delay_mse)
        length_mse = mean_squared_error(y_test_length, length_pred)
        length_rmse = sqrt(length_mse)
        # Get MAE
        delay_mae = mean_absolute_error(y_test_delay, delay_pred)
        length_mae = mean_absolute_error(y_test_length, length_pred)
        # Get MAPE
        delay_mape = mean_absolute_percentage_error(y_test_delay, delay_pred)
        length_mape = mean_absolute_percentage_error(y_test_length, length_pred)
        # Display results
        print(f'*****{name}*****')
        print(f'Delay MSE: {delay_mse}\nLength MSE: {length_mse}')
        print(f'Delay RMSE: {delay_rmse}\nLength RMSE: {length_rmse}')
        print(f'Delay MAE: {delay_mae}\nLength MAE: {length_mae}')
        print(f'Delay MAPE: {delay_mape}\nLength MAPE: {length_mape}')
        print('-' * 30)

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
    test_other_models()

if __name__ == '__main__':
    main()


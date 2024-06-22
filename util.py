import numpy as np
import pandas as pd
import keras
import sys
import os

def first_value_loc(list_of_lists: list, index: int):
    if len(list_of_lists) > 0:
        while isinstance(list_of_lists[0], list):
            list_of_lists = list_of_lists[0]
        return list_of_lists[index]
    else:
        return None
    
def cyclic_encode(df: pd.DataFrame, column: str, max_val, drop=True) -> pd.DataFrame:
    '''
    ADAPTED FROM: https://www.kaggle.com/code/avanwyk/encoding-cyclical-features-for-deep-learning
    - Function takes in a dataframe, desired column, and max value within that column
    - It applies cyclic encoding to the column to get the sin and cos values (adds _sin and _cos to col name)
    - Drops the original column by default, the returns original dataframe 
    '''
    df[column + '_sin'] = np.sin((2 * np.pi * df[column]) / max_val)
    df[column + '_cos'] = np.cos((2 * np.pi * df[column]) / max_val)
    if drop:
        df.drop(columns=column, inplace=True)
    return df

def one_hot_encode(df: pd.DataFrame, column: str, drop=True) -> pd.DataFrame:
    '''
    - Function takes in a datafram, desired column, and applied one hot encoding to it
    - Added columns take the names of the values within the column
    - Drops the orignial column by default, then returns original dataframe
    '''
    for value in df[column].unique():
        df[value] = df[column].apply(lambda x: 1 if not pd.isna(x) and x == value else 0)
    if drop:
        df.drop(columns=column, inplace=True)
    return df

def save_model(model: keras.Model, result: float, r2_delay: float, r2_length: float) -> None:
    '''
    - Function takes in a keras model and the metrics used for detetmining efficacy
    - Creates a text and model file of the model with the metrics saved based on the file running this
    - If there is a file that already exists it will check its metric file to compare and ask if you want to overwrite
    '''
    def create_txt_file(file_path: str) -> None:
        '''
        - Inner function to create text file
        - WARNING THIS IS DESTRUCTIVE
        '''
        with open(file_path, 'w') as file:
            lines = []
            lines.append(f'Model: {parent_file}\n')
            lines.append(f'Overall loss: {result[0]}\n')
            lines.append(f'Delay MSE: {result[2]}\n')
            lines.append(f'Length MSE: {result[5]}\n')
            lines.append(f'Delay RMSE: {result[3]}\n')
            lines.append(f'Length RMSE: {result[6]}\n')
            lines.append(f'Delay MAE: {result[1]}\n')
            lines.append(f'Length MAE: {result[4]}\n')
            lines.append(f'Delay R^2: {r2_delay}\n')
            lines.append(f'Length R^2: {r2_length}\n')
            file.writelines(lines)

    parent_file = os.path.basename(sys.argv[0]).split('.')[0]
    print(f'Save model function is being called from {parent_file}')
    text_file = parent_file + '.txt'
    model_file = parent_file + '.keras'
    # If this is the first time saving the model
    if model_file not in os.listdir() and text_file not in os.listdir():
        print('Creating intial files...')
        create_txt_file(text_file)
        model.save(model_file)
        print('Model is saved!')
    else:
        print('Model files already exist')
        with open(text_file, 'a+') as current_file:
            # Move cursor to beginning of file and then read
            current_file.seek(0)
            current_contents = current_file.readlines()
            # Get old metrics
            current_result_overall = float(current_contents[1].split(': ')[-1].strip())
            current_delay_mse = float(current_contents[2].split(': ')[-1].strip())
            current_length_mse = float(current_contents[3].split(': ')[-1].strip())
            current_delay_rmse = float(current_contents[4].split(': ')[-1].strip())
            current_length_rmse = float(current_contents[5].split(': ')[-1].strip())
            current_delay_mae = float(current_contents[6].split(': ')[-1].strip())
            current_length_mae = float(current_contents[7].split(': ')[-1].strip())
            current_delay_r2 = float(current_contents[8].split(': ')[-1].strip())
            current_length_r2 = float(current_contents[9].split(': ')[-1].strip())
            # Show model comparisons and see ask to save
            print('Current best vs. input model:')
            print(f'Overall loss: {[current_result_overall, result[0]]}')
            print(f'Delay MSE: {[current_delay_mse, result[2]]}')
            print(f'Length MSE: {[current_length_mse, result[5]]}')
            print(f'Delay RMSE: {[current_delay_rmse, result[3]]}')
            print(f'Length RMSE: {[current_length_rmse, result[6]]}')
            print(f'Delay MAE: {[current_delay_mse, result[1]]}')
            print(f'Length MAE: {[current_length_mse, result[4]]}')
            print(f'Delay R^2: {[current_delay_r2, r2_delay]}')
            print(f'Length R^2: {[current_length_r2, r2_length]}')
            save = input('Do you wish to save this model? (Y/N): ')
            while save != 'Y' and save != 'N':
                save = input('Please enter a valid answer (Y/N): ')
            if save == 'Y':
                create_txt_file(text_file)
                model.save(model_file, overwrite=True)
                print('Model is saved!')
            else:
                print('Not saving model, exiting...')
    return None

def get_model_info(model_file: str) -> None:
    model = keras.models.load_model(model_file)
    model.summary()
    text_file = model_file.split('.')[0] + '.txt'
    with open(text_file, 'r') as file:
        print(file.read())
    return None

            
ALERTS_COLUMNS = ['properties.city', 'properties.confidence',
       'properties.country', 'properties.location.x', 'properties.location.y',
       'properties.reliability', 'properties.reportDescription',
       'properties.reportRating', 'properties.roadType', 'properties.street',
       'properties.subtype', 'properties.type', 'properties.uuid',
       'properties.utc_timestamp', 'properties.day_of_week',
       'properties.weekday_weekend']

JAMS_COLUMNS = ['properties.city',
       'properties.country', 'properties.delay', 'properties.endNode',
       'properties.id', 'properties.length', 'properties.level',
       'properties.line', 'properties.roadType',
       'properties.segments', 'properties.speed', 'properties.speedKMH',
       'properties.street', 'properties.turnType', 'properties.type',
       'properties.uuid', 'properties.utc_timestamp', 'properties.day_of_week',
       'properties.weekday_weekend', 'geometry.coordinates']

DESIRED_COLUMN_INFO = {
    'properties.delay': 'jam delay in seconds',
    'properties.length': 'jam length in meters, will convert to miles for consistency',
    'properties.level': 'traffic congestion level, (0 = free flow, 5 = blocked)',
    'properties.speed': 'speed of jam traffic in m/s will convert to mph for consistency',
    'properties.street': 'DITCH, says which street/road name the jam is on, will extract if interstate number is even or odd',
    'properties.day_of_week': 'day of the week the jam occurs on, 0 - 6, 0 = monday',
    'properties.weekday_weekend': 'DITCH if it is a weekend = True, weekday  = False',
    'X': 'longitude coordinate of crash in degrees',
    'Y': 'latitude coordinate of crash in degrees',
    'MaxInjuryS': 'injury severity, A_ - C_ =  most to least severe, U_  = Unkown / only property damage, handle blanks as unkown',
    'CollisionT': 'type of collision, other can be handled as 0, text classes need full encoding',
    'CrashDate': 'SWITCH TO JUST MONTHS date of crash, M/D/YYYY format, split into multiple columns for month/day/year',
    'CrashTimeF': 'time of crash, 24 hr time, HH:MM, ignore date attatched to column, split into multiple columns for hr/min',
    'CrashType': 'what was hit / involved in crash, text classes need full encoding',
    'LightCondi': 'light condition at time of crash, text classes could use ordinal encoding',
    'RoadwayCha': 'tells if road has slope and/or curve, split into curve and grade columns, use one hot encoding',
    'RoadSurfac': 'tells condition of road, if snow or slush etc., use one hot encoding',
    'TrafficCon': 'tells if there is anything special about where the crash took place, (work area or signal), will need full encoding',
    'TrafficWay': 'says ways of road, if divided, and if median is protected, split and use one hot encoding',
    'WeatherCon': 'whether condition, use one hot encoding',
    'Commercial': 'whether crash was in a commerical zone, 0 or 1',
    'NumberOfFa': 'number of fatalities',
    'NumberOfIn': 'number of injuries',
    'NumberOfOt': 'number of other injuries',
    'NumberOfSe': 'nmber of serious injuries',
    'NumberOfVe': 'number of vehicles',
    'Intersecti': 'not, related, or at intersection, can use ordinal encoding',
    'CountyName': 'country name, would need full encoding',
    'CityTownNa': 'name of city/town, would need full encoding'
}

TEXT_COLUMNS = [
    'CollisionT',
    'CrashType',
    'TrafficCon',
    'CountyName',
    'CityTownNa'
]
TARGET_COLUMNS = [
    'properties.delay',
    'properties.length'
]

DITCH_COLUMNS = [
    'properties.weekday_weekend',
    'NumberOfFa',
    'day_sin',
    'day_cos',
    'year',
    'interstate_is_odd',
    'CityTownNa'
]
'''
DITCH day, year, number of fatality, even_odd, weekday_weekend
get rmse, and mae, map error/loss
'''


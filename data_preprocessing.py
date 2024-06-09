import pandas as pd
import json
import os, sys
from util import first_value_loc, DESIRED_COLUMN_INFO, cyclic_encode, one_hot_encode, TARGET_COLUMNS, TEXT_COLUMNS
import datetime as dt
import ast
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sb

INTERSTATE_NUMS = [95, 90, 86, 88, 81, 87, 390]


class primary_filter(Enum):
    LOCATION = 'location_based'
    DATETIME = 'datetime_based'
    PERFECTFIT = 'perfect_fit'
    THRESH = 'basic_thresh'


def combine_all_jams(interstate_num: int) -> None:
    """
    This function combines all JSON data from the 'waze_jams' directory
    into a single DataFrame and writes it to a CSV file.
    Only data with 'I-95 N' or 'I-95 S' street names is included.
    """
    interstates = [f'I-{str(interstate_num)} ', f'I-{str(interstate_num)} ']
    # Even interstate numbers go east and west and odd interstate numbers go north and south
    if interstate_num % 2 == 0:
        interstates[0] = interstates[0] + 'E'
        interstates[1] = interstates[1] + 'W'
    else:
        interstates[0] = interstates[0] + 'N'
        interstates[1] = interstates[1] + 'S'
    # Initialize an empty DataFrame
    df = pd.DataFrame()
    # Loop over the subdirectories in 'waze_jams'
    for i in os.listdir('waze_jams'):
        # Loop over the files in each subdirectory
        for j in os.listdir('waze_jams/' + i):
            print('waze_jams/' + i + '/' + j)
            # Open the JSON file
            with open('waze_jams/' + i + '/' + j) as json_file:
                # Load the JSON data
                data = json.load(json_file)
                # Normalize the JSON data
                data = pd.json_normalize(data['features'])
                # Filter data to include only 'I-95 N' or 'I-95 S' street names
                #data = data.loc[data['properties.street'].isin(street_names)]
                #data = data[data['properties.street'].apply(lambda x: any(x == street for street in street_names))]
                data = data.loc[(data['properties.street'] == interstates[0]) | (data['properties.street'] == interstates[1])]
                # Concatenate the current data to the DataFrame
                df = pd.concat([df, data], ignore_index=True)
    # The columns in the df had weird leading space for whatever reason
    df.rename(columns=lambda x: x.strip(), inplace=True)
    # Filter out all datapoints that arent in new york
    df['properties.city'] = df['properties.city'].apply(lambda x: x.strip() if isinstance(x, str) else x)
    df = df[df['properties.city'].str[-2:] == 'NY']
    # Write the DataFrame to a CSV file
    df.to_csv(f'waze_jams_I{str(interstate_num)}.csv')
    return None

def combine_all_jams_interstate() -> None:
    '''
    - Function for looping through the 'waze_jams' directory which contains all the json file data
    - Gets all the interstate data and outputs it to a CSV file
    - RUN THIS IF BASE INTERSTATE JAM DATA IS GONE
    '''
    # Initialize an empty DataFrame
    df = pd.DataFrame()
    # Loop over the subdirectories in 'waze_jams'
    for i in os.listdir('waze_jams'):
        # Loop over the files in each subdirectory
        for j in os.listdir('waze_jams/' + i):
            print('waze_jams/' + i + '/' + j)
            # Open the JSON file
            with open('waze_jams/' + i + '/' + j) as json_file:
                # Load the JSON data
                data = json.load(json_file)
                # Normalize the JSON data
                data = pd.json_normalize(data['features'])
                # Filter data to include only 'I-##' or 'INTERSTATE ##' street names
                data = data.loc[(data['properties.street'].str.startswith('I-')) & (data['properties.street'].str[2:-2].str.isdigit())]
                # Concatenate the current data to the DataFrame
                df = pd.concat([df, data], ignore_index=True)
    # The columns in the df had weird leading space for whatever reason
    df.rename(columns=lambda x: x.strip(), inplace=True)
    # Filter out all datapoints that arent in new york
    df['properties.city'] = df['properties.city'].apply(lambda x: x.strip() if isinstance(x, str) else x)
    df = df[df['properties.city'].str[-2:] == 'NY']
    # Write the DataFrame to a CSV file
    df.to_csv(f'waze_jams_interstates.csv')
    return None

def generate_all_jams() -> None:
    '''
    - This was for when I was going to just look at some of the major intersections and get the jams files for them
    - DEPRECTATED, we are looking at all interstates
    '''
    for i in INTERSTATE_NUMS:
        combine_all_jams(i)
    return None

def combine_all_alerts() -> None:
    """
    - This function combines all JSON data from the 'waze_jams' directory into a single DataFrame and writes it to a CSV file.
    - Only data with 'I-95 N' or 'I-95 S' street names is included.
    - DEPRECATED, no longer using alerts data since connecting uuid field is not in dataset
    """
    # Initialize an empty DataFrame
    df = pd.DataFrame()
    # Loop over the subdirectories in 'waze_alerts'
    for i in os.listdir('waze_alerts'):
        # Loop over the files in each subdirectory
        for j in os.listdir('waze_alerts/' + i):
            print('waze_alerts/' + i + '/' + j)
            # Open the JSON file
            with open('waze_alerts/' + i + '/' + j, encoding='utf-8') as json_file:
                # Load the JSON data
                data = json.load(json_file)
                # Normalize the JSON data
                data = pd.json_normalize(data['features'])
                # Filter data to include only 'I-95 N' or 'I-95 S' street names
                data = data.loc[(data['properties.street'] == 'I-95 N') | (data['properties.street'] == 'I-95 S')]
                # Concatenate the current data to the DataFrame
                df = pd.concat([df, data], ignore_index=True)
    # Write the DataFrame to a CSV file
    df.to_csv('waze_alerts_I95_total.csv')
    return None

def see_all_jams() -> None:
    """
    - This function counts the occurrence of different street names in the 'waze_jams' directory.
    - The output is written to 'counts.txt'.
    - No longer needed since this was for the very beginning when we thought we would have enough data from 1 interstate
    - DEPRECATED
    """
    # Dictionary to store the counts of different street names
    counts = {}
    # Loop over the subdirectories in 'waze_jams'
    for i in os.listdir('waze_jams'):
        # Loop over the files in each subdirectory
        for j in os.listdir('waze_jams/' + i):
            print('waze_jams/' + i + '/' + j)
            try:
                # Open the JSON file
                with open('waze_jams/' + i + '/' + j) as json_file:
                    # Load the JSON data
                    data = json.load(json_file)
                    # Normalize the JSON data
                    data = pd.json_normalize(data['features'])
                    # Get the count of different street names
                    data = data['properties.street'].value_counts().to_dict()
                    # Update the counts dictionary
                    for i in data.keys():
                        counts[i] = counts.get(i, 0) + data[i]
            except FileNotFoundError:
                print('Error: File not found')
            finally:
                # Continue to the next file
                continue
    
    # Write the counts to a file
    with open('counts.txt', 'w') as f:
        sys.stdout = f
        # Loop over the keys in counts dictionary
        for i in counts.keys():
            # Print each key and its count
            print(i + ': ' + str(counts[i]))
    return None

def filter_I95() -> None:
    '''
    - Filters out all the points in the I95 data that isnt in NYS (single use function for when we were looking at singel intersection)
    - DEPRECATED
    '''
    df = pd.read_csv('waze_jams_I95_total.csv')
    # The columns in the df had weird leading space for whatever reason
    df.rename(columns=lambda x: x.strip(), inplace=True)
    # Filter out all datapoints that arent in new york
    df['properties.city'] = df['properties.city'].apply(lambda x: x.strip())
    df = df[df['properties.city'].str[-2:] == 'NY']
    df.to_csv('waze_jams_I95.csv')
    return None

def clean_crash(crash_file: str, *on_street: str) -> None:
    '''
    - Gets rid of some weird spaces that werte added to column names (prolly due to vscode extension)
    - Filters out data that is not on the street names, (was designed for when we were looking at per interstate basis)
    - DEPRECATED
    '''
    df = pd.read_csv(crash_file)
    # The columns in the df had weird leading space for whatever reason
    df.rename(columns=lambda x: x.strip(), inplace=True)
    # Filter out all datapoints not on the desired interstate
    df = df[df['OnStreet'].isin(on_street)]
    df.to_csv(crash_file)
    return None

def combine_all_crashes(crash_file_dir: str) -> None:
    '''
    - Designed to combine all the csv crash files for the different counties into 1 file
    - A one time use function for the 'NYS_Crash_CSVs' folder which has the csv files that have been converted FROM shp files
    - RUN THIS IF ALL INTERSTATE CRASH DATA IS GONE 'combine_all_crashes('NYS_Crash_CSVs')'
    '''
    df = None
    for i in os.listdir(crash_file_dir):
        temp = pd.read_csv(f"{crash_file_dir}/{i}")
        df = pd.concat([df, temp], ignore_index=True)
        df.drop_duplicates(inplace=True)
    print(len(df))
    df.to_csv('interstate_crashes.csv')
    return None

def validate_data(jams_file: str, crashes_file: str, filter: primary_filter, test: bool = False) -> None:
    '''
    - Input a jam file and a crash file that has been fed throught the shp to csv website because it changes some of the column names
    - DISTANCE and LOCATION filters are depricated and were for initial testing
    - PERFECTFIT filter finds all the points where the each jam has 1 crash that is the closest AND the soonest,
        we did not use this method due to the fact that multiple jam alerts can be generated from one crash which
        this method loses out on
    - THRESH filter just uses the thresholds so it preserves the multiple jam alerts, this is what we used in our data
    - thresholds are set within the function, we are going with 1/2hr and 2mi
    - test parameter is just to change output file name for testing so that nothing important is overwritten
    - RUN THIS IF VALIDATED DATA IS GONE 'validate_data('waze_jams_interstates.csv', 'interstate_crashes.csv', primary_filter.THRESH)'
    '''
    jams_date_time_col = 'properties.utc_timestamp'
    crashes_street_col = 'OnStreet'
    crashes_date_col = 'CrashDate'
    crashes_time_col = 'CrashTimeF'

    # Difference in time threshold
    time_threshold = .5 # [hours]

    # Distance threshold
    dist_threshold = 2 # [miles]
    lat_to_miles = 69
    lon_to_miles = 54.6


    # Open up both csv files
    df_jams = pd.read_csv(jams_file)
    # Clean up weird trailing spaces, idk why they keep popping up
    df_jams.columns = df_jams.columns.str.strip()
    df_crashes = pd.read_csv(crashes_file)
    #df_crashes = df_crashes[(df_crashes[crashes_street_col] == 'I 95') | (df_crashes[crashes_street_col] == 'Interstate 95')]

    # Convert both time/date rows into datetime objects
    df_jams['key_date_jams'] = pd.to_datetime(df_jams[jams_date_time_col], format='ISO8601')
    df_jams['key_date_jams'] = df_jams['key_date_jams'].dt.date
    df_crashes['key_date_crashes'] = pd.to_datetime(df_crashes[crashes_date_col], format='ISO8601').dt.date

    # Use cartesian product to match each jams with a crash that happened that day
    df_merged = None
    for key_date in df_jams['key_date_jams'].unique():
        print(f'key_date: {key_date}')
        jams_group = df_jams[df_jams['key_date_jams'] == key_date]
        #print(f'Jams group: {len(jams_group)}')
        crashes_group = df_crashes[df_crashes['key_date_crashes'] == key_date]
        #print(f'Crashes group: {len(crashes_group)}')
        cartesian_product = pd.merge(jams_group, crashes_group, how='cross')
        #print(f'Cartesian product: {len(cartesian_product)}')
        df_merged = pd.concat([df_merged, cartesian_product], ignore_index=True)


    #Ouput for testing
    print(f'{len(df_jams)} unique jams with {len(df_crashes)} crashes totalling {len(df_merged)} possiblities')

    if filter == primary_filter.DATETIME:
        # Group by each identical jam
        # Each grouping of jam will have the same time but different crash data times
        # Within each group only select the crash data that has the smallest time delta between the jams and crash
        df_merged[jams_date_time_col] = pd.to_datetime(df_merged[jams_date_time_col], format='ISO8601')
        df_merged[crashes_time_col] = pd.to_datetime(df_merged[crashes_time_col], format='%I:%M %p')
        #df_merged['time_delta']
        df_merged = df_merged[(df_merged[jams_date_time_col].dt.hour - df_merged[crashes_time_col].dt.hour) <= time_threshold]
        df_merged = df_merged.groupby(by='properties.uuid', as_index=False).apply(
            lambda group: group.loc[
                ((group[jams_date_time_col].dt.hour - group[crashes_time_col].dt.hour).multiply(60) + 
                (group[jams_date_time_col].dt.minute - group[crashes_time_col].dt.minute)
                ).abs().idxmin()
            ]
        )
    elif filter == primary_filter.LOCATION:
        # Group by each identical jam
        # Each grouping of jam will have the same location but different crash locations
        # Within each group only select the crash data that has the smallest distance to the first jam coord
        # Change coordinate data to float since pandas reads them as strings by default
        df_merged['geometry.coordinates'] = df_merged['geometry.coordinates'].apply(ast.literal_eval)
        df_merged['jams_x'] = df_merged['geometry.coordinates'].apply(lambda x: first_value_loc(x, 0))
        df_merged['jams_y'] = df_merged['geometry.coordinates'].apply(lambda x: first_value_loc(x, 1))
        df_merged = df_merged[(
            (((df_merged['jams_x'] * lon_to_miles) - (df_merged['X'] * lon_to_miles))**2
                ) + 
            (((df_merged['jams_y'] * lat_to_miles) - (df_merged['Y'] * lat_to_miles))**2
                )**.5) <= dist_threshold]
        df_merged = df_merged.groupby(by='properties.uuid', as_index=False).apply(
            lambda group: group.loc[
                (((group['jams_x'] - group['X'])**2) + ((group['jams_y'] - group['Y'])**2)**.5
                ).abs().idxmin()
            ]
        )
    elif filter == primary_filter.PERFECTFIT:
        #Time threshold
        df_merged[jams_date_time_col] = pd.to_datetime(df_merged[jams_date_time_col], format='ISO8601')
        df_merged[crashes_time_col] = pd.to_datetime(df_merged[crashes_time_col], format='%I:%M %p')
        df_merged = df_merged[
            abs(
                ((df_merged[jams_date_time_col].dt.hour * 60) + (df_merged[jams_date_time_col].dt.minute)) - 
                ((df_merged[crashes_time_col].dt.hour * 60) + (df_merged[crashes_time_col].dt.minute))
            ) <= (time_threshold * 60)]
        print(f'after time thresholding: {len(df_merged)}')
        #Distance threshold
        df_merged['geometry.coordinates'] = df_merged['geometry.coordinates'].apply(ast.literal_eval)
        df_merged['jams_x'] = df_merged['geometry.coordinates'].apply(lambda x: first_value_loc(x, 0))
        df_merged['jams_y'] = df_merged['geometry.coordinates'].apply(lambda x: first_value_loc(x, 1))
        df_merged = df_merged[(
            ((((df_merged['jams_x'] * lon_to_miles) - (df_merged['X'] * lon_to_miles))**2
                ) + 
            (((df_merged['jams_y'] * lat_to_miles) - (df_merged['Y'] * lat_to_miles))**2
                ))**.5) <= dist_threshold]
        print(f'after distance thresholding: {len(df_merged)}')
        # Group by each identical jam
        # Each grouping of jam will have the same time but different crash data times
        # Within each group only select the crash data that has the smallest time delta between the jams and crash       
        df_merged_time = df_merged.groupby(by='properties.uuid', as_index=False).apply(
            lambda group: group.loc[
                ((group[jams_date_time_col] - group[crashes_time_col])
                ).abs().idxmin()
            ]
        )
        # Group by each identical jam
        # Each grouping of jam will have the same location but different crash locations
        # Within each group only select the crash data that has the smallest distance to the first jam coord
        # Change coordinate data to float since pandas reads them as strings by default
        df_merged_loc = df_merged.groupby(by='properties.uuid', as_index=False).apply(
            lambda group: group.loc[
                (((group['jams_x'] - group['X'])**2) + ((group['jams_y'] - group['Y'])**2)**.5
                ).abs().idxmin()
            ]
        )
        df_merged = pd.merge(df_merged_time, df_merged_loc, how='inner', on=['properties.uuid', 'CaseNumber'], suffixes=('', 'XXX'))
        df_merged.drop(columns=[col for col in df_merged.columns if col.endswith('XXX')], inplace=True)
    elif filter == primary_filter.THRESH:
        # Time threshold
        df_merged[jams_date_time_col] = pd.to_datetime(df_merged[jams_date_time_col], format='ISO8601')
        df_merged[crashes_time_col] = pd.to_datetime(df_merged[crashes_time_col], format='%I:%M %p')
        df_merged = df_merged[
            abs(
                ((df_merged[jams_date_time_col].dt.hour * 60) + (df_merged[jams_date_time_col].dt.minute)) - 
                ((df_merged[crashes_time_col].dt.hour * 60) + (df_merged[crashes_time_col].dt.minute))
            ) <= (time_threshold * 60)]
        print(f'after time thresholding: {len(df_merged)}')
        # Distance threshold
        df_merged['geometry.coordinates'] = df_merged['geometry.coordinates'].apply(ast.literal_eval)
        df_merged['jams_x'] = df_merged['geometry.coordinates'].apply(lambda x: first_value_loc(x, 0))
        df_merged['jams_y'] = df_merged['geometry.coordinates'].apply(lambda x: first_value_loc(x, 1))
        df_merged = df_merged[(
            ((((df_merged['jams_x'] * lon_to_miles) - (df_merged['X'] * lon_to_miles))**2
                ) + 
            (((df_merged['jams_y'] * lat_to_miles) - (df_merged['Y'] * lat_to_miles))**2
                ))**.5) <= dist_threshold]
        print(f'after distance thresholding: {len(df_merged)}')

    else:
        print('Unrecognized filter')
        return None
    # Clean out some extra unwanted columns, delete duplicates, reset indexes, output to csv
    df_merged.drop(columns=[col for col in df_merged.columns if 'Unnamed' in col], inplace=True)
    columns_to_check = [col for col in df_merged.columns if not isinstance(df_merged[col].iloc[0], (list, dict, tuple))]
    df_merged.drop_duplicates(subset=columns_to_check, inplace=True)
    df_merged.reset_index(drop=True, inplace=True)
    if not test:
        df_merged.to_csv(f'{filter.value}_validated_interstates.csv')
    else:
        df_merged.to_csv('TEST.csv')

def preprocess_validated_data(validated_data_file: str, test: bool = False) -> None:
    '''
    - Input the file that has the validated crash and jam data
    - Will filter out unwanted columns, encode text based data, and prepare data to be used for ML algorithm
    - Has lots of troubleshooting output to make sure everything is working as intended, also doubles as pseudo documentation
    '''
    df = pd.read_csv('basic_thresh_validated_interstates.csv')
    # Get only the columns we want (with troubeshooting output)
    print(f'df originally has {len(df.columns)} columns')
    df.drop(columns=[col for col in df.columns if col not in DESIRED_COLUMN_INFO.keys()], inplace=True)
    print(f'df should have {len(DESIRED_COLUMN_INFO.keys())} columns, df has {len(df.columns)} columns')
    print(f'columns that it did not find: {[col for col in DESIRED_COLUMN_INFO.keys() if col not in df.columns]}')

    # Convert the jam length from meters to miles
    meters_to_miles =  0.000621371
    df['properties.length'] = df['properties.length'] * meters_to_miles
    # Convert jam length from seconds to minutes
    df['properties.delay'] = df['properties.delay'] / 60
    # Convert speed of jam traffic from m/s to mph
    mps_to_mph = 2.23694
    df['properties.speed'] = df['properties.speed'] * mps_to_mph

    # Extract if the interstate number is even or odd
    df['interstate_is_odd'] = df['properties.street'].apply(lambda x: 1 if (int(x[2:-2]) % 2 == 1) else 0)
    print(df[['properties.street', 'interstate_is_odd']].head())
    df.drop(columns='properties.street', inplace=True)

    # Convert weekday_weekend column from T/F to 1/0
    print(f'weekday_weekend originally \n{df["properties.weekday_weekend"].head()}\n with values {df["properties.weekday_weekend"].unique()}')
    df['properties.weekday_weekend'] = df['properties.weekday_weekend'].astype(dtype=bool).astype(dtype=int)
    print(f'weekday_weekend after conversion \n{df["properties.weekday_weekend"].head()}\n with values {df["properties.weekday_weekend"].unique()}')

    # Use ordinal encoding for max injury
    injury_encoding = {
        'A': 3,
        'B': 2,
        'C': 1,
        'U': 0,
    }
    print(f'MaxInjuryS originally \n{df["MaxInjuryS"].head()}\n with values {df["MaxInjuryS"].unique()}')
    df['MaxInjuryS'] = df['MaxInjuryS'].apply(lambda x: injury_encoding[x[0]] if not pd.isna(x) else 0)
    print(f'MaxInjuryS originally after encoding \n{df["MaxInjuryS"].head()}\n with values {df["MaxInjuryS"].unique()}')

    # Split the crash date into year/month/day columns, apply cyclic encoding to month and day
    print(f'CrashDate originally \n{df["CrashDate"].head()}')
    df[['year', 'month', 'day']] = df['CrashDate'].str.split('/', expand=True).astype(int)
    print(f'Seprated into columns \n{df[["year", "month", "day"]].head()}')
    df = cyclic_encode(df, 'day', df['day'].max())
    df = cyclic_encode(df, 'month', df['month'].max())
    df.drop(columns='CrashDate', inplace=True)
    print(f'Dates with circular encoding \n{df[["year", "month_sin", "month_cos", "day_sin", "day_cos"]].head()}')


    # Split the crash time into hrs and minutes and apply cyclic encoding
    print(f'CrashTimeF originally \n{df["CrashTimeF"].head()}')
    df['CrashTimeF'] = pd.to_datetime(df['CrashTimeF'], format='ISO8601')
    df['hour'] = df['CrashTimeF'].dt.hour
    df['minute'] = df['CrashTimeF'].dt.minute
    print(f'CrashTimeF after splitting \n{df[["hour", "minute"]].head()}')
    df = cyclic_encode(df, 'hour', df['hour'].max())
    df = cyclic_encode(df, 'minute', df['minute'].max())
    df.drop(columns='CrashTimeF', inplace=True)
    print(f'Times with circular encoding \n{df[["hour_sin", "hour_cos", "minute_sin", "minute_cos"]].head()}')

    # Apply ordinal encoding to the light condition column
    print(f'LightCondi originally \n{df["LightCondi"].head()}\nwith values {df["LightCondi"].unique()}')
    light_encoding = {
        'DAYLIGHT': 0,
        'UNKNOWN': 0,
        'DUSK': 1,
        'DAWN': 1,
        'DARK-ROAD LIGHTED': 2,
        'DARK-ROAD UNLIGHTED': 3
    }
    df['LightCondi'] = df['LightCondi'].apply(lambda x: light_encoding[x] if not pd.isna(x) else 0)
    print(f'LightCondi after encoding \n{df["LightCondi"].head()}\nwith values {df["LightCondi"].unique()}')

    # Apply one hot encoding to RoadwayCha
    print(f'RoadwayCha before originally \n{df["RoadwayCha"].head()}\nwith values {df["RoadwayCha"].unique()}')
    df['curve'] = df['RoadwayCha'].str.split(expand=True)[0]
    df['grade'] = df['RoadwayCha'].str.split(expand=True)[2]
    df['curve'] = df['curve'].apply(lambda x: 1 if (not pd.isna(x)) and ('CURVE' in x) else 0)
    df['grade'] = df['grade'].apply(lambda x: 1 if (not pd.isna(x)) and ('GRADE' in x) else 0)
    df.drop(columns='RoadwayCha', inplace=True)
    print(f'Curve and grade one hot encoding \n{df[["curve", "grade"]].head()}\nwith values {df["curve"].unique()} and {df["grade"].unique()}')

    # Apply one hot encoding to RoadSurfac
    print(f'RoadSurfac orginially \n{df["RoadSurfac"].head()}\n with values {df["RoadSurfac"].unique()}')
    df = one_hot_encode(df, 'RoadSurfac')
    print(f'After one hot encoding\n{df[["WET", "DRY", "UNKNOWN", "FLOODED WATER", "SNOW/ICE", "SLUSH"]].head()}')

    # Apply one hot encoding to TrafficWay have to split myself because data is condensed
    print(f'TrafficWay originallly \n{df["TrafficWay"].head()}\n with value {df["TrafficWay"].unique()}')
    df['two_way'] = df['TrafficWay'].str.split(',', expand=True)[0]
    df['divided'] = df['TrafficWay'].str.split(',', expand=True)[1]
    df['protected_medium'] = df['TrafficWay'].str.split(',', expand=True)[2]
    df['two_way'] = df['two_way'].apply(lambda x: 1 if (not pd.isna(x)) and ('TWO' in x) else 0)
    df['divided'] = df['divided'].apply(lambda x: 1 if (not pd.isna(x)) and (not 'NOT' in x) else 0)
    df['protected_medium'] = df['protected_medium'].apply(lambda x: 1 if (not pd.isna(x)) and ('POSITIVE' in x) else 0)
    df.drop(columns='TrafficWay', inplace=True)
    print(f'After one hot encoding\n{df[["two_way", "divided", "protected_medium"]].head()}')

    # Apply one hot encoding to weather condition
    print(f'WeatherCon originally \n{df["WeatherCon"].head()}\nwith values {df["WeatherCon"].unique()}')
    df['WeatherCon'] = df['WeatherCon'].str.replace('\r\n', '')
    df = one_hot_encode(df, 'WeatherCon')
    print(f'After one hot encoding')
    print(df[["CLOUDY", "CLEAR", "UNKNOWN", "RAIN", "SLEET/HAIL/FREEZING RAIN", "SNOW"]].head())

    # Apply ordinal encoding to Intersecti
    print(f'Intersection originally\n{df["Intersecti"].head()}\nwith values {df["Intersecti"].unique()}')
    intersection_encoding = {
        'INTERSECTION-RELATED': 1,
        'Not an intersection crash': 0,
        'AT-INTERSECTION': 2
    }
    df['Intersecti'] = df['Intersecti'].apply(lambda x: intersection_encoding[x] if not pd.isna(x) else 0)
    print(f'After encoding\n{df["Intersecti"].head()}\nwith values {df["Intersecti"].unique()}')

    #Fix some of the text data
    df['CollisionT'] = df['CollisionT'].str.replace(r'[()]', '', regex=True).str.lower()
    df['CrashType'] = df['CrashType'].str.replace('COLL.', 'COLLISION')
    df['CrashType'] = df['CrashType'].str.replace('W/', 'WITH ')
    df['CrashType'] = df['CrashType'].str.replace('ELE.', 'ELEMENT')
    df['CrashType'] = df['CrashType'].str.replace(r'/|\s-\s|-', ' ', regex=True).str.lower()
    df['TrafficCon'] = df['TrafficCon'].str.replace('/', ' ').str.lower()
    df['CountyName'] = df['CountyName'].str.lower()
    df['CityTownNa'] = df['CityTownNa'].str.lower()
    
    if not test:
        df.to_csv('encoded_model_data.csv')
    else:
        df.to_csv('encoded_TEST.csv')
    return None

def see_correlation(correlation_thresh: float = 0.4, drop_thresh: int = 0) -> list[str]:
    '''
    - A function that goes through the encoded data and displays the pearson coefficients of the independent variables,
    it returns a list of the columns to drop to reduce correlation
    - correlation_thresh: minimum value of variable correlation to flag as high correlation
    - drop_thresh: the number of the allowable high correlation variable combinations within a column
    '''
    # Make the heatmap for the pearson correlation
    df = pd.read_csv('encoded_model_data.csv').drop(columns=TARGET_COLUMNS).drop(columns=TEXT_COLUMNS).drop(columns='Unnamed: 0')
    corr = df.corr(method='pearson')
    plot = sb.heatmap(corr, cmap='RdBu', xticklabels=df.columns, yticklabels=df.columns)
    plot.set_title('Pearson Correlation Between Independent Variables')
    plot.set_xlabel('Variable')
    plot.set_ylabel('Variable')
    plt.show()
    # See which variables have the highest level of correlation
    bad_cols = []
    while True:
        corr = df.corr(method='pearson')
        bad_vars = corr.map(lambda x: ((abs(x) >= correlation_thresh) & (abs(x) < 1))).sum()
        max = bad_vars.max()
        max_col = bad_vars.idxmax()  
        print(f'Max: {max_col} - {max}')
        if max <= drop_thresh:
            break
        bad_cols.append(max_col)
        df.drop(columns=max_col, inplace=True)
    print(f'The columns to reduce correlation below thresh: {bad_cols}')
    return bad_cols

def main():
    #combine_all_crashes('NYS_Crash_CSVs')
    #validate_data('waze_jams_interstates.csv', 'interstate_crashes.csv', primary_filter.THRESH)
    #preprocess_validated_data('basic_thresh_validated_interstates.csv')
    bad_cols = see_correlation(drop_thresh=1)
    df = pd.read_csv('encoded_model_data.csv').drop(columns=bad_cols)
    df.to_csv('correlation_free.csv')
if __name__ == '__main__':
    main()
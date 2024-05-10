import pandas as pd
import json
import os, sys
from util import first_value_loc
import datetime as dt
import ast
from enum import Enum

INTERSTATE_NUMS = [95, 90, 86, 88, 81, 87, 390]


class primary_filter(Enum):
    LOCATION = 'location_based'
    DATETIME = 'datetime_based'
    PERFECTFIT = 'perfect_fit'
    THRESH = 'basic_thresh'


def combine_all_jams(interstate_num: int):
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

def combine_all_jams_interstate():
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

def generate_all_jams():
    '''
    - This was for when I was going to just look at some of the major intersections and get the jams files for them
    - DEPRECTATED, we are looking at all interstates
    '''
    for i in INTERSTATE_NUMS:
        combine_all_jams(i)

def combine_all_alerts():
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

def see_all_jams():
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

def filter_I95():
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

def clean_crash(crash_file: str, *on_street: str):
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

def combine_all_crashes(crash_file_dir: str):
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
        return
    # Clean out some extra unwanted columns, delete duplicates, reset indexes, output to csv
    df_merged.drop(columns=[col for col in df_merged.columns if 'Unnamed' in col], inplace=True)
    columns_to_check = [col for col in df_merged.columns if not isinstance(df_merged[col].iloc[0], (list, dict, tuple))]
    df_merged.drop_duplicates(subset=columns_to_check, inplace=True)
    df_merged.reset_index(drop=True, inplace=True)
    if not test:
        df_merged.to_csv(f'{filter.value}_validated_interstates.csv')
    else:
        df_merged.to_csv('TEST.csv')

def preprocess_validated_data(validated_data_file: str):
    '''
    - Input the file that has the validated crash and jam data
    - Will filter out unwanted columns, encode text based data, and prepare data to be used for ML algorithm
    '''
    COLUMNS = [
        'properties.delay', 'properties.length', 'properties.level',

    ]


def main():
    #combine_all_crashes('NYS_Crash_CSVs')
    validate_data('waze_jams_interstates.csv', 'interstate_crashes.csv', primary_filter.THRESH)
if __name__ == '__main__':
    main()
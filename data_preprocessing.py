import pandas as pd
import json
import os, sys
from data_visualization import first_value_loc
import datetime as dt
import ast
 
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

def combine_all_jams():
    """
    This function combines all JSON data from the 'waze_jams' directory
    into a single DataFrame and writes it to a CSV file.
    Only data with 'I-95 N' or 'I-95 S' street names is included.
    """
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
                data = data.loc[(data['properties.street'] == 'I-95 N') | (data['properties.street'] == 'I-95 S')]
                # Concatenate the current data to the DataFrame
                df = pd.concat([df, data], ignore_index=True)
    # Write the DataFrame to a CSV file
    df.to_csv('waze_jams_I95_total.csv')

def combine_all_alerts():
    """
    This function combines all JSON data from the 'waze_jams' directory
    into a single DataFrame and writes it to a CSV file.
    Only data with 'I-95 N' or 'I-95 S' street names is included.
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
    This function counts the occurrence of different street names in the 'waze_jams' directory.
    The output is written to 'counts.txt'.
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
    Filters out all the points in the I95 data that isnt in NYS
    '''
    df = pd.read_csv('waze_jams_I95_total.csv')
    # The columns in the df had weird leading space for whatever reason
    df.rename(columns=lambda x: x.strip(), inplace=True)
    # Filter out all datapoints that arent in new york
    df['properties.city'] = df['properties.city'].apply(lambda x: x.strip())
    df = df[df['properties.city'].str[-2:] == 'NY']
    df.to_csv('waze_jams_I95.csv')

def validate_data():
    '''
    Match up the I95 jams data to the I95 crash data
    '''
    # Open up both csv files
    df_jams = pd.read_csv('waze_jams_I95_slice.csv')
    # Clean up weird trailing spaces, idk why they keep popping up
    df_jams.columns = df_jams.columns.str.strip()
    df_crashes = pd.read_csv('I95_crashes.csv', header=1)

    # Convert both time/date rows into datetime objects
    df_jams['key_date'] = pd.to_datetime(df_jams['properties.utc_timestamp'], format='ISO8601')
    df_jams['key_date'] = df_jams['key_date'].dt.date
    df_crashes['key_date'] = pd.to_datetime(df_crashes['Crash Date'], format='ISO8601').dt.date

    # Use cartesian product to match each jams with a crash that happened that day
    df_merged = pd.merge(df_jams, df_crashes, on='key_date', suffixes=('_jams', '_crashes'))
    #df_merged.to_csv('TEST.csv')

    # Group by each identical jam
    # Each grouping of jam will have the same time but different crash data times
    # Within each group only select the crash data that has the smallest time delta between the jams and crash
    df_merged['properties.utc_timestamp'] = pd.to_datetime(df_merged['properties.utc_timestamp'], format='ISO8601')
    df_merged['Crash Time'] = pd.to_datetime(df_merged['Crash Time'], format='%I:%M %p')
    df_merged = df_merged.groupby(by='properties.uuid').apply(
        lambda group: group.loc[
            ((group['properties.utc_timestamp'].dt.hour - group['Crash Time'].dt.hour).multiply(60) + 
             (group['properties.utc_timestamp'].dt.minute - group['Crash Time'].dt.minute)
            ).abs().idxmin()
        ]
    )

    df_merged.to_csv('TEST2.csv')
    


def main():
    validate_data()
if __name__ == '__main__':
    main()
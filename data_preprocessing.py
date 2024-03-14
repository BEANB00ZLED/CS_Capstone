import pandas as pd
import json
import os, sys
 
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

def combine_all():
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
    df.to_csv('waze_jams_I90_total')

def see_all():
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




def main():
   pass

if __name__ == '__main__':
    main()
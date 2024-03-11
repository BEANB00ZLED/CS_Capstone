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
    df = pd.DataFrame()
    for i in os.listdir('waze_jams'):
        for j in os.listdir('waze_jams/' + i):
            print('waze_jams/' + i + '/' + j)
            with open('waze_jams/' + i + '/' + j) as json_file:
                data = json.load(json_file)
                data = pd.json_normalize(data['features'])
                data = data.loc[(data['properties.street'] == 'I-95 N') | (data['properties.street'] == 'I-95 S')]
                df = pd.concat([df, data], ignore_index=True)
    df.to_csv('waze_jams_I90_total')

def see_all():
    counts = {}
    for i in os.listdir('waze_jams'):
        for j in os.listdir('waze_jams/' + i):
            print('waze_jams/' + i + '/' + j)
            try:
                with open('waze_jams/' + i + '/' + j) as json_file:
                    data = json.load(json_file)
                    data = pd.json_normalize(data['features'])
                    data = data['properties.street'].value_counts().to_dict()
                    for i in data.keys():
                        counts[i] = counts.get(i, 0) + data[i]
            except FileNotFoundError:
                print('shit broke again')
            finally:
                continue
    with open('counts.txt', 'w') as f:
        sys.stdout = f
        for i in counts.keys():
                print(i + ': ' + str(counts[i]))




def main():
   combine_all()
   ''' with open('waze_alerts_2022_06\waze_alerts_2022_06.json') as alerts_file:
        alerts_data = json.load(alerts_file)

    df_alerts = pd.json_normalize(alerts_data['features'])
    df_alerts.to_csv('waze_alerts_2022_06.csv')
    print(df_alerts.columns)

    with open('waze_jams_2022_06\waze_jams_2022_06.json') as jams_file:
        jams_data = json.load(jams_file)

    df_jams = pd.json_normalize(jams_data['features'])
    df_jams.to_csv('waze_jams_2022_06.csv')
    print(df_jams.columns)
   df = pd.read_csv('waze_alerts_2022_06.csv')
   df.iloc[0:1000].to_csv('alerts.csv')
   df = pd.read_csv('waze_jams_2022_06.csv')
   df.iloc[0:1000].to_csv('jams.csv')'''

   '''df_alerts = pd.read_csv('waze_alerts_2022_06.csv')
   df_jams = pd.read_csv('waze_jams_2022_06.csv')
   df_overlap = df_alerts['properties.uuid'].isin(df_jams['properties.blockingAlertUuid'])
   df_combined = df_alerts.where(df_overlap).merge(df_jams.where(df_overlap), left_index=True, right_index=True)
   df_combined = df_combined.dropna(how='all', axis=0)
   df_combined.to_csv('waze_combined_2022_06.csv', index=False)'''

   '''df = pd.read_csv('waze_jams_2022_06.csv')
   df = df.loc[(df['properties.street'] == 'I-90 E') | (df['properties.street'] == 'I-90 W')]
   df.to_csv('waze_jams_2022_06_I90.csv')'''





if __name__ == '__main__':
    main()
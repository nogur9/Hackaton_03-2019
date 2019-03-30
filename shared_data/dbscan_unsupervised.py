
import pandas as pd
import numpy as np
import math

from sklearn.cluster import DBSCAN

min_timestamp_cutoff = 30

pd.options.display.max_columns =200
pd.options.display.max_rows =500
pd.options.display.max_colwidth =500


train = pd.read_csv('train.csv', encoding='windows-1255')
train.drop_duplicates(subset=['trip_index']).route_id.value_counts()
unique_samples_of_trip_index = train.groupby('trip_index').time_recorded.count()

remove_trip_index = unique_samples_of_trip_index.index[unique_samples_of_trip_index.values <min_timestamp_cutoff]

remove_trip_index = list(remove_trip_index)

train = train[train.trip_index.apply(lambda x: x not in remove_trip_index)]

dots = train[['lon', 'lat']]
labels = train['route_id']

X = dots[labels==5189]
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)


print("num clusters = ", n_clusters_, "n_noise = ", n_noise_)
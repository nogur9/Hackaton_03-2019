

import pandas as pd
import numpy as np
import math

from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import geopandas as gpd



# follow the steps
def find_route_id_similarity(trip, train_coordinates_gdf):
    similarity_summary = {}

    for trip_index in train_coordinates_gdf.trip_index:

        trip_index_buffers = train_coordinates_gdf[train_coordinates_gdf['trip_index'] == trip_index]['buffer']

        trip_index_coordinates_results = {key: trip.within(geom) for key, geom in trip_index_buffers.items()}

        trip_index_coordinates_summary = {}
        for coordinates in trip_index_coordinates_results.keys():
            trip_index_coordinates_summary[coordinates] = np.any(trip_index_coordinates_results[coordinates])

        route_id_mean = np.mean(list(trip_index_coordinates_summary.values()))

        route_id_count = len(trip_index_coordinates_summary)

        similarity_summary[str(trip_index) + '_mean'] = [route_id_mean]

        similarity_summary[str(trip_index) + '_cnt'] = [route_id_count]
    similarity_summary = pd.DataFrame(similarity_summary)
    similarity_summary.to_csv('trip_index_coordinates_summary.csv')
    return similarity_summary


def set_buffer(train, buffer_rad=0.01):
    train['coordinates'] = train['coordinates'].apply(Point)
    gdf = gpd.GeoDataFrame(train, geometry='coordinates')
    gdf['buffer'] = gdf.buffer(buffer_rad)
    gdf.set_index('stop_id', inplace=True, drop=False)

    return gdf


# step 1 - create route stats unique

train = pd.read_csv('train.csv', encoding='windows-1255')
train.drop_duplicates(subset=['trip_index']).route_id.value_counts()


train['coordinates'] = list(zip(train.lat, train.lon))
train['coordinates'] = train['coordinates'].apply(Point)

train_coordinates_gdf = set_buffer(train)



train_gdf = gpd.GeoDataFrame(train, geometry='coordinates')

train_similarity_matrix_on_train_recods = train_gdf.groupby(['trip_index',
                'route_id']).apply(lambda trip: find_route_id_similarity(trip, train_coordinates_gdf))


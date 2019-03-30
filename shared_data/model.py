#imports

import pandas as pd
import numpy as np
import math

from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import geopandas as gpd
#from seaborn import palplot


pd.options.display.max_columns =200
pd.options.display.max_rows =500
pd.options.display.max_colwidth =500


## functions

def route_stats_split_coordinates(route_id):
    coordinates_strings = [latlon.split(',') for latlon in route_id.all_stop_latlon.split(';')]
    coordinates_floats = [(float(lat), float(lon)) for lat, lon in coordinates_strings]

    return coordinates_floats


def route_stats_split_stops(route_id, col):
    return route_id[col].split(';')


def route_stats_2_stops(route_stats):
    route_stats = route_stats.copy()

    route_stats_stops = pd.DataFrame(columns=['route_id', 'stop_index', 'stop_id', 'coordinates'])

    route_stats['all_stop_latlon'] = route_stats.apply(lambda r: route_stats_split_coordinates(r), axis=1)
    route_stats['all_stop_id'] = route_stats.apply(lambda r: route_stats_split_stops(r, 'all_stop_id'), axis=1)

    for route_id in route_stats['route_id']:
        route_id_stops = pd.DataFrame()
        route_id_stops['stop_id'] = pd.Series(
            route_stats[route_stats['route_id'] == route_id]['all_stop_id'].tolist()[0])
        route_id_stops['coordinates'] = pd.Series(
            route_stats[route_stats['route_id'] == route_id]['all_stop_latlon'].tolist()[0])
        route_id_stops['stop_index'] = route_id_stops.index + 1
        route_id_stops['route_id'] = route_id

        route_stats_stops = route_stats_stops.append(route_id_stops)

    route_stats_stops = route_stats_stops[['route_id', 'stop_index', 'stop_id', 'coordinates']] \
        .sort_values(by=['route_id', 'stop_index'])

    return route_stats_stops


def plot_route_id_stops(route_stats_stops):
    route_stats_stops['coordinates'] = route_stats_stops['coordinates'].apply(Point)
    gdf = gpd.GeoDataFrame(route_stats_stops, geometry='coordinates')

    route_ids = dict(tuple(gdf.groupby('route_id')))

    fig = plt.figure(num=None, figsize=(10, 8))

    first_route_id = list(route_ids.keys())[0]

    ax = route_ids[first_route_id].plot()

    for route_id in route_ids.keys():
        if route_id == first_route_id:
            pass
        else:
            route_ids[route_id].plot(ax=ax)


def set_buffer(route_stats_stops, buffer_rad=0.01):
    route_stats_stops['coordinates'] = route_stats_stops['coordinates'].apply(Point)
    gdf = gpd.GeoDataFrame(route_stats_stops, geometry='coordinates')
    gdf['buffer'] = gdf.buffer(buffer_rad)
    gdf.set_index('stop_id', inplace=True, drop=False)

    return gdf


def find_route_id_similarity(trip, route_stats_stops_gdf, route_stats_unique):
    similarity_summary = {}

    for route_id in route_stats_unique.route_id:

        route_id_buffers = route_stats_stops_gdf[route_stats_stops_gdf['route_id'] == route_id]['buffer']

        route_id_stops_results = {key: trip.within(geom) for key, geom in route_id_buffers.items()}

        route_id_stops_summary = {}
        for stop in route_id_stops_results.keys():
            route_id_stops_summary[stop] = np.any(route_id_stops_results[stop])

        route_id_mean = np.mean(list(route_id_stops_summary.values()))

        route_id_count = len(route_id_stops_summary)

        similarity_summary[str(route_id) + '_mean'] = [route_id_mean]

        similarity_summary[str(route_id) + '_cnt'] = [route_id_count]

    return pd.DataFrame(similarity_summary)


def match_route_id(trip, route_stats_unique, min_match_prob=0.8, floor_val=0.05):
    match_summary = {}

    # drop route_ids with % prob < min_match_prob
    for route_id in route_stats_unique.route_id:

        # for each route
        if trip[str(route_id) + '_mean'] >= min_match_prob:
            match_summary[route_id] = round(math.floor(trip[str(route_id) + '_mean'] / floor_val) * floor_val, 2)

    if not match_summary:
        return 'No match'

    else:

        # choose best match
        max_match = max(match_summary.values())

        match_summary = {key: val for key, val in match_summary.items() if val == max_match}

        if len(match_summary) == 1:

            # if
            return list(match_summary.keys())[0]

        else:
            max_match_route_ids = match_summary.keys()
            match_summary = {}  # reset match_summary to contain route_ids number of stops
            for route_id in max_match_route_ids:
                match_summary[route_id] = trip[str(route_id) + '_cnt']

            # match_summary = sorted(match_summary.items(), key=lambda x: x[1], reverse=True)

            max_match = max(match_summary.values())

            match_summary = {key: val for key, val in match_summary.items() if val == max_match}

            if len(match_summary) == 1:

                # if
                return list(match_summary.keys())[0]
            else:

                # if
                return 'No match'





## script

train = pd.read_csv('train.csv', encoding='windows-1255')
train.drop_duplicates(subset=['trip_index']).route_id.value_counts()
route_stats = pd.read_csv('route_stats.csv', index_col=[0])
route_stats_unique = route_stats.groupby(['route_id','all_stop_id','all_stop_latlon'],
                                        as_index=False)['date'].agg(['min','max'])\
                                        .rename(columns = {'min':'start_date', 'max': 'end_date'})\
                                        .reset_index().sort_values(by = ['route_id','start_date'])

route_stats_stops = route_stats_2_stops(route_stats_unique)
route_stats_stops_gdf = set_buffer(route_stats_stops)
train['coordinates'] = list(zip(train.lat, train.lon))
train['coordinates'] = train['coordinates'].apply(Point)

train_gdf = gpd.GeoDataFrame(train, geometry='coordinates')
train_similarity_matrix = train_gdf.groupby(['trip_index',
                'route_id']).apply(lambda trip: find_route_id_similarity(trip, route_stats_stops_gdf, route_stats_unique))

train_similarity_matrix.tail(1).apply(lambda trip: match_route_id(trip, route_stats_unique), axis=1).to_frame()

train_trips = train_similarity_matrix.apply(lambda trip: match_route_id(trip, route_stats_unique), axis=1).to_frame()

train_trips.rename(columns = {0: 'route_id_pred'}, inplace=True)

train_trips.reset_index(inplace=True)

train_trips.drop(columns=['level_2'], inplace=True)

train_trips['route_id_pred_true'] = np.where(train_trips.route_id==train_trips.route_id_pred,1,0)


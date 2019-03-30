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



load ()
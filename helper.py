import math
import heapq
import numpy as np
from math import pi
import igraph as ig
from global_land_mask import globe

def shortest_path(g, src, target):
    """
    Implementation of Dijkstra's Algorithm
    """
    q = [(0, src, ())]
    visited, dist = set(), {src: 0.0}
    while q:
        cost, v, path = heapq.heappop(q)
        print(v, target)
        print("distance left: ", math.sqrt((v[0] - target[0])*(v[0] - target[0]) + (v[1] - target[1])*(v[1] - target[1])))
        if v not in visited:
            visited.add(v)
            path += (v,)
            if v == target:
                return (cost, path)
            
            for cost2, v2 in g.get(v, ()):
                if v2 in visited:
                    continue
                if cost + cost2 < dist.get(v2, float('inf')):
                    dist[v2] = cost + cost2
                    heapq.heappush(q, (cost + cost2, v2, path)) 
    return (float('inf'), ())

def create_graph(x, y):
    """
    Creates Graph with Ocean (lon, lat) as nodes and edges with neighbouring ocean nodes 
    """
    edges = []
    for i in range(len(x)):
        for j in range(len(y)):
            if globe.is_land(y[j], x[i]):
                continue

            center = get_node_index(i, j, len(y))

            if not globe.is_land(y[j], x[(i - 1 + len(x))%len(x)]):
                top = get_node_index((i-1+len(x)) % len(x), j, len(y))
                edges.append((center, top))

            if not globe.is_land(y[j], x[(i+1)%len(x)]):
                bottom = get_node_index((i+1)%len(x), j, len(y))
                edges.append((center, bottom))
                
            if not globe.is_land(y[(j+1)%len(y)], x[i]):
                right = get_node_index(i, (j+1)%len(y), len(y))
                edges.append((center, right))

            if not globe.is_land(y[(j-1+len(y))%len(y)], x[i]):
                left = get_node_index(i, (j-1+len(y))%len(y), len(y))
                edges.append((center, left))

            if not globe.is_land(y[(j+1)%len(y)], x[(i-1+len(x)) % len(x)]):
                top_right = get_node_index((i-1+len(x)) % len(x), (j+1)%len(y), len(y))
                edges.append((center, top_right))

            if not globe.is_land(y[(j-1+len(y))%len(y)], x[(i-1+len(x)) % len(x)]):
                top_left = get_node_index((i-1+len(x)) % len(x), (j-1+len(y))%len(y), len(y))
                edges.append((center, top_left))

            if not globe.is_land(y[(j+1)%len(y)], x[(i+1) % len(x)]):
                bottom_right = get_node_index((i+1) % len(x), (j+1)%len(y), len(y))
                edges.append((center, bottom_right))

            if not globe.is_land(y[(j-1+len(y))%len(y)], x[(i+1) % len(x)]):
                bottom_left = get_node_index((i+1) % len(x), (j-1+len(y))%len(y), len(y))
                edges.append((center, bottom_left))

    G = ig.Graph(len(x) * len(y), edges)  
    return G

def calculate_cost(X, Y, U, V, v1, v2, s0):
    """
    Calculates time taken by vessel to travel a distance considering ocean currents
    """
    j1, i1 = v1
    j2, i2 = v2
    
    u = (U[j1,i1] + U[j2,i2])/2.
    v = (V[j1,i1] + V[j2,i2])/2.
    
    ds = distance(Y[v1], X[v1], Y[v2], X[v2])
    a = bearing(Y[v1], X[v1], Y[v2], X[v2])
    
    # Velocity along track
    s = s0 + u*np.cos(a) + v*np.sin(a)

    if s < 0:
        return np.inf
    else:
        return ds/s

def distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Great-circle distance
    """
    # http://www.movable-type.co.uk/scripts/latlong.html
    R = 6.371e6
    lat1 *= pi/180.
    lon1 *= pi/180.
    lat2 *= pi/180.
    lon2 *= pi/180.
    return R*np.arccos(
        np.sin(lat1)*np.sin(lat2) + 
        np.cos(lat1)*np.cos(lat2)*np.cos(lon2-lon1))

def bearing(lat1, lon1, lat2, lon2):
    """
    Calculates Bearing (angle)
    """
    lat1 *= pi/180.
    lon1 *= pi/180.
    lat2 *= pi/180.
    lon2 *= pi/180.
    y = np.sin(lon2-lon1)*np.cos(lat2)
    x = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(lon2-lon1)
    return (pi/2) - np.arctan2(y, x)

def get_node_index(i, j, len_y):
    """
    2D Index -> 1D Index
    """
    return i * len_y + j

def get_coord(index, len_y):
    """
    1D Index -> 2D Index
    """
    return (index // len_y, index % len_y)

def get_index_from_lat_long(x, y, coord):
    """
    coord: list
        (latitude, longitude)
    """
    return (np.absolute(y - coord[0]).argmin(), np.absolute(x - coord[1]).argmin())

def get_distance(v1, v2):
    """
    Calculate the Great-circle distance
    """
    return (np.linalg.norm(v1-v2))


# Voyage-Optimization - [Live App](https://app.daisi.io/daisies/soul0101/Voyage%20Optimization/app)
<p align="center">
    <img src="https://user-images.githubusercontent.com/53980340/192162286-6806f674-22be-4d96-bb61-4ababaa5b875.png" alt="Logo" width="700">        
</p>


When traveling on the surface of the Earth one cannot take a constant heading (an angle with respect to North) to travel the shortest route from point __A__ to __B__. <br>
Instead, the heading must be constantly readjusted so that the arc of the trajectory corresponds to the intersection between the globe and a plane that passes through the center of the Earth. <br>
For ships the ocean currents are an important factor.<br>

## How does this work?
- The daisi generates a __Graph__ with the (latitude, longitude) grid of ocean points around the globe as vertices.
- Edges are the __8-point__ ocean neighbours with weights being the time taken by the ship to travel the length of the edge considering the __Ocean Currents__. 
- This daisi generates an optimal route for a ship by running __Dijkstra's Shortest Path Algorithm__ on this graph with the __Start__ and __End__ points taken as inputs.
## TEST API
``` python
import pydaisi as pyd
voyage_optimization = pyd.Daisi("soul0101/Voyage Optimization")

start_coord = (14.37, 89.45)
end_coord = (32.64, -18.90)
boat_avg_speed = 1.0

lon, lat, U, V = voyage_optimization.get_ocean_current_dataset().value
lon, lat, U, V = voyage_optimization.process_ds(lon, lat, U, V).value

G = voyage_optimization.graph_factory(lon, lat, U, V, boat_avg_speed).value
path = voyage_optimization.get_optimal_routes(G, start_coord, end_coord, lon, lat).value
xx, yy = voyage_optimization.get_coordinates_from_path_indices(path, lon, lat).value
```

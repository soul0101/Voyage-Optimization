# Voyage-Optimization
When traveling on the surface of the Earth one cannot take a constant heading (an angle with respect to North) to travel the shortest route from point __A__ to __B__. <br>
Instead, the heading must be constantly readjusted so that the arc of the trajectory corresponds to the intersection between the globe and a plane that passes through the center of the Earth. <br>
For ships the ocean currents are an important factor.<br>

## How does this work?
- The daisi generates a __Graph__ with the (latitude, longitude) grid of ocean points around the globe as vertices.
- Edges are the 8-point ocean neighbours with weights being the time taken by the ship to travel the length of the edge considering the __Ocean Currents__. 
- This daisi generates an optimal route for a ship by running __Dijkstra's Shortest Path Algorithm__ on this graph with the start and end points taken as inputs.
## TEST API
```
```

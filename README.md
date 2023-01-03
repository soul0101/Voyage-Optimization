# Voyage-Optimization
<p align="center">
    <img src="https://user-images.githubusercontent.com/53980340/192162286-6806f674-22be-4d96-bb61-4ababaa5b875.png" alt="Logo" width="700">        
</p>


When traveling on the surface of the Earth one cannot take a constant heading to travel the shortest route from point __A__ to __B__. <br>
Instead, the heading must be constantly readjusted so that the arc of the trajectory corresponds to the intersection between the globe and a plane that passes through the center of the Earth. <br>
For ships the ocean currents are an important factor which can be harnessed to produce the optimal path.<br>

## How does this work?
- This app generates a __Graph__ with the (latitude, longitude) grid of ocean points around the globe as vertices.
- Edges are the __8-point__ ocean neighbours with weights being the time taken by the ship to travel the length of the edge considering the __Ocean Currents__. 
- This app generates an optimal route for a ship by running __Dijkstra's Shortest Path Algorithm__ on this graph with the __Start__ and __End__ points taken as inputs.

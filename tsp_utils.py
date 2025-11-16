"""
Helper functions for TSP assignment
Includes:
    - load_distance_data: Load city and distance data from CSV
    - calculate_tour_distance: Compute total distance of a tour
    - plot_plan: Visualize a tour on a map
"""

import matplotlib.pyplot as plt
import csv

# Map of Europe
europe_map = plt.imread('map.png')

# List of city coordinates
city_coords = {
    "Barcelona": [2.154007, 41.390205], "Belgrade": [20.46, 44.79], "Berlin": [13.40, 52.52], 
    "Brussels": [4.35, 50.85], "Bucharest": [26.10, 44.44], "Budapest": [19.04, 47.50],
    "Copenhagen": [12.57, 55.68], "Dublin": [-6.27, 53.35], "Hamburg": [9.99, 53.55], 
    "Istanbul": [28.98, 41.02], "Kyiv": [30.52, 50.45], "London": [-0.12, 51.51], 
    "Madrid": [-3.70, 40.42], "Milan": [9.19, 45.46], "Moscow": [37.62, 55.75],
    "Munich": [11.58, 48.14], "Paris": [2.35, 48.86], "Prague": [14.42, 50.07],
    "Rome": [12.50, 41.90], "Saint Petersburg": [30.31, 59.94], "Sofia": [23.32, 42.70],
    "Stockholm": [18.06, 60.33], "Vienna": [16.36, 48.21], "Warsaw": [21.02, 52.24]
}

# Read CSV and build distance dictionary
def load_distance_data(filename):
    """
    Load city names and distance matrix from CSV file

    Args:
        filename: Path to CSV file
    
    Returns:
        cities: List of city names
        distances: Distance dictionary 
    """
    with open("european_cities.csv", "r") as f:
        data = list(csv.reader(f, delimiter=';'))
    
    cities = data[0] # First row is city names

    # Build distance dictionary: distances[city1][city2] = distance
    distances = {}
    for i, city1 in enumerate(cities):
        distances[city1] = {}
        for j, city2 in enumerate(cities):
            distances[city1][city2] = float(data[i + 1][j])
    
    return cities, distances

# Function to calculate total tour distance
def calculate_tour_distance(tour, distances):
    """
    Calculate total distance of a tour

    Args:
        tour: List of cities in the order they are visited
        distances: Distance dictionary 

    Returns:
        total_distance: Total distance of the tour
    """
    total_distance = 0
    for i in range(len(tour)):
        # Distance from current city to next city (wraps around to first city)
        current_city = tour[i]
        next_city = tour[(i + 1) % len(tour)]
        total_distance += distances[current_city][next_city]
    
    return total_distance 

# Function to visualize a tour
def plot_plan(city_order, title="TSP Tour"):
    """
    Plot a tour on the map

    Args:
        city_order: List of cities in the order they are visited
        title: Title for the plot

    Returns:
        fig: Matplotlib figure object
    """
    try:
        europe_map = plt.imread('map.png')
        has_map = True
    except:
        print("Warning: map.png not found. Plotting without background map")
        has_map = False
    
    fig, ax = plt.subplots(figsize=(10, 10))

    if has_map:
        ax.imshow(europe_map, extent=[-14.56, 38.43, 37.697 + 0.3, 64.344 + 2.0], aspect="auto")

    # Plot the tour
    for index in range(len(city_order) - 1):
        current_city_coords = city_coords[city_order[index]]
        next_city_coords = city_coords[city_order[index + 1]]
        x, y = current_city_coords[0], current_city_coords[1]
        next_x, next_y = next_city_coords[0], next_city_coords[1]

        plt.plot([x, next_x], [y, next_y], 'b-', linewidth=2)
        plt.plot(x, y, 'ok', markersize=5)
        plt.text(x, y, f' {index}', fontsize=10)

    # Plot from last city back to first
    first_city_coords = city_coords[city_order[0]]
    first_x, first_y = first_city_coords[0], first_city_coords[1]
    plt.plot([next_x, first_x], [next_y, first_y], 'b-', linewidth=2)
    plt.plot(next_x, next_y, 'ok', markersize=5)
    plt.text(next_x, next_y, f' {index + 1}', fontsize=10)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.tight_layout()

    return fig
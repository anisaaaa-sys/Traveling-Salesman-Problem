"""
TSP Hill Climber Solution
Implements a simple hill climbing algorithm for the Traveling Salesman Problem.

Run command: python3 tsp_hill_climbing.py
"""

import random
import time
import matplotlib.pyplot as plt
import numpy as np
from tsp_utils import *

# Generate a random tour
def generate_random_tour(cities):
    """
    Generate a random permutation of cities

    Args:
        cities: List of cities to visit

    Returns:
        tour: Random tour (list of cities)
    """
    tour = cities.copy()
    random.shuffle(tour)
    return tour

# Get all neighbors using 2-opt swap
def get_neighbors(tour):
    """
    Generate all neighbors by swapping two cities (2-opt)
    This creates all possible tours that differ by one swap

    Args:
        tour: Current tour (list of cities)
    
    Returns:
        neighbors: List of neighboring tours
    """
    neighbors = []
    n = len(tour)

    for i in range(n - 1):
        for j in range(i + 1, n):
            # Create a neighbor by swapping cities at positions i and j
            neighbor = tour.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)
    
    return neighbors

# Hill Climber algorithm 
def hill_climber(cities, distances, max_iterations=10000):
    """
    Simple hill climbing algorithm for TSP

    Args:
        cities: List of cities to visit
        distances: Distance dictionary
        max_iterations: Maximum numbers o iterations (prevents infinite loops)

    Returns:
        best_tour: Best tour found
        best_distance: Distance of best tour
        iterations: Number of iterations performed
    """

    # Start with a random tour
    current_tour = generate_random_tour(cities)
    current_distance = calculate_tour_distance(current_tour, distances)

    iterations = 0
    improvements = 0

    while iterations < max_iterations:
        # Generate all neighbors
        neighbors = get_neighbors(current_tour)

        # Find the best neighbor
        best_neighbor = None
        best_neighbor_distance = current_distance

        for neighbor in neighbors:
            neighbor_distance = calculate_tour_distance(neighbor, distances)
            if neighbor_distance < best_neighbor_distance:
                best_neighbor = neighbor
                best_neighbor_distance = neighbor_distance
            
        # If no better neighbor found, we've reached a local optimum
        if best_neighbor is None:
            break

        # Move to the better neighbor
        current_tour = best_neighbor
        current_distance = best_neighbor_distance
        improvements += 1
        iterations += 1
    
    return current_tour, current_distance, iterations

# Run multiple hill climber trials
def run_hill_climber_trials(cities, distances, num_trials=20):
    """
    Run hill climber multiple times and collect statistics

    Args:
        cities: List of cities to visit
        distances: Distance dictionary
        num_trials: Number of trials to run

    Returns:
        results: Dictionary with statistics and all tour results
    """

    print(f"\nRunning {num_trials} trials of hill climber on {len(cities)} cities...")

    all_distances = []
    all_tours = []
    all_times = []

    for trial in range(num_trials):
        start_time = time.time()
        tour, distance, iterations = hill_climber(cities, distances)
        elapsed_time = time.time() - start_time

        all_distances.append(distance)
        all_tours.append(tour)
        all_times.append(elapsed_time)

        if (trial + 1) % 5 == 0:
            print(f"  Completed {trial + 1}/{num_trials} trials...")
        
    # Calculate statistics
    best_idx = np.argmin(all_distances)
    worst_idx = np.argmax(all_distances)

    results = {
        'best_distance' : all_distances[best_idx],
        'best_tour' : all_tours[best_idx],
        'worst_distance' : all_distances[worst_idx],
        'worst_tour' : all_tours[worst_idx],
        'mean_distance' : np.mean(all_distances),
        'std_distance' : np.std(all_distances),
        'all_distances' : all_distances,
        'all_tours' : all_tours,
        'avg_time' : np.mean(all_times)
    }

    return results

# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("TSP HILL CLIMBER")
    print("=" * 70)

    # Load data
    cities, distances = load_distance_data('european_cities.csv')

    # Set random seed for reproducibility 
    random.seed(57)
    np.random.seed(57)

    # Test configurations
    test_configs = [
        {'n' : 10, 'cities' : cities[:10], 'name' : '10 cities'},
        {'n' : 24, 'cities' : cities, 'name' : '24 cities (all)'}
    ]

    all_results = {}

    for config in test_configs:
        n = config['n']
        test_cities = config['cities']
        name = config['name']

        print("\n" + "=" * 70)
        print(f"TESTING WITH {name.upper()}")
        print("=" * 70)
        print(f"Cities: {test_cities}")

        # Run 20 trials
        results = run_hill_climber_trials(test_cities, distances, num_trials=20)
        all_results[n] = results

        # Print statistics
        print(f"\n{'=' * 70}")
        print(f"RESULTS FOR {name.upper()}")
        print(f"{'=' * 70}")
        print(f"Best distance:    {results['best_distance']:.2f} km")
        print(f"Worst distance:   {results['worst_distance']:.2f} km")
        print(f"Mean distance:    {results['mean_distance']:.2f} km")
        print(f"Std deviation:    {results['std_distance']:.2f} km")
        print(f"Avg time/run:     {results['avg_time']:.4f} seconds")

        # Plot the best tour (one of the 20 runs)
        print(f"\nPlotting best tour for {name}...")
        title = f"Hill Climber - Best Tour ({name})\nDistance: {results['best_distance']:.2f} km"
        plot_plan(results['best_tour'], title=title)
        plt.savefig(f'hill_climber_best_{n}_cities.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # Comparison with exhaustive search (for 10 cities)
    print("\n" + "=" * 70)
    print("COMPARISON WITH EXHAUSTIVE SEARCH (10 cities)")
    print("=" * 70)

    exhaustive_best_10 = 7486.31 # From exhaustive search result

    if exhaustive_best_10 is not None:
        hc_best_10 = all_results[10]['best_distance']
        difference = hc_best_10 - exhaustive_best_10
        percentage = (difference / exhaustive_best_10) * 100

        print(f"\nExhaustive search optimal: {exhaustive_best_10:.2f} km")
        print(f"Hill climber best:         {hc_best_10:.2f} km")
        print(f"Difference:                {difference:.2f} km ({percentage:.2f}%)")
        print(f"Hill climber worst:        {all_results[10]['worst_distance']:.2f} km")
        print(f"Hill climber mean:         {all_results[10]['mean_distance']:.2f} km")
    else:
        print("\nTo compare with exhaustive search:")
        print("1. Run your exhaustive search code first")
        print("2. Update the 'exhaustive_best_10' variable with the optimal distance")
        print("3. Re-run this script")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Cities':<10} {'Best (km)':<12} {'Worst (km)':<12} {'Mean (km)':<12} {'Std Dev':<12}")
    print("-" * 70)
    for n in [10, 24]:
        r = all_results[n]
        print(f"{n:<10} {r['best_distance']:<12.2f} {r['worst_distance']:<12.2f} "
              f"{r['mean_distance']:<12.2f} {r['std_distance']:<12.2f}")
    
    print("\n" + "=" * 70)
    print("HILL CLIMBER COMPLETE")
    print("=" * 70)
    print("✓ Ran 20 trials for both 10 and 24 cities")
    print("✓ Plots saved for best tours")
    print("✓ Statistics calculated (best, worst, mean, std)")
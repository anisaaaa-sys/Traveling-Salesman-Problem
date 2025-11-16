"""
TSP Exhaustive Search Solution
Implements exhaustive search to find optimal tour for small subset of cities.

Run command: python3 tsp_exhaustive_search.py
"""

import matplotlib.pyplot as plt
import math
import time
from itertools import permutations
from tsp_utils import *

# Exhaustive Search algorithm: try all possible permutations
def exhaustive_search(cities_subset, distances):
    """
    Find the shortest tour using exhaustive search

    Args:
        cities_subset: List of cities to include in the tour
        distances: Dictionary of distances between cities

    Returns:
        best_tour: The shortest tour found
        best_distance: Distance of the shortest tour
        elapsed_time: Time taken to find the solution
    """
    print(f"\nSearching through {len(cities_subset)} cities...")
    print(f"\nCities: {cities_subset}")
    print(f"Total permutations to check: {len(list(permutations(cities_subset[1:])))}")

    start_time = time.time()

    # Fix the first city to reduce redundant tours (since tours are circular)
    first_city = cities_subset[0]
    remaining_cities = cities_subset[1:]

    best_tour = None
    best_distance = float('inf')

    count = 0
    for perm in permutations(remaining_cities):
        tour = [first_city] + list(perm)
        distance = calculate_tour_distance(tour, distances)

        if distance < best_distance:
            best_distance = distance
            best_tour = tour
        
        count += 1
        if count % 1000 == 0:
            print(f"Checked {count} permutations...")
        
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nBest tour found: {best_tour}")
    print(f"Total distance: {best_distance:.2f} km")
    print(f"Time taken: {elapsed_time:.4f} seconds")
    print(f"Permutations checked: {count}")

    return best_tour, best_distance, elapsed_time

# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("TSP EXHAUSTIVE SEARCH")
    print("=" * 70)
    # Load data
    cities, distances = load_distance_data('european_cities.csv')

    print(f"Total cities available: {len(cities)}")
    print(f"All cities: {cities}\n")

    # Test with different numbers of cities
    test_sizes = [6, 7, 8, 9, 10]
    results = {}

    for n in test_sizes:
        print("=" * 70)
        print(f"TESTING WITH {n} CITIES")
        print("=" * 70)

        # Take first n cities
        cities_subset = cities[:n]

        best_tour, best_distance, elapsed_time = exhaustive_search(
            cities_subset, distances
        )

        results[n] = {
            'tour' : best_tour,
            'distance' : best_distance,
            'time' : elapsed_time
        }

        # Plot tours for 6 and 10 cities
        if n == 6 or n == 10:
            print(f"\nPlotting tour for {n} cities...")
            fig = plot_plan(best_tour,
                            title=f"Best Tour for {n} Cities\nDistance: {best_distance:.2f} km | Time: {elapsed_time:.4f}s")
            plt.savefig(f'exhaustive_search_best_{n}_cities.png', dpi=150, bbox_inches='tight')
            plt.show()
            print(f"✓ Saved as 'tsp_tour_{n}_cities.png")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF RESULTS")
    print("=" * 70)
    print(f"{'Cities':<10} {'Distance (km)':<15} {'Time (seconds)':<20} {'Factorial':<15}")
    print("-" * 70)

    for n in test_sizes:
        if n in results:
            factorial = math.factorial(n - 1) # Fix first city
            print(f"{n:<10} {results[n]['distance']:<15.2f} {results[n]['time']:<20.4f} {factorial:<15}")

    print("\n" + "=" * 70)
    print("EXHAUSTIVE SEARCH COMPLETE")
    print("=" * 70)
    print(f"✓ Tours visualized for 6 and 10 cities")
    print("✓ Plots saved for 6 and 10 cities")
"""
TSP Genetic Algorithm Solution
Implements a genetic algorithm for the Traveling Salesman Problem.

Run command: python3 tsp_genetic_algorithm.py
"""

import random
import time
import matplotlib.pyplot as plt
import numpy as np
from tsp_utils import *

def initialize_population(cities, population_size):
    """
    Initialize population with random tours

    Args: 
        cities: List of cities
        population_size: Number of individuals in population

    Returns:
        List of random tours (individuals)
    """
    population = []
    for _ in range(population_size):
        tour = cities.copy()
        random.shuffle(tour)
        population.append(tour)
    
    return population

def fitness(tour, distances):
    """
    Calculate fitness of a tour (negative distance, since we want to minimize)

    Args:
        tour: A tour (list of cities)
        distances: Distance dictionary
    
    Returns: 
        Fitness value (negative distance for maximization)
    """
    return -calculate_tour_distance(tour, distances)

def tournament_selection(population, distances, tournament_size=5):
    """
    Select an individual using tournament selection

    Args:
        population: Current population
        distances: Distance dictionary
        torunament_size: Number of individuals in tournament
    
    Returns:
        Selected individual
    """
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda tour: fitness(tour, distances))

def order_crossover(parent1, parent2):
    """
    Order Crossover (OX) - appropriate for TSP as it preserves order

    Args:
        parent1, parent2: Parent tours
    
    Returns:
        Two offspring tours
    """
    size = len(parent1)

    # Choose two random crossover points
    cx_point1 = random.randint(0, size - 1)
    cx_point2 = random.randint(cx_point1 + 1, size)

    # Create offspring 1
    offspring1 = [None] * size
    offspring1[cx_point1:cx_point2] = parent1[cx_point1:cx_point2]

    # Fill remaining positions with cities from parent2 in order
    pointer = cx_point2
    for city in parent2[cx_point2:] + parent2[:cx_point2]:
        if city not in offspring1:
            if pointer >= size:
                pointer = 0
            offspring1[pointer] = city
            pointer += 1

    # Create offspring 2
    offspring2 = [None] * size
    offspring2[cx_point1:cx_point2] = parent2[cx_point1:cx_point2]

    pointer = cx_point2
    for city in parent1[cx_point2:] + parent1[:cx_point2]:
        if city not in offspring2:
            if pointer >= size:
                pointer = 0
            offspring2[pointer] = city
            pointer += 1
    
    return offspring1, offspring2

def swap_mutation(tour, mutation_rate=0.1):
    """
    Swap Mutation - swap two random cities
    (Chapter 4.5.2 in Eiben & Smith)
    
    Args:
        tour: Tour to mutate
        mutation_rate: Probability of mutation

    Returns:
        Mutated tour
    """
    tour = tour.copy()

    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(tour)), 2)
        tour[idx1], tour[idx2] = tour[idx2], tour[idx1]
    
    return tour

def inversion_mutation(tour, mutation_rate=0.1):
    """
    Inversion Mutation - reverse a segment of the tour
    Often effective for TSP as it can untangle crossed paths

    Args: 
        tour: Tour to mutate
        mutation_rate: Probability of mutation

    Returns:
        Mutated tour
    """
    tour = tour.copy()

    if random.random() < mutation_rate:
        idx1 = random.randint(0, len(tour) - 2)
        idx2 = random.randint(idx1 + 1, len(tour))
        tour[idx1:idx2] = reversed(tour[idx1:idx2])
    
    return tour

def genetic_algorithm(cities, distances, population_size=100, generations=500,
                      mutation_rate=0.1, crossover_rate=0.9, 
                      tournament_size=5, elitism= 2):
    """
    Genetic Algorithm for TSP

    Args:
        cities: List of cities
        distances: Distance dictionary
        population_size: Number of individuals in population
        generations: Number of generations to evolve
        mutation_rate: Probability of mutation
        crossover_rate: Probability of crossover
        tournament_size: Size of tournament for selection
        elitism: Number of best individuals to preserve

    Returns:
        best_tour: Best tour found
        best_distance: Distance of best tour
        best_fitness_history: Best fitness in each generation
    """
    # Initialize population
    population = initialize_population(cities, population_size)

    # Track best fitness over generations
    best_fitness_history = []

    for generation in range(generations):
        # Evaluate fitness
        fitness_values = [fitness(tour, distances) for tour in population]

        # Track best individual
        best_idx = np.argmax(fitness_values)
        best_fitness_history.append(-fitness_values[best_idx]) # Convert back to distance

        # Create new population
        new_population = []

        # Elitism: keep best individuals
        sorted_population = [tour for _, tour in sorted(zip(fitness_values, population),
                                                        key=lambda x: x[0], reverse=True)]
        new_population.extend(sorted_population[:elitism])

        # Generate offspring
        while len(new_population) < population_size:
            # Selection
            parent1 = tournament_selection(population, distances, tournament_size)
            parent2 = tournament_selection(population, distances, tournament_size)

            # Crossover
            if random.random() < crossover_rate:
                offspring1, offspring2 = order_crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1.copy(), parent2.copy()
            
            # Mutation (using inversion mutation for TSP)
            offspring1 = inversion_mutation(offspring1, mutation_rate)
            offspring2 = inversion_mutation(offspring2, mutation_rate)

            new_population.extend([offspring1, offspring2])
        
        # Trim to population size
        population = new_population[:population_size]
    
    # Return best individuals from final population
    fitness_values = [fitness(tour, distances) for tour in population]
    best_idx = np.argmax(fitness_values)
    best_tour = population[best_idx]
    best_distance = -fitness_values[best_idx]

    return best_tour, best_distance, best_fitness_history

def run_ga_trials(cities, distances, population_size, num_trials=20, generations=500):
    """
    Run GA multiple times and collect statistics

    Args:
        cities: List of cities to visit
        distances: Distance dictionary
        population_size: Number of individuals in population
        num_trials: Number of trials to run
        generations: Number of generations to evolve

    Returns 
        Dictionary with results and statistics
    """
    print(f"\nRunning {num_trials} trials of GA with population size {population_size}...")

    all_distances = []
    all_tours = []
    all_fitness_histories = []
    all_times = []

    for trial in range(num_trials):
        start_time = time.time()

        tour, distance, fitness_history = genetic_algorithm(
            cities, distances,
            population_size=population_size,
            generations=generations,
            mutation_rate=0.1,
            crossover_rate=0.9,
            tournament_size=5,
            elitism=2
        )

        elapsed_time = time.time() - start_time

        all_distances.append(distance)
        all_tours.append(tour)
        all_fitness_histories.append(fitness_history)
        all_times.append(elapsed_time)

        if (trial + 1) % 5 == 0:
            print(f"  Completed {trial + 1}/{num_trials} trials...")
    
    # Calculate statistics
    best_idx = np.argmin(all_distances)
    worst_idx = np.argmax(all_distances)

    # Calculate average fitness over generations (average across all runs)
    avg_fitness_history = np.mean(all_fitness_histories, axis=0)

    results = {
        'best_distance' : all_distances[best_idx],
        'best_tour' : all_tours[best_idx],
        'worst_distance' : all_distances[worst_idx],
        'worst_tour' : all_tours[worst_idx],
        'mean_distance' : np.mean(all_distances),
        'std_distance' : np.std(all_distances),
        'all_distances' : all_distances,
        'all_tours' : all_tours,
        'avg_fitness_history' : avg_fitness_history,
        'all_fitness_histories' : all_fitness_histories,
        'avg_time' : np.mean(all_times)
    }

    return results 

# Main execution 
if __name__ == "__main__":
    print("=" * 70)
    print("TSP GENETIC ALGORITHM")
    print("=" * 70)
    
    # Load data
    cities, distances = load_distance_data('european_cities.csv')
    
    # Use all 24 cities for GA
    test_cities = cities

    print(f"\nTesting with all {len(test_cities)} cities")
    print(f"Cities: {test_cities}\n")

    # Set random seed for reproducibility
    random.seed(57)
    np.random.seed(57)

    # GA Parameters 
    print("=" * 70)
    print("GENETIC ALGORITHM PARAMETERS")
    print("=" * 70)
    print("• Generations: 500")
    print("• Mutation rate: 0.1 (10%)")
    print("• Crossover rate: 0.9 (90%)")
    print("• Crossover operator: Order Crossover (OX)")
    print("• Mutation operator: Inversion Mutation")
    print("• Selection: Tournament Selection (size=5)")
    print("• Elitism: 2 (best individuals preserved)")
    print("• Number of trials: 20")
    print("=" * 70)

    test_configs = [
        {'n': 10, 'cities': cities[:10], 'name': '10 cities'}, 
        {'n': 24, 'cities': cities, 'name': '24 cities (all)'}
    ]
    # Three different population sizes to test
    population_sizes = [50, 100, 200]

    for n in [10, 24]:  
        print(f"\n{'=' * 70}")
        print(f"TESTING WITH {n} CITIES")
        print(f"{'=' * 70}")
    
        test_cities = cities[:n] if n == 10 else cities
        all_results = {}

        for pop_size in population_sizes:
            print("\n" + "=" * 70)
            print(f"TESTING WITH POPULATION SIZE: {pop_size}")
            print("=" * 70)

            results = run_ga_trials(test_cities, distances, pop_size,
                                    num_trials=20, generations=500)
            all_results[pop_size] = results

            # Print statistics
            print(f"\n{'=' * 70}")
            print(f"RESULTS FOR POPULATION SIZE {pop_size}")
            print(f"{'=' * 70}")
            print(f"Best distance:    {results['best_distance']:.2f} km")
            print(f"Worst distance:   {results['worst_distance']:.2f} km")
            print(f"Mean distance:    {results['mean_distance']:.2f} km")
            print(f"Std deviation:    {results['std_distance']:.2f} km")
            print(f"Avg time/run:     {results['avg_time']:.2f} seconds")

            # Plot best tour for this population size
            print(f"\nPlotting best tour for population size {pop_size}...")
            title = f"GA - Best Tour (Pop Size: {pop_size})\nDistance: {results['best_distance']:.2f} km"
            plot_plan(results['best_tour'], title=title)
            plt.savefig(f'ga_best_tour_pop{pop_size}_{n}_cities.png', dpi=150, bbox_inches='tight')
            plt.show()

        # Plot average fitness over generations for all three populations
        print("\n" + "=" * 70)
        print("PLOTTING FITNESS EVOLUTION")
        print("=" * 70)
    
        plt.figure(figsize=(12, 7))

        colors = ['blue', 'red', 'green']
        for i, pop_size in enumerate(population_sizes):
            avg_fitness = all_results[pop_size]['avg_fitness_history']
            generations = range(len(avg_fitness))
            plt.plot(generations, avg_fitness, color=colors[i], linewidth=2,
                     label=f'Population Size: {pop_size}', alpha=0.8)
        
        plt.xlabel('Generation', fontsize=13)
        plt.ylabel('Average Best Tour Distance (km)', fontsize=13)
        plt.title('GA Fitness Evolution: Average Best Distance per Generation\n(Averaged over 20 runs)',
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'ga_fitness_evolution_{n}_cities.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✓ Saved fitness evolution plot as 'ga_fitness_evolution_{n}_cities.png'")

        # Summary comparison table
        print("\n" + "=" * 70)
        print("SUMMARY TABLE - COMPARISON OF POPULATION SIZES")
        print("=" * 70)
        print(f"{'Pop Size':<12} {'Best (km)':<12} {'Worst (km)':<12} {'Mean (km)':<12} {'Std Dev':<12} {'Avg Time (s)':<15}")
        print("-" * 95)

        for pop_size in population_sizes:
            r = all_results[pop_size]
            print(f"{pop_size:<12} {r['best_distance']:<12.2f} {r['worst_distance']:<12.2f} "
                  f"{r['mean_distance']:<12.2f} {r['std_distance']:<12.2f} {r['avg_time']:<15.2f}")
    
        # Determine configuration
        print("\n" + "=" * 70)
        print("CONCLUSION")
        print("=" * 70)

        best_pop_size = min(population_sizes, key=lambda ps: all_results[ps]['mean_distance'])
        fastest_convergence = min(population_sizes,
                                  key=lambda ps: np.argmin(all_results[ps]['avg_fitness_history']
                                                           <= all_results[ps]['mean_distance']))
    
        print(f"\n✓ Best population size (by mean distance): {best_pop_size}")
        print(f"  - Mean distance: {all_results[best_pop_size]['mean_distance']:.2f} km")
        print(f"  - Best distance: {all_results[best_pop_size]['best_distance']:.2f} km")

        print(f"\n✓ Fastest convergence: Population size {fastest_convergence}")
    
        print("\n" + "=" * 70)
        print("GENETIC ALGORITHM COMPLETE")
        print("=" * 70)
        print("✓ Ran 20 trials for each of 3 population sizes")
        print("✓ Plots saved for best tours (3 plots)")
        print("✓ Fitness evolution plot created")
        print("✓ Statistics calculated (best, worst, mean, std)")
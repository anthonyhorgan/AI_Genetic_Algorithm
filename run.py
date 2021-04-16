from framework import *
import copy
from matplotlib import pyplot as plt
import os
from itertools import product


def train(algorithm, num_generations, verbose=True):
    fitness_scores = {"best": [], "mean": [], "worst": []}
    for generation in range(num_generations):
        algorithm.evaluate_population()
        algorithm.select()
        algorithm.mutate()
        algorithm.crossover()
        population = algorithm.get_population()
        fitness_scores["best"].append(population[0].fitness_score)
        fitness_scores["worst"].append(population[-1].fitness_score)
        fitness_scores["mean"].append(algorithm.mean_population_fitness())

        if verbose:
            print(f"Iteration {generation + 1} \tMean Fitness: {fitness_scores['mean'][-1]:.2f} \tBest: {population[0]}")

    return algorithm, fitness_scores


def plot_fitness(fitness_scores, output_path=None, show=False, title=None, include=["best", "worst", "mean"]):
    plt.figure()
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    x_values = [i for i in range(len(next(iter(fitness_scores.values()))))]
    for fitness_type, fitness_values in fitness_scores.items():
        if fitness_type in include:
            plt.plot(x_values, fitness_values, label=fitness_type)
    plt.legend()
    if title:
        plt.title(title)
    if show:
        plt.show()
    if output_path:
        plt.savefig(output_path)
    plt.close()


# run the main generative algorithms
def main1():
    print()
    print("Max One")
    max_one_alg = MaxOneGA(string_length=20, population_size=100, crossover_prob=0.8, mutation_prob=0.01, tournament_size=10)
    alg, fitness_scores = train(max_one_alg, num_generations=150, verbose=True)
    print(f"max_one best solution: {alg.get_population()[0]}\tfitness: {alg.get_population()[0].fitness_score}")
    plot_fitness(fitness_scores, show=False, include=["best", "mean"], title="max one", output_path=os.path.join("plots", "max_one_plot.png"))

    print()
    print("Match String")
    match_string = [random.randint(0, 1) for _ in range(20)]
    match_string_alg = MatchStringGA(target_string=match_string, valid_values=[0, 1], population_size=100, crossover_prob=0.8, mutation_prob=0.01, tournament_size=10)
    alg, fitness_scores = train(match_string_alg, num_generations=150, verbose=True)
    print(f"match string best solution: {alg.get_population()[0]}\tfitness: {alg.get_population()[0].fitness_score}")
    plot_fitness(fitness_scores, show=False, include=["best", "mean"], title="match string", output_path=os.path.join("plots", "match_string_plot.png"))

    print()
    print("Deceptive")
    deceptive_alg = DeceptiveMaxOneGA(string_length=20, population_size=100, crossover_prob=0.8, mutation_prob=0.01, tournament_size=10)
    alg, fitness_scores = train(deceptive_alg, num_generations=150, verbose=True)
    print(f"deceptive best solution: {alg.get_population()[0]}\tfitness: {alg.get_population()[0].fitness_score}")
    plot_fitness(fitness_scores, show=False, include=["best", "mean"], title="deceptive", output_path=os.path.join("plots", "deceptive_string_plot.png"))

    print()
    print("knapsack 1")
    knapsack_alg = KnapSackGA(value_list=[78, 35, 89, 36, 94, 75, 74, 79, 80, 16],
                              weight_list=[18, 9, 23, 20, 59, 61, 70, 75, 76, 30],
                              max_weight=103,
                              population_size=100,
                              crossover_prob=0.8,
                              mutation_prob=0.01,
                              tournament_size=10,
                              num_elite=1)
    # print(knapsack_alg.get_population())
    alg, fitness_scores = train(knapsack_alg, num_generations=300, verbose=True)
    print(f"best solution: {alg.get_population()[0]}\tfitness: {alg.get_population()[0].fitness_score}")
    plot_fitness(fitness_scores, show=False, include=["best", "mean"], title="knapsack 103", output_path=os.path.join("plots", "knapsack_103.png"))

    print()
    print("knapsack 2")
    knapsack_alg = KnapSackGA(value_list=[78, 35, 89, 36, 94, 75, 74, 79, 80, 16],
                              weight_list=[18, 9, 23, 20, 59, 61, 70, 75, 76, 30],
                              max_weight=156,
                              population_size=100,
                              crossover_prob=0.8,
                              mutation_prob=0.01,
                              tournament_size=10,
                              num_elite=1)
    # print(knapsack_alg.get_population())
    alg, fitness_scores = train(knapsack_alg, num_generations=300, verbose=True)
    print(f"best solution: {alg.get_population()[0]}\tfitness: {alg.get_population()[0].fitness_score}")
    plot_fitness(fitness_scores, show=False, title="knapsack 156", include=["best", "mean"], output_path=os.path.join("plots", "knapsack_156.png"))


def main2():
    for algorithm in [KnapSackGA, KnapSackGAUniform, KnapSackGASwap, KnapSackGAUniformSwap]:
        print(algorithm.__name__)
        print("\tknapsack 1")
        knapsack_alg = KnapSackGA(value_list=[78, 35, 89, 36, 94, 75, 74, 79, 80, 16],
                                  weight_list=[18, 9, 23, 20, 59, 61, 70, 75, 76, 30],
                                  max_weight=103,
                                  population_size=100,
                                  crossover_prob=0.8,
                                  mutation_prob=0.01,
                                  tournament_size=10,
                                  num_elite=1)
        # print(knapsack_alg.get_population())
        alg, fitness_scores = train(knapsack_alg, num_generations=300, verbose=False)
        print(f"\tbest solution: {alg.get_population()[0]}\tfitness: {alg.get_population()[0].fitness_score}")
        # plot_fitness(fitness_scores, show=False, include=["best", "mean"], title="knapsack 103", output_path=os.path.join("plots", "knapsack_103.png"))

        print("\tknapsack 2")
        knapsack_alg = KnapSackGA(value_list=[78, 35, 89, 36, 94, 75, 74, 79, 80, 16],
                                  weight_list=[18, 9, 23, 20, 59, 61, 70, 75, 76, 30],
                                  max_weight=156,
                                  population_size=100,
                                  crossover_prob=0.8,
                                  mutation_prob=0.01,
                                  tournament_size=10,
                                  num_elite=1)
        # print(knapsack_alg.get_population())
        alg, fitness_scores = train(knapsack_alg, num_generations=300, verbose=False)
        print(f"\tbest solution: {alg.get_population()[0]}\tfitness: {alg.get_population()[0].fitness_score}")
        # plot_fitness(fitness_scores, show=False, title="knapsack 156", include=["best", "mean"], output_path=os.path.join("plots", "knapsack_156.png"))


if __name__ == "__main__":
    main2()


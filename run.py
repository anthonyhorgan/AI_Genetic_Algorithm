from framework import *
import copy
from matplotlib import pyplot as plt


def main1():
    ga = MaxOneGA(4, 10, crossover_prob=1, mutation_prob=0.0)
    init_pop = copy.deepcopy(ga.population)
    ga.crossover()
    pop = ga.population
    for initial, crossed in zip(init_pop, pop):
        print(initial, crossed)
    print()
    for initial in init_pop:
        print(initial)
    print()
    for entry in pop:
        print(entry)
    l = [1,2,3,4,5,6,7,8]


def fitness(individual):
    return sum(individual, 1)


def train_debug(algorithm, num_generations):
    fitness_scores = []
    print("population")
    for individual in algorithm.get_population():
        print(individual)
    print()
    for generation in range(num_generations):
        algorithm.evaluate_population()
        # stop training loop when at least one member of the population reaches the optimal fitness score
        # if gen_alg.get_population()[0].fitness_score >= optimal_fitness:
        #     break
        population_fitness = algorithm.mean_population_fitness()
        algorithm.select()
        algorithm.mutate()
        algorithm.crossover()
        fitness_scores.append(population_fitness)
        print(f"Iteration {generation + 1}\tMean Fitness: {population_fitness:.2f}")

    return algorithm, fitness_scores


def train(algorithm, num_generations, verbose=True):
    fitness_scores = {"best": [], "mean": [], "worst": []}
    for generation in range(num_generations):
        algorithm.evaluate_population()
        # stop training loop when at least one member of the population reaches the optimal fitness score
        # if gen_alg.get_population()[0].fitness_score >= optimal_fitness:
        #     break
        algorithm.select()
        algorithm.mutate()
        algorithm.crossover()
        population = algorithm.get_population()
        fitness_scores["best"].append(population[0].fitness_score)
        fitness_scores["worst"].append(population[-1].fitness_score)
        fitness_scores["mean"].append(algorithm.mean_population_fitness())

        if verbose:
            print(f"Iteration {generation + 1} \tMean Fitness: {fitness_scores['mean'][-1]:.2f}")

    return algorithm, fitness_scores


def train_maxone(num_generations):
    gen_alg = MaxOneGA(100, 20, crossover_prob=0.8, mutation_prob=0.01, tournament_size=10)
    optimal_fitness = 20
    fitness_scores = []
    for generation in range(num_generations):
        gen_alg.evaluate_population()
        # stop training loop when at least one member of the population reaches the optimal fitness score
        # if gen_alg.get_population()[0].fitness_score >= optimal_fitness:
        #     break
        population_fitness = gen_alg.mean_population_fitness()
        gen_alg.select()
        gen_alg.mutate()
        gen_alg.crossover()
        fitness_scores.append(population_fitness)
        print(f"Iteration {generation + 1}\tMean Fitness: {population_fitness:.2f}")

    return gen_alg, fitness_scores


def plot_fitness(fitness_scores, output_path=None, show=False, include=["best", "worst", "mean"]):
    plt.figure()
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    x_values = [i for i in range(len(next(iter(fitness_scores.values()))))]
    for fitness_type, fitness_values in fitness_scores.items():
        if fitness_type in include:
            plt.plot(x_values, fitness_values, label=fitness_type)
    plt.legend()
    if show:
        plt.show()
    if output_path:
        plt.savefig(output_path)


def main():
    # TODO retest
    max_one_alg = MaxOneGA(string_length=20, population_size=100, crossover_prob=0.8, mutation_prob=0.01, tournament_size=20)
    # individual = ListIndividual.empty(chrom_length=20, valid_values=[0, 1])
    # individual[:] = [1 for _ in range(20)]
    # print(max_one_alg.fitness(individual))
    # alg, scores = train(max_one_alg, 100, verbose=True)
    # alg, scores = train_debug(max_one_alg, n_gen)
    # plot_fitness(scores, show=True)

    # print()
    # TODO valid values
    match_string = [random.randint(0, 1) for _ in range(20)]
    match_string_alg = MatchStringGA(target_string=match_string, valid_values=[0, 1], population_size=100, crossover_prob=0.8, mutation_prob=0.01, tournament_size=10)
    # ag, scores = train(match_string_alg, 100, verbose=True)
    # plot_fitness(scores, show=True)
    #
    # print()
    # deceptive_alg = DeceptiveMaxOneGA(string_length=20, population_size=1000, crossover_prob=0.8, mutation_prob=0.01, tournament_size=10)
    # alg, scores = train(deceptive_alg, n_gen, verbose=True)
    # plot_fitness(scores, show=True)

    # knapsack_alg = KnapSackGA(value_list=[78, 35, 89, 36, 94, 75, 74, 79, 80, 16],
    #                           weight_list=[18, 9, 23, 20, 59, 61, 70, 75, 76, 30],
    #                           max_weight=103,
    #                           population_size=1000, crossover_prob=0.8, mutation_prob=0.01, tournament_size=50)
    knapsack_alg = KnapSackGA(value_list=[78, 35, 89, 36, 94, 75, 74, 79, 80, 16],
                              weight_list=[18, 9, 23, 20, 59, 61, 70, 75, 76, 30],
                              max_weight=103,
                              population_size=1000,
                              crossover_prob=0.8,
                              mutation_prob=0.01,
                              tournament_size=50,
                              num_elite=1)
    # print(knapsack_alg.get_population())
    alg, fitness_scores = train(knapsack_alg, num_generations=200, verbose=True)
    print(f"best solution: {alg.get_population()[0]}\tfitness: {alg.get_population()[0].fitness_score}")
    plot_fitness(fitness_scores, show=True, include=["best", "mean"])

    knapsack_alg = KnapSackGA(value_list=[78, 35, 89, 36, 94, 75, 74, 79, 80, 16],
                              weight_list=[18, 9, 23, 20, 59, 61, 70, 75, 76, 30],
                              max_weight=156,
                              population_size=1000,
                              crossover_prob=0.8,
                              mutation_prob=0.01,
                              tournament_size=50,
                              num_elite=1)
    # print(knapsack_alg.get_population())
    alg, fitness_scores = train(knapsack_alg, num_generations=200, verbose=True)
    print(f"best solution: {alg.get_population()[0]}\tfitness: {alg.get_population()[0].fitness_score}")
    plot_fitness(fitness_scores, show=True, include=["best", "mean"])


if __name__ == "__main__":
    main()







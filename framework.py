import numpy as np
import math
import copy
import random
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt


class ListPopulation(list):
    '''
    population whose chromosomes are list objects
    '''
    population_size = None
    chrom_length = None
    values = None


# TODO add abstract class
class ListChromosome(list):
    '''
    A chromosome whose genes are stored in a list
    '''
    chrom_length = None
    valid_values = None
    fitness_score = None

    @staticmethod
    def random(chrom_length, valid_values):
        # returns ListChromosome object with random genes
        chromosome = ListChromosome([random.choice(valid_values) for _ in range(chrom_length)])
        chromosome.chrom_length = chrom_length
        chromosome.valid_values = valid_values
        return chromosome

    @staticmethod
    def empty(chrom_length, valid_values):
        # returns empty ListChromosome object
        chromosome = ListChromosome([None for _ in range(chrom_length)])
        chromosome.chrom_length = chrom_length
        chromosome.valid_values = valid_values
        return chromosome


class AbstractGA(ABC):
    '''
    abstract class for generative algorithms
    '''
    def __init__(self):
        self.population = None
        self.crossover_prob = None
        self.mutation_prob = None

    @abstractmethod
    def select(self):
        raise NotImplementedError

    @abstractmethod
    def crossover(self):
        raise NotImplementedError

    @abstractmethod
    def mutate(self):
        raise NotImplementedError

    @abstractmethod
    def fitness(self, chromosome):
        raise NotImplementedError

    @abstractmethod
    def evaluate_population(self):
        raise NotImplementedError

    def get_population(self):
        # returns population sorted by fitness
        self.evaluate_population()
        self.population.sort(key=lambda x: x.fitness_score, reverse=True)
        return self.population


class ListStringGA(AbstractGA, ABC):
    '''
    provides functionality for GA's using a list of strings as representation, one point crossover standard mutation, and tournament select
    '''
    population = None
    crossover_prob = None
    mutation_prob = None
    tournament_size = None
    target_string = None
    num_elite = 0

    def crossover(self):
        # perform one point crossover on a random subset of the population
        parent_pool = random.sample(self.population, int(math.floor(len(self.population) * self.crossover_prob)))
        while len(parent_pool) > 1:
            father = parent_pool.pop(0)
            mother = random.choice(parent_pool)
            parent_pool.remove(mother)
            # if np.random.rand() < self.crossover_prob:
            son = ListChromosome.empty(father.chrom_length, father.valid_values)
            daughter = ListChromosome.empty(father.chrom_length, father.valid_values)
            # perform crossover
            crossover_length = random.randint(0, father.chrom_length)
            son[:crossover_length] = father[:crossover_length]
            son[crossover_length:] = mother[crossover_length:]
            daughter[:crossover_length] = mother[:crossover_length]
            daughter[crossover_length:] = father[crossover_length:]
            father_index = self.population.index(father)
            self.population[father_index] = son
            mother_index = self.population.index(mother)
            self.population[mother_index] = daughter

    def mutate(self):
        # performs standard mutation on a random subset of the population
        for i in range(len(self.population)):
            chromosome = self.population[i]
            if np.random.rand() < self.mutation_prob:
                # select gene to flip
                gene_choice = random.choice([j for j in range(len(chromosome))])
                current_value = chromosome[gene_choice]
                candidate_values = copy.deepcopy(chromosome.valid_values)
                candidate_values.remove(current_value)
                value_choice = random.choice(candidate_values)
                chromosome[gene_choice] = value_choice
                # print(f"flipping gene no.{gene_choice} from {current_value} to {value_choice}")

    def select(self):
        # tournament selection with elitism
        self.evaluate_population()
        new_population = []
        population = self.get_population()
        for i in range(self.num_elite):
            new_population.append(population.pop(0))
        # tournament selection with replacement
        for tourney_counter in range(len(self.population) - self.num_elite):
            contestants = random.sample(self.population, self.tournament_size)
            contestants.sort(key=lambda x: x.fitness_score, reverse=True)
            winner = contestants[0]
            new_population.append(winner)

        self.population = new_population

    def evaluate_population(self):
        '''
        evaluate the fitness of each chromosome in the population
        '''
        for chromosome in self.population:
            chromosome.fitness_score = self.fitness(chromosome)

    def mean_population_fitness(self):
        running_total = 0
        self.evaluate_population()
        for chromosome in self.population:
            # assert chromosome.fitness_score, f"fitness score not computed for chromosome: {chromosome}"
            assert chromosome.fitness_score is not None, f"Error{chromosome}\t\t{chromosome.fitness_score}"
            running_total += chromosome.fitness_score
        mean = running_total / len(self.population)
        return mean


class MatchStringGA(ListStringGA):
    def __init__(self, target_string, valid_values, population_size, crossover_prob, mutation_prob, tournament_size):
        # NOTE target_string should be a list
        super(MatchStringGA, self).__init__()
        chrom_length = len(target_string)
        self.population = [ListChromosome.random(chrom_length, valid_values=valid_values) for _ in range(population_size)]  # randomly initialize population
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.target_string = target_string

    def fitness(self, chromosome):
        # fitness = number of genes in chromosome which match the target string
        correct_count = 0
        for actual, target in zip(chromosome, self.target_string):
            correct_count += (actual == target)
        return correct_count


class MaxOneGA(MatchStringGA):
    '''
    genetic algorithm to match a given bitstring
    '''
    def __init__(self, string_length, population_size, crossover_prob, mutation_prob, tournament_size):
        super(MaxOneGA, self).__init__(target_string=[1 for _ in range(string_length)],
                                       valid_values=[0, 1],
                                       population_size=population_size,
                                       crossover_prob=crossover_prob,
                                       mutation_prob=mutation_prob,
                                       tournament_size=tournament_size)


class DeceptiveMaxOneGA(MaxOneGA):
    def fitness(self, chromosome):
        '''
        the fitness of a chromosome is equal to the number of ones it has or is equal to 2 * length if it is all zeros
        :param chromosome: the chromosome to evaluate
        :return:
        '''
        if set(chromosome) == {0}:
            return len(self.target_string) * 2
        else:
            return super(DeceptiveMaxOneGA, self).fitness(chromosome)


class KnapSackGA(ListStringGA):
    '''
    Genetic algorithm for the knapsack problem
    '''
    def __init__(self, value_list, weight_list, max_weight, population_size, crossover_prob, mutation_prob,
                 tournament_size, num_elite=1):
        # NOTE target_string should be a list
        super(KnapSackGA, self).__init__()
        assert(len(value_list) == len(weight_list)), f"weight_list and value_list must be same length. Received lengths:" \
                                                     f" weight_list: {len(weight_list)} value_list:{len(value_list)}"
        chrom_length = len(value_list)
        # representation are lists of bits with 1 and 0 indicating whether or not to include an item in the sack
        self.population = [ListChromosome.random(chrom_length, valid_values=[0, 1]) for _ in range(population_size)]
        self.num_elite = num_elite
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.max_weight = max_weight
        self.weight_list = weight_list
        self.value_list = value_list

    def crossover(self):
        # perform uniform crossover on a random subset of the population
        parent_pool = random.sample(self.population, int(math.floor(len(self.population) * self.crossover_prob)))
        while len(parent_pool) > 1:
            father = parent_pool.pop(0)
            mother = random.choice(parent_pool)
            parent_pool.remove(mother)

            # create new chromosome objects
            son = ListChromosome.empty(father.chrom_length, father.valid_values)
            daughter = ListChromosome.empty(father.chrom_length, father.valid_values)
            # for each gene (entry in the list), there is a 50% chance that the son will receive the fathers gene and
            # a 50% chance that it will receive the mothers gene
            for i, (father_gene, mother_gene) in enumerate(zip(father, mother)):
                if np.random.rand() < 0.5:
                    son[i] = mother_gene
                    daughter[i] = father_gene
                else:
                    son[i] = father_gene
                    daughter[i] = mother_gene

            father_index = self.population.index(father)
            self.population[father_index] = son
            mother_index = self.population.index(mother)
            self.population[mother_index] = daughter

    def mutate(self):
        # 50% to perform standard mutation (flip a bit)
        # 50% to take an item out of the sack and replace it with another one
        for i in range(len(self.population)):
            chromosome = self.population[i]
            if np.random.rand() < self.mutation_prob:
                if np.random.rand() < 0.5:
                    print(f"flip mutation")
                    # perform standard mutation
                    gene_choice = random.choice([j for j in range(len(chromosome))])  # select gene to flip
                    current_value = chromosome[gene_choice]
                    if current_value == 0:
                        chromosome[gene_choice] = 1
                    else:
                        chromosome[gene_choice] = 0
                else:
                    print(f"swap mutation")
                    # perform swap mutation
                    # take one item out and replace it with another item
                    if len(set(chromosome)) == 1:
                        # if either no items or all items are included then skip
                        continue

                    zero_indices = [i for i in range(len(chromosome)) if chromosome[i] == 0]
                    one_indices = [i for i in range(len(chromosome)) if chromosome[i] == 1]
                    zero_choice = random.choice(zero_indices)
                    one_choice = random.choice(one_indices)
                    chromosome[zero_choice] = 1
                    chromosome[one_choice] = 0

    def fitness(self, chromosome):
        # if total weight of items < max weight, then fitness = the sum of values of all items in the sacke
        # if total weight of items > max_weight, the fitness = 0
        included_items = []
        for idx, is_included in enumerate(chromosome):
            if is_included:
                included_items.append(idx)
        total_value = sum([self.value_list[item_idx] for item_idx in included_items])
        total_weight = sum([self.weight_list[item_idx] for item_idx in included_items])
        if total_weight > self.max_weight:
            return 0
        else:
            return total_value




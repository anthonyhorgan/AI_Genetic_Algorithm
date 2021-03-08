import numpy as np
from abc import ABC, abstractmethod


class AbstractGA(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def select(self):
        raise NotImplementedError

    @abstractmethod
    def select(self):
        raise NotImplementedError

    @abstractmethod
    def crossover(self):
        raise NotImplementedError

    @abstractmethod
    def mutate(self):
        raise NotImplementedError

    def train(self, num_generations):
        pass
        #while
        #   evaluate
        #   select
        #   mutate
        #   crossover

import numpy as np
from torch import optim
from src.models import *
from src.genetic_algorithm import seed_everything
import torch
import random
import os
from dataclasses import dataclass
import gc  # Import gc module for garbage collection

@dataclass
class CuckooSearchConfig:
    population_size: int = 20
    num_epochs: int = 10
    pa: float = 0.25  # Probability of a cuckoo abandoning its nest (nest replacement probability)
    alpha: float = 0.01  # Step size scaling factor

class Nest:
    def __init__(self):
        self.name = ''.join(map(str, np.random.randint(0, 9, size=7).tolist()))
        self.num_epochs_base = np.random.choice(np.arange(60, 300))
        self.hidden_size = np.random.choice([2 ** power for power in range(2, 10)])
        self.num_layers = np.random.choice(np.arange(2, 15))
        self.learning_rate = round(np.random.random(), 2)

        self.loss = np.inf
        self.fitness = None

    def __repr__(self):
        """
        For convenience only.
        """
        string = 'Nest ' + self.name + f' with the loss of {self.loss:.4}' + f' and {self.num_epochs_base} epochs:\n'
        string = string + f'learning_rate = {self.learning_rate:.4}, '
        string = string + f'num_layers = {self.num_layers}, ' + f'hidden_size = {self.hidden_size}'
        return string

@dataclass
class PopulationCS:
    def __init__(self, config: CuckooSearchConfig):
        self.nests = [Nest() for _ in range(config.population_size)]
        self.best_nest = None

class CuckooSearch:
    def __init__(self, optimized_block, criterion,
                 population: PopulationCS, config: CuckooSearchConfig,
                 device, verbose=True, seed: int = 77):
        self.optimized_block = optimized_block
        self.criterion = criterion
        self.population = population
        self.config = config
        self.device = device
        self.verbose = verbose
        self.seed = seed

        self.val_loss_history = []

    def fit(self, X_val, y_val):
        saved_values=[]
        for epoch in range(self.config.num_epochs):
            self.evaluate(X_val, y_val, self.population)
            self.val_loss_history.append(self.population.best_nest.loss)

            for i, nest in enumerate(self.population.nests):
                new_nest = self.generate_new_nest(nest)
                if new_nest.loss < nest.loss:
                    self.population.nests[i] = new_nest

            self.abandon_nests()

            if self.verbose:
                clear_output(wait=True)
                print(f"Epoch: {epoch + 1}")

                plot_metric(self.criterion.__class__.__name__,
                            val_metric=self.val_loss_history)

                print(f'{self.population.best_nest}')
                saved_values.append(self.population.best_nest)
        print(saved_values)

    def evaluate(self, X_val, y_val, population):
        losses = []

        for nest in population.nests:
            gc.collect()
            torch.cuda.empty_cache()

            if self.optimized_block == 'LSTM':
                seed_everything(self.seed)
                model = LSTM(input_size=X_val.shape[2],
                             hidden_size=int(nest.hidden_size),
                             num_layers=nest.num_layers).to(self.device)

            elif self.optimized_block == 'GRU':
                seed_everything(self.seed)
                model = GRU(input_size=X_val.shape[2],
                            hidden_size=int(nest.hidden_size),
                            num_layers=nest.num_layers).to(self.device)

            else:
                raise ValueError('Only LSTM and GRU blocks are available for optimization.')

            optimizer = optim.Adam(model.parameters(), lr=nest.learning_rate)

            seed_everything(self.seed)

            train(model, self.criterion, optimizer, device, X_val, y_val, nest.num_epochs_base,
                  verbose=False, return_loss_history=False, compute_test_loss=False)

            nest.loss = predict(model, X_val, y_val, self.criterion, device)

            losses.append(nest.loss)

            del model

        self.update_fitness(population)

    def update_fitness(self, population):
        losses = [nest.loss for nest in population.nests]
        for nest in population.nests:
            nest.fitness = self.normalize(nest.loss, min(losses), max(losses))

        population.best_nest = min(population.nests, key=lambda x: x.loss)

    def normalize(self, z, loss_best, loss_worst) -> float:
        return (z - loss_worst) / (loss_best - loss_worst)

    def generate_new_nest(self, nest):
        new_nest = Nest()
        new_nest.hidden_size = nest.hidden_size + self.config.alpha * np.random.randn()
        new_nest.hidden_size = int(np.clip(new_nest.hidden_size, 2 ** 3, 2 ** 9))

        new_nest.num_layers = nest.num_layers + self.config.alpha * np.random.randn()
        new_nest.num_layers = np.clip(new_nest.num_layers, 2, 14)

        new_nest.learning_rate = nest.learning_rate + self.config.alpha * np.random.randn()
        new_nest.learning_rate = np.clip(new_nest.learning_rate, 0.001, 1)

        new_nest.num_epochs_base = nest.num_epochs_base + self.config.alpha * np.random.randn()
        new_nest.num_epochs_base = np.clip(new_nest.num_epochs_base, 10, 300)

        return new_nest

    def abandon_nests(self):
        sorted_nests = sorted(self.population.nests, key=lambda x: x.fitness, reverse=True)
        num_to_abandon = int(self.config.pa * len(sorted_nests))
        for i in range(num_to_abandon):
            new_nest = Nest()
            self.population.nests[-(i + 1)] = new_nest

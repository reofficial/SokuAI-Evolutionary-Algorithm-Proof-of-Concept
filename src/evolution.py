# evolution.py
import torch
import copy
import random

def select_elites(population, fitnesses, elite_fraction=0.2):
    """
    Selects the top-performing agents from the population.
    Returns a list of deep copies of the elite agents.
    """
    num_elites = max(1, int(len(population) * elite_fraction))
    sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
    elites = [copy.deepcopy(population[i]) for i in sorted_indices[:num_elites]]
    return elites

def crossover(parent1, parent2, crossover_rate=0.5):
    """
    Creates a child state dict by combining parameters from two parents.
    Each parameter element is chosen from parent1 with probability crossover_rate; otherwise from parent2.
    """
    state_dict1 = parent1.state_dict()
    state_dict2 = parent2.state_dict()
    child_state = {}
    for key in state_dict1.keys():
        p1 = state_dict1[key]
        p2 = state_dict2[key]
        mask = torch.rand_like(p1) < crossover_rate
        child_param = torch.where(mask, p1, p2)
        child_state[key] = child_param.clone()
    return child_state

def mutate(state_dict, mutation_rate=0.02, mutation_scale=0.1):
    """
    Mutates the state dict by adding Gaussian noise to parameters with probability mutation_rate.
    """
    for key in state_dict:
        param = state_dict[key]
        mask = torch.rand_like(param) < mutation_rate
        noise = torch.randn_like(param) * mutation_scale
        state_dict[key] = param + mask.float() * noise
    return state_dict

def create_offspring(elites, population_size, agent_constructor, 
                     crossover_rate=0.5, mutation_rate=0.02, mutation_scale=0.1):
    """
    Generates a new population from the elites.
    agent_constructor: a function that returns a new RLAgent with the desired configuration.
    """
    new_population = []
    # Retain the elites (elitism)
    for elite in elites:
        new_population.append(elite)
    # Generate new agents until the new population is of the desired size
    while len(new_population) < population_size:
        parent1 = random.choice(elites)
        parent2 = random.choice(elites)
        child_state = crossover(parent1, parent2, crossover_rate)
        child_state = mutate(child_state, mutation_rate, mutation_scale)
        child_agent = agent_constructor()
        child_agent.load_state_dict(child_state)
        new_population.append(child_agent)
    return new_population

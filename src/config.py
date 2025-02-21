# config.py
# ----------------------------
# Hybrid RL + Evolutionary Algorithm Trainer Configuration

# Population settings
population_size = 10             # Number of agents in the population
episodes_per_generation = 5      # Self-play episodes per generation
generation_count = 1000           # Total number of generations

# RL hyperparameters
learning_rate = 1e-4             # Learning rate for Adam
gamma = 0.99                   # Discount factor (not used extensively if using a final reward)
rl_update_frequency = 1          # RL update after every episode

# EA hyperparameters
elite_fraction = 0.1             # Fraction of top agents to keep as elites
crossover_rate = 0.5             # Probability for choosing a parent's parameter during crossover
mutation_rate = 0.3             # Probability per parameter element to mutate
mutation_scale = 0.1             # Scale (std. dev.) of mutation noise

# Self-play / Episode settings
episode_duration = 10            # Seconds per self-play episode

# Neural network architecture
frame_stack = 10
target_size = (128, 128)
num_actions = 10
feature_dim = 256

# Fitness function parameters (can be modified as needed)
damage_dealt_weight = 0.6
damage_taken_weight = 0.4

# Other settings
print_interval = 1             # How often to print progress (in generations)

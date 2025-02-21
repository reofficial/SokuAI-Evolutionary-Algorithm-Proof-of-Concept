# hybrid_trainer.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import time
import os

# Import configuration parameters
from config import (population_size, episodes_per_generation, generation_count,
                    learning_rate, episode_duration,
                    frame_stack, target_size, num_actions, feature_dim,
                    elite_fraction, crossover_rate, mutation_rate, mutation_scale,
                    damage_dealt_weight, damage_taken_weight)

# Import our agent and evolution modules
from rl_agent import RLAgent
from evolution import select_elites, create_offspring

# Import your game-specific modules (assumed to exist)
from frame_processor import FrameProcessor
from input_controller import InputController
from macro_controller import GameMacro
from screen_capture import get_game_region, capture_screen

# Set the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use a single checkpoint filename for saving and loading
CHECKPOINT_FILENAME = "hybrid_checkpoint.pth"

# ----------------------------
# Agent Constructor Function
# ----------------------------
def create_agent():
    agent = RLAgent(frame_stack=frame_stack, target_size=target_size, 
                    num_actions=num_actions, feature_dim=feature_dim).to(DEVICE)
    return agent

# ----------------------------
# Fitness Function
# ----------------------------
def compute_fitness(health_history, is_player1=True):
    """
    Compute a fitness value based on the health history recorded during an episode.
    For player1, fitness = (damage dealt * weight) - (damage taken * weight).
    For player2, the roles are reversed.
    health_history: list of dicts with keys 'player' and 'opponent'
    """
    if len(health_history) < 2:
        return 0.0
    if is_player1:
        your_health = [h['player'] for h in health_history]
        their_health = [h['opponent'] for h in health_history]
    else:
        your_health = [h['opponent'] for h in health_history]
        their_health = [h['player'] for h in health_history]
    damage_dealt = their_health[0] - their_health[-1]
    damage_taken = your_health[0] - your_health[-1]
    fitness = damage_dealt * damage_dealt_weight - damage_taken * damage_taken_weight
    return fitness

# ----------------------------
# RL Update Function (Actor–Critic)
# ----------------------------
def rl_update(agent, optimizer, log_probs, values, entropies, reward):
    """
    Performs a gradient descent step on an agent using an actor–critic loss.
    Here we simply assign the same reward to every time step.
    """
    returns = torch.full_like(torch.stack(values), reward, device=DEVICE)
    advantages = returns - torch.stack(values)
    policy_loss = -(torch.stack(log_probs) * advantages.detach()).mean()
    value_loss = F.mse_loss(torch.stack(values), returns)
    entropy_bonus = -0.001 * torch.stack(entropies).mean()
    loss = policy_loss + value_loss + entropy_bonus
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# ----------------------------
# Self‑Play Episode Function
# ----------------------------
def self_play_episode(agent1, agent2, frame_processor, controller1, controller2, game_region):
    """
    Runs one self‑play episode (match) between agent1 and agent2.
    Returns:
      - RL data for agent1: (log_probs, values, entropies, reward)
      - RL data for agent2: (log_probs, values, entropies, reward)
      - The common health_history recorded during the match.
    """
    # Reset frame buffer and game state
    frame_processor.frame_buffer.clear()
    macro = GameMacro()
    controller1.release_all()
    controller2.release_all()
    macro.reset_match()
    
    # Containers for RL data for each agent
    log_probs1, values1, entropies1 = [], [], []
    log_probs2, values2, entropies2 = [], [], []
    
    health_history = []  # Records health info from each time step
    
    episode_start = time.time()
    done = False
    while not done:
        raw_frame = capture_screen(game_region)
        if raw_frame is None:
            continue
        # Preprocess frame and update the shared buffer
        timestamp_before = time.perf_counter_ns()
        processed_frame = frame_processor.preprocess(raw_frame)
        frame_processor.update_frame_buffer(processed_frame)
        
        # Get the current state tensor (shape: [1, channels, H, W])
        state_tensor = frame_processor.get_processed_input_tensor()
        
        # Detect health information from the raw frame
        health_data = frame_processor.detect_health(raw_frame)
        health_history.append(health_data)
        
        # Prepare extra features for each agent
        extra_features1 = torch.tensor([[1.0, 0.0, health_data['player'], health_data['opponent']]], 
                                       dtype=torch.float32, device=DEVICE)
        extra_features2 = torch.tensor([[0.0, 1.0, health_data['opponent'], health_data['player']]], 
                                       dtype=torch.float32, device=DEVICE)
        
        # Forward pass for agent1
        policy_logits1, value1 = agent1(state_tensor, extra_features1)
        action_probs1 = torch.sigmoid(policy_logits1)
        dist1 = torch.distributions.Bernoulli(probs=action_probs1)
        action1 = dist1.sample()
        log_prob1 = dist1.log_prob(action1).sum()
        entropy1 = dist1.entropy().sum()
        log_probs1.append(log_prob1)
        values1.append(value1.squeeze())
        entropies1.append(entropy1)
        
        # Forward pass for agent2
        policy_logits2, value2 = agent2(state_tensor, extra_features2)
        action_probs2 = torch.sigmoid(policy_logits2)
        dist2 = torch.distributions.Bernoulli(probs=action_probs2)
        action2 = dist2.sample()
        log_prob2 = dist2.log_prob(action2).sum()
        entropy2 = dist2.entropy().sum()
        log_probs2.append(log_prob2)
        values2.append(value2.squeeze())
        entropies2.append(entropy2)
        
        # Convert actions to NumPy arrays and send inputs
        actions_np1 = action1.cpu().detach().numpy().flatten()
        actions_np2 = action2.cpu().detach().numpy().flatten()
        controller1.send_inputs(actions_np1)
        controller2.send_inputs(actions_np2)
        
        timestamp_after = time.perf_counter_ns()
        input_send_latency_ns = timestamp_after - timestamp_before
        #print(f"Input send latency: {input_send_latency_ns/1000000} ms")
        
        # End the episode after the specified duration
        if time.time() - episode_start > episode_duration:
            done = True
    
    # End of episode: release any held keys
    controller1.release_all()
    controller2.release_all()
    
    # Compute rewards (fitness) for each agent based on the health history
    reward1 = compute_fitness(health_history, is_player1=True)
    reward2 = compute_fitness(health_history, is_player1=False)
    
    return (log_probs1, values1, entropies1, reward1), (log_probs2, values2, entropies2, reward2), health_history

# ----------------------------
# Main Training Loop
# ----------------------------
def main():
    # Get the game region once (assumes the game is running)
    game_region = get_game_region()
    if not game_region:
        print("Game region not found!")
        return
    
    # Initialize the shared frame processor
    frame_processor = FrameProcessor(target_size=target_size, frame_stack=frame_stack)
    
    # Create input controllers for the two players
    controller1 = InputController(1)
    controller2 = InputController(2)
    
    # Load checkpoint if it exists; otherwise, initialize a new population.
    if os.path.exists(CHECKPOINT_FILENAME):
        print(f"Checkpoint file '{CHECKPOINT_FILENAME}' found. Loading checkpoint...")
        checkpoint = torch.load(CHECKPOINT_FILENAME, map_location=DEVICE)
        start_generation = checkpoint.get("generation", 0)
        # Create new agents and load their state dictionaries
        population = [create_agent() for _ in range(population_size)]
        for i, agent in enumerate(population):
            agent.load_state_dict(checkpoint["population"][i])
        print(f"Resuming from generation {start_generation+1}.")
    else:
        start_generation = 0
        population = [create_agent() for _ in range(population_size)]
    
    # Create optimizers for the population
    optimizers = [optim.Adam(agent.parameters(), lr=learning_rate) for agent in population]
    
    current_generation = start_generation

    try:
        # For each generation, we accumulate fitness for every agent.
        for generation in range(start_generation, generation_count):
            current_generation = generation  # Update the current generation index
            print(f"\n--- Generation {generation+1} ---")
            population_fitness = [0.0 for _ in range(population_size)]
            
            # Run a set number of self-play episodes for this generation.
            for ep in range(episodes_per_generation):
                print(f"  Episode {ep+1} of Generation {generation+1}")
                # Pair agents randomly for self‑play.
                indices = list(range(population_size))
                random.shuffle(indices)
                pairs = []
                for i in range(0, len(indices)-1, 2):
                    pairs.append((indices[i], indices[i+1]))
                
                # Run self-play episodes for each pair.
                for idx1, idx2 in pairs:
                    agent1 = population[idx1]
                    agent2 = population[idx2]
                    optimizer1 = optimizers[idx1]
                    optimizer2 = optimizers[idx2]
                    
                    # Run the self-play episode between this pair.
                    (logp1, vals1, ents1, rew1), (logp2, vals2, ents2, rew2), health_history = \
                        self_play_episode(agent1, agent2, frame_processor, controller1, controller2, game_region)
                    
                    # Perform RL updates for both agents.
                    loss1 = rl_update(agent1, optimizer1, logp1, vals1, ents1, rew1)
                    loss2 = rl_update(agent2, optimizer2, logp2, vals2, ents2, rew2)
                    
                    print(f"    Pair ({idx1}, {idx2}) -> Agent {idx1}: loss {loss1:.4f}, reward {rew1:.2f} | "
                          f"Agent {idx2}: loss {loss2:.4f}, reward {rew2:.2f}")
                    
                    # Accumulate fitness (sum rewards over episodes).
                    population_fitness[idx1] += rew1
                    population_fitness[idx2] += rew2
            
            # Average fitness over episodes per agent.
            avg_fitness = [fit / episodes_per_generation for fit in population_fitness]
            print("  Average fitness per agent this generation:")
            for i, fit in enumerate(avg_fitness):
                print(f"    Agent {i}: {fit:.2f}")
            
            # ----------------------------
            # Evolutionary Algorithm Update
            # ----------------------------
            elites = select_elites(population, avg_fitness, elite_fraction)
            print(f"  Selected {len(elites)} elites for reproduction.")
            # Create new population using the elites and our genetic operators.
            population = create_offspring(elites, population_size, create_agent, 
                                          crossover_rate, mutation_rate, mutation_scale)
            # Reinitialize optimizers for the new population.
            optimizers = [optim.Adam(agent.parameters(), lr=learning_rate) for agent in population]
            
            # Save a checkpoint at the end of every generation.
            checkpoint = {
                'population': [agent.state_dict() for agent in population],
                'generation': generation + 1  # Next generation number.
            }
            torch.save(checkpoint, CHECKPOINT_FILENAME)
            print(f"  Saved checkpoint to '{CHECKPOINT_FILENAME}'.")
        
        print("Training complete.")
    
    except KeyboardInterrupt:
        # Save checkpoint upon Ctrl+C interruption.
        print("\nKeyboardInterrupt detected. Saving checkpoint before exiting...")
        checkpoint = {
            'population': [agent.state_dict() for agent in population],
            'generation': current_generation + 1  # Next generation number.
        }
        torch.save(checkpoint, CHECKPOINT_FILENAME)
        print(f"Checkpoint saved to '{CHECKPOINT_FILENAME}'. Exiting gracefully.")

if __name__ == "__main__":
    # Disable pydirectinput's pause delay (if using pydirectinput in your controllers)
    import pydirectinput
    pydirectinput.PAUSE = 0.0
    main()

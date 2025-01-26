import torch
import pickle
import time
import cv2
import numpy as np
from screen_capture import get_game_region, capture_screen
from frame_processor import FrameProcessor
from input_controller import InputController
from macro_controller import GameMacro
import pydirectinput
import os
import sys

# Set default device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reset_game(controller_p1, controller_p2):
    macro = GameMacro()
    """Reset game state between matches"""
    controller_p1.release_all()
    controller_p2.release_all()
    macro.reset_match()

class GeneticAI:
    def __init__(self, population_size=100,  # Increased population
                 input_shape=(128, 128, 4*4),
                 output_size=10,
                 hidden_layers=[256, 128, 64, 32],  # Deeper architecture
                 activation_sequence=['tanh', 'sin', 'relu', 'tanh'],
                 mutation_rate=0.20):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.input_shape = input_shape
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.activation_map = {  # Map strings to actual functions
            'tanh': torch.tanh,
            'sin': torch.sin,
            'relu': torch.relu,
            'sigmoid': torch.sigmoid
        }
        self.activation_sequence = [self.activation_map[a] for a in activation_sequence]
        self.fitness_scores = torch.zeros(population_size, device=DEVICE)
        self.elite_keep = 4
        self.current_generation = 0
        self.action_history = []
        input_size = np.prod(input_shape) + 2 + 2
        self.population = []
        for _ in range(population_size):
            net = self.create_individual(input_size)
            self.population.append(net)
            torch.cuda.empty_cache()

    def create_individual(self, input_size):
        """Create network with activation-aware initialization"""
        weights = {}
        prev_size = input_size
        
        for i, layer_size in enumerate(self.hidden_layers):
            # Scale initialization for sin layers
            init_scale = 0.05 if self.activation_sequence[i%len(self.activation_sequence)] == torch.sin else 0.1
            weights[f'w{i+1}'] = torch.randn(prev_size, layer_size, device=DEVICE) * init_scale
            prev_size = layer_size
        
        weights[f'w{len(self.hidden_layers)+1}'] = (
            torch.randn(prev_size, self.output_size, device=DEVICE) * 0.1
        )
        return weights

    def predict(self, individual, processed_input, health_data, is_player1):
        """Safer forward pass with activation sequence"""
        with torch.no_grad():
            # Input conversion with null checks
            flat_input = torch.tensor(
                processed_input.flatten(), 
                dtype=torch.float32,
                device=DEVICE
            )
            player_pos = torch.tensor(
                [1.0, 0.0] if is_player1 else [0.0, 1.0],
                dtype=torch.float32,
                device=DEVICE
            )
            health = torch.tensor(
                [health_data.get('player', 0.0), health_data.get('opponent', 0.0)],
                dtype=torch.float32,
                device=DEVICE
            )
            
            combined_input = torch.cat([flat_input, player_pos, health])
            
            x = combined_input
            for i in range(len(self.hidden_layers)):
                x = torch.matmul(x, individual[f'w{i+1}'])
                activation = self.activation_sequence[i % len(self.activation_sequence)]
                x = activation(x)
                # Add safety for extreme values
                x = torch.clamp(x, min=-10.0, max=10.0)
            
            output = torch.sigmoid(torch.matmul(x, individual[f'w{len(self.hidden_layers)+1}']))
            return output.cpu().numpy()

    def mutate(self, individual):
        """In-place mutation with PyTorch tensors"""
        with torch.no_grad():
            for key in individual:
                mask = torch.rand_like(individual[key], device=DEVICE) < self.mutation_rate
                individual[key] += mask * torch.randn_like(individual[key]) * 0.1
        return individual

    def crossover(self, parent1, parent2):
        """Crossover using PyTorch tensor operations"""
        child = {}
        for key in parent1:
            mask = torch.rand_like(parent1[key], device=DEVICE) > 0.5
            child[key] = parent1[key] * mask + parent2[key] * (~mask)
        return child

    def evolve_population(self):
        sorted_indices = torch.argsort(self.fitness_scores, descending=True)
        sorted_population = [self.population[i] for i in sorted_indices]
        new_population = []
        
        # Keep elites
        new_population.extend(sorted_population[:self.elite_keep])
        
        def select_parent():
            candidates = torch.randperm(len(sorted_population))[:4]
            return sorted_population[torch.argmax(self.fitness_scores[candidates])]
            
        # Breed new population
        while len(new_population) < self.population_size:
            parent1 = select_parent()
            parent2 = select_parent()
            
            child = self.crossover(parent1, parent2)
            new_population.append(self.mutate(child))
    
        self.population = new_population
        self.fitness_scores = torch.zeros(self.population_size, device=DEVICE)
        self.current_generation += 1

    def save_checkpoint(self, filename="ai_checkpoint.pkl"):
        """Save checkpoint with CPU tensors for compatibility"""
        cpu_population = []
        for ind in self.population:
            cpu_ind = {k: v.cpu() for k, v in ind.items()}
            cpu_population.append(cpu_ind)
            
        with open(filename, 'wb') as f:
            pickle.dump({
                'population': cpu_population,
                'generation': self.current_generation
            }, f)

    def load_checkpoint(self, filename="ai_checkpoint.pkl"):
        """Load checkpoint and move tensors to GPU"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.population = []
            for ind in data['population']:
                gpu_ind = {k: v.to(DEVICE) for k, v in ind.items()}
                self.population.append(gpu_ind)
            self.current_generation = data['generation']

# Rest of the code remains the same as original, except fitness calculation adjustments

def calculate_fitness(health_history, action_history, is_player1):
    """Modified to handle numpy arrays from PyTorch outputs"""
    if is_player1:
        your_health = [h['player'] for h in health_history]
        their_health = [h['opponent'] for h in health_history]
    else:
        your_health = [h['opponent'] for h in health_history]
        their_health = [h['player'] for h in health_history]
    
    damage_dealt = (their_health[0] - their_health[-1])
    damage_taken = (your_health[0] - your_health[-1])
    
    aggression = 0.0
    if len(their_health) > 1:
        health_changes = np.diff(their_health)
        aggression = np.mean(health_changes[health_changes > 0])
    
    # Convert action history to numpy if needed
    if isinstance(action_history[0], torch.Tensor):
        action_history = [a.cpu().numpy() for a in action_history]
    
    
    return (
        damage_dealt * 1.2 - damage_taken * 0.45    #higher reward for dealing damage, minor reward for getting damaged
    )

def train():
    game_region = get_game_region()
    if not game_region:
        return

    ai = GeneticAI(population_size=20)
    
    if os.path.exists("ai_checkpoint.pkl"):
        print("Previous checkpoint found!")
        choice = input("(L)oad / (D)elete / (N)ew? [L]: ").lower() or 'l'
        if choice == 'l':
            ai.load_checkpoint()
        elif choice == 'd':
            os.remove("ai_checkpoint.pkl")
            print("Checkpoint deleted.")
            
    frame_processor = FrameProcessor()
    
    try:
        while True:                
            print(f"\n--- Generation {ai.current_generation} ---")
            ai.action_history = []
            
            np.random.shuffle(ai.population)
            pairs = [(ai.population[i], ai.population[i+1]) 
                    for i in range(0, len(ai.population), 2)]
            
            fitness = []            
            for idx, (p1, p2) in enumerate(pairs):                 
                
                controller_p1 = InputController(1)
                controller_p2 = InputController(2)
                
                frame_processor.frame_buffer.clear()
                health_history = []
                action_history_p1 = []
                action_history_p2 = []
                
                reset_game(controller_p1, controller_p2)
                
                start_health = frame_processor.detect_health(capture_screen(game_region))
                if start_health['player'] < 0.8 or start_health['opponent'] < 0.8:
                    print("Reset failed! Manual intervention required.")
                    input("Press Enter after manually resetting the game...")
                
                start_time = time.time()
                
                while time.time() - start_time < 10:
                    raw_frame = capture_screen(game_region)
                    if raw_frame is None:
                        continue
                        
                    processed = frame_processor.preprocess(raw_frame)
                    frame_processor.update_frame_buffer(processed)
                    raw_health = frame_processor.detect_health(raw_frame)
                    
                    health_p1 = {
                        'player': raw_health['player'],
                        'opponent': raw_health['opponent']
                    }
                    health_p2 = {
                        'player': raw_health['opponent'],
                        'opponent': raw_health['player']
                    }
                    
                    if raw_health['player'] <= 0.01 or raw_health['opponent'] <= 0.01:
                        print("KO detected!")
                        break
                    
                    health_history.append(raw_health)
                    
                    processed_input = frame_processor.get_processed_input()
                    actions_p1 = ai.predict(p1, processed_input, health_p1, True)
                    actions_p2 = ai.predict(p2, processed_input, health_p2, False)
                    
                    action_history_p1.append(actions_p1)
                    action_history_p2.append(actions_p2)
                    
                    controller_p1.send_inputs(actions_p1)
                    controller_p2.send_inputs(actions_p2)
                
                try:
                    ai.fitness_scores[idx * 2] = calculate_fitness(
                        health_history, action_history_p1, True
                    )
                    ai.fitness_scores[idx * 2 + 1] = calculate_fitness(
                        health_history, action_history_p2, False
                    )
                    print(f"\nMatch {idx+1} Results:")
                    print(f"P1 Fitness: {ai.fitness_scores[idx*2]:.2f}")
                    print(f"P2 Fitness: {ai.fitness_scores[idx*2+1]:.2f}")
                    fitness.append(ai.fitness_scores[idx * 2].item())
                    fitness.append(ai.fitness_scores[idx * 2 + 1].item())
                except Exception as e:
                    print(f"Error calculating fitness: {str(e)}")
                    ai.fitness_scores[idx * 2] = 0.001
                    ai.fitness_scores[idx * 2 + 1] = 0.001
                
                controller_p1.release_all()
                controller_p2.release_all()
            
            print(f"Average fitness this generation: {np.mean(fitness):.3f}")
            print(f"Average top 4 fitness: {np.mean(np.sort(fitness)[-4:]):.3f}")
            print("Top AI-chans are now reproducing, please wait warmly...")
            ai.evolve_population()
            ai.save_checkpoint()
            
    except KeyboardInterrupt:
        print("\nTraining stopped by user. Final generation saved.")
        ai.save_checkpoint()

if __name__ == "__main__":
    pydirectinput.PAUSE = 0.0
    train()
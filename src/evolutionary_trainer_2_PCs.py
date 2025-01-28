# Add these new imports
import torch
import pickle
import time
import cv2
import numpy as np
from screen_capture import get_game_region, capture_screen
from frame_processor import FrameProcessor
from input_controller import InputController
from macro_controller import GameMacro
from evolutionary_trainer import GeneticAI, DEVICE, reset_game
import pydirectinput
import os
import sys
import socket
import json
import struct
import threading

# Add this class for network communication
class NetworkManager:
    def __init__(self, role='server', host='0.0.0.0', port=65432):
        self.role = role
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn = None
        self.lock = threading.Lock()
        
    def start_server(self):
        self.sock.bind((self.host, self.port))
        self.sock.listen()
        print(f"Server listening on {self.host}:{self.port}")
        self.conn, addr = self.sock.accept()
        print(f"Connected to {addr}")

    def connect_client(self):
        self.sock.connect((self.host, self.port))
        print(f"Connected to server at {self.host}:{self.port}")

    def send_data(self, data):
        with self.lock:
            # Convert tensors to serializable format
            serializable_data = self._convert_tensors(data)
            json_data = json.dumps(serializable_data)
            encoded = json_data.encode('utf-8')
            self.conn.sendall(struct.pack('>I', len(encoded)))
            self.conn.sendall(encoded)

    def receive_data(self):
        with self.lock:
            raw_len = self.recvall(4)
            if not raw_len:
                return None
            msg_len = struct.unpack('>I', raw_len)[0]
            data = json.loads(self.recvall(msg_len).decode('utf-8'))
            return self._restore_tensors(data)

    def _convert_tensors(self, obj):
        if isinstance(obj, dict):
            return {k: self._convert_tensors(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_tensors(v) for v in obj]
        elif torch.is_tensor(obj):
            return {'__tensor__': True, 'data': obj.cpu().tolist()}
        return obj

    def _restore_tensors(self, obj):
        if isinstance(obj, dict):
            if '__tensor__' in obj:
                return torch.tensor(obj['data'])
            return {k: self._restore_tensors(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._restore_tensors(v) for v in obj]
        return obj

    def recvall(self, n):
        data = bytearray()
        while len(data) < n:
            packet = self.conn.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

# Modified GeneticAI class
class DistributedGeneticAI(GeneticAI):
    def __init__(self, network_manager=None, **kwargs):
        super().__init__(**kwargs)
        self.network = network_manager
        
    def evolve_population_distributed(self, remote_fitness, remote_population):
        combined_fitness = torch.cat([self.fitness_scores, remote_fitness])
        combined_population = self.population + remote_population
        
        sorted_indices = torch.argsort(combined_fitness, descending=True)
        sorted_population = [combined_population[i] for i in sorted_indices]
        
        new_population = []
        new_population.extend(sorted_population[:self.elite_keep])
        
        while len(new_population) < self.population_size:
            parent1 = sorted_population[torch.randint(0, len(sorted_population), (1,))]
            parent2 = sorted_population[torch.randint(0, len(sorted_population), (1,))]
            child = self.crossover(parent1[0], parent2[0])
            new_population.append(self.mutate(child))
        
        self.population = new_population[:self.population_size]
        self.fitness_scores = torch.zeros(self.population_size, device=DEVICE)
        self.current_generation += 1

    def tensor_to_cpu(self, tensor_dict):
        return {k: v.cpu().tolist() if torch.is_tensor(v) else v 
                for k, v in tensor_dict.items()}

    def tensor_to_device(self, tensor_dict):
        return {k: torch.tensor(v, device=DEVICE) if isinstance(v, list) else v 
                for k, v in tensor_dict.items()}

# Modified train function for server (PC1)
def train_server():
    game_region = get_game_region()
    if not game_region:
        return

    net_manager = NetworkManager(role='server')
    net_manager.start_server()

    ai = DistributedGeneticAI(
        network_manager=net_manager,
        population_size=20
    )

    if os.path.exists("ai_checkpoint.pkl"):
        ai.load_checkpoint()

    frame_processor = FrameProcessor()

    try:
        while True:
            print(f"\n--- Generation {ai.current_generation} ---")
            
            # Split population between local and remote
            split = len(ai.population) // 2
            local_pop = ai.population[:split]
            remote_pop = [ai.tensor_to_cpu(p) for p in ai.population[split:]]
            
            # Send remote population to client
            net_manager.send_data({
                'population': remote_pop,
                'generation': ai.current_generation
            })
            
            # Run local matches
            local_fitness = run_local_matches(local_pop, game_region, frame_processor)
            
            # Receive remote results
            remote_data = net_manager.receive_data()
            remote_fitness = torch.tensor(remote_data['fitness'], device=DEVICE)
            remote_pop = [ai.tensor_to_device(p) for p in remote_data['population']]
            
            # Combine results and evolve
            ai.fitness_scores = torch.cat([local_fitness, remote_fitness])
            ai.population = local_pop + remote_pop
            ai.evolve_population_distributed(remote_fitness, remote_pop)
            
            ai.save_checkpoint()
            
    except KeyboardInterrupt:
        print("\nTraining stopped. Final generation saved.")
        ai.save_checkpoint()

def run_local_matches(population, game_region, frame_processor):
    fitness = torch.zeros(len(population), device=DEVICE)
    
    for idx, individual in enumerate(population):
        controller_p1 = InputController(1)
        controller_p2 = InputController(2)
        
        health_history = []
        action_history_p1 = []
        action_history_p2 = []
        
        reset_game(controller_p1, controller_p2)
        
        start_time = time.time()
        while time.time() - start_time < 10:
            raw_frame = capture_screen(game_region)
            if raw_frame is None:
                continue
                
            processed = frame_processor.preprocess(raw_frame)
            frame_processor.update_frame_buffer(processed)
            raw_health = frame_processor.detect_health(raw_frame)
            
            if raw_health['player'] <= 0.01 or raw_health['opponent'] <= 0.01:
                break
                
            # Process inputs and get actions
            processed_input = frame_processor.get_processed_input()
            actions_p1 = ai.predict(individual, processed_input, 
                                  {'player': raw_health['player'], 
                                   'opponent': raw_health['opponent']}, True)
            controller_p1.send_inputs(actions_p1)
            
            health_history.append(raw_health)
            action_history_p1.append(actions_p1)
            
        fitness[idx] = calculate_fitness(health_history, action_history_p1, True)
        controller_p1.release_all()
        controller_p2.release_all()
    
    return fitness

# Client code (PC2)
def train_client():
    game_region = get_game_region()
    if not game_region:
        return

    net_manager = NetworkManager(role='client', host='PC1_IP_ADDRESS')
    net_manager.connect_client()

    ai = DistributedGeneticAI(network_manager=net_manager, population_size=20)
    frame_processor = FrameProcessor()

    while True:
        # Receive population from server
        data = net_manager.receive_data()
        population = [ai.tensor_to_device(p) for p in data['population']]
        ai.current_generation = data['generation']
        
        # Run matches
        fitness = run_local_matches(population, game_region, frame_processor)
        
        # Send results back
        net_manager.send_data({
            'fitness': fitness.cpu().tolist(),
            'population': [ai.tensor_to_cpu(p) for p in population]
        })

if __name__ == "__main__":
    role = sys.argv[1] if len(sys.argv) > 1 else 'server'
    
    if role == 'server':
        pydirectinput.PAUSE = 0.0
        train_server()
    elif role == 'client':
        pydirectinput.PAUSE = 0.0
        train_client()
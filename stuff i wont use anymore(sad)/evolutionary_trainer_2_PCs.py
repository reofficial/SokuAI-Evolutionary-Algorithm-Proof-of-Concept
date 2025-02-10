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
import hashlib

class NetworkManager:
    def __init__(self, role='server', host='0.0.0.0', port=65432):
        self.role = role
        self.host = host
        self.port = port
        self.sock = None
        self.lock = threading.Lock()
        self.timeout = 10  # Add timeout
        self.connected = False
        
    def start_server(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.settimeout(self.timeout)
            self.sock.bind((self.host, self.port))
            self.sock.listen()
            print(f"Server listening on {self.host}:{self.port}")
            while True:
                try:
                    conn, addr = self.sock.accept()
                    conn.settimeout(self.timeout)
                    self.conn = conn
                    self.connected = True
                    print(f"Connected to {addr}")
                    break
                except socket.timeout:
                    print("Waiting for client connection...")
        except Exception as e:
            print(f"Server setup error: {str(e)}")
            raise

    def connect_client(self):
        while not self.connected:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(self.timeout)
                self.sock.connect((self.host, self.port))
                self.connected = True
                print(f"Connected to server at {self.host}:{self.port}")
            except (ConnectionRefusedError, socket.timeout) as e:
                print(f"Connection failed: {str(e)}, retrying in 3s...")
                time.sleep(3)
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                raise

    def send_data(self, data):
        if not self.connected:
            raise ConnectionError("Not connected")
        try:
            # Add debug logging
            print(f"[NET] Preparing to send data type: {type(data)}")
            serializable_data = self._convert_tensors(data)
            json_data = json.dumps(serializable_data)
            print(f"[NET] Sending {len(json_data)} bytes")
            encoded = json_data.encode('utf-8')
            # Add checksum
            checksum = hashlib.md5(encoded).hexdigest()
            header = struct.pack('>I16s', len(encoded), checksum.encode())
            self._get_connection().sendall(header + encoded)
        except Exception as e:
            print(f"Send error: {str(e)}")
            self.connected = False
            raise

    def receive_data(self):
        try:
            conn = self._get_connection()
            # Receive header with checksum
            header = self.recvall(conn, 4 + 16)  # 4 bytes length + 16 bytes MD5
            if not header:
                print("Connection closed by peer")
                self.connected = False
                return None

            msg_len, checksum = struct.unpack('>I16s', header)
            msg_len = int(msg_len)
            checksum = checksum.decode()
            
            print(f"[NET] Expecting {msg_len} bytes")
            encoded = self.recvall(conn, msg_len)
            if not encoded:
                print("Incomplete data received")
                self.connected = False
                return None

            # Verify checksum
            calc_checksum = hashlib.md5(encoded).hexdigest()
            if calc_checksum != checksum:
                print(f"Checksum mismatch: {calc_checksum} vs {checksum}")
                self.connected = False
                return None

            data = json.loads(encoded.decode('utf-8'))
            print(f"[NET] Received data type: {type(data)}")
            return self._restore_tensors(data)
        except socket.timeout:
            print("Receive timeout")
            self.connected = False
            return None
        except Exception as e:
            print(f"Receive error: {str(e)}")
            self.connected = False
            raise

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

    def _get_connection(self):
        if self.role == 'server':
            if not self.conn:
                raise ConnectionError("No client connection")
            return self.conn
        else:
            if not self.sock:
                raise ConnectionError("No server connection")
            return self.sock

# Modified GeneticAI class
class DistributedGeneticAI(GeneticAI):
    def __init__(self, network_manager=None, **kwargs):
        super().__init__(**kwargs)
        self.network = network_manager
        
    def evolve_population_distributed(self, remote_fitness, remote_population):
        print(f"[EVO] Starting evolution with {len(remote_population)} remote agents")
        # Add validation
        if len(self.population) + len(remote_population) != self.population_size * 1.5:
            raise ValueError("Population size mismatch")
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
            print(f"Initial population size: {len(ai.population)}")
            print(f"First agent structure: {ai.population[0].keys()}")
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
    net_manager.connect_client()  # This now has retry logic

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
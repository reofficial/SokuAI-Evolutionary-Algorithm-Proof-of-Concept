"""
FrameProcessor

This module captures, processes, and extracts features from game frames for AI decision-making.
It prepares a sequence of the last 10 frames, processes them using YUV conversion and Sobel edge detection,
and feeds them into a Convolutional Neural Network (CNN) to generate a 256-dimensional feature vector.

Key Features:
- Captures game frames and preprocesses them (YUV conversion, contrast enhancement, edge detection).
- Maintains a rolling buffer of the last 10 frames for temporal awareness.
- Resizes frames to 128×128 for efficient CNN processing.
- Uses a CNN to extract meaningful game-state features.
- Provides health bar detection to track player and opponent HP.

Main Methods:
- preprocess(frame): Converts a raw frame into a processed (128,128,4) format with Y, U, V, and edges.
- update_frame_buffer(frame): Stores processed frames in a rolling buffer.
- get_processed_input(): Returns the last 10 frames as a (128,128,40) NumPy array.
- extract_features(): Passes stacked frames through the CNN and returns a 256-dimensional feature vector.
- detect_health(frame): Extracts player and opponent HP from the frame.

Designed for real-time performance in fighting game AI.
"""

import cv2
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F

from screen_capture import get_game_region, capture_screen

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels=40, output_size=256, target_size=(128, 128)):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=8, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Compute the flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, target_size[0], target_size[1])
            out = self.conv1(dummy)
            out = self.conv2(out)
            out = self.conv3(out)
            flattened_size = out.view(1, -1).shape[1]

        self.fc = nn.Linear(flattened_size, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc(x)))
        return x
    
class FrameProcessor:
    def __init__(self, target_size=(128, 128), frame_stack=10):
        self.target_size = target_size
        self.frame_stack = frame_stack
        self.frame_buffer = deque(maxlen=frame_stack)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Health detection parameters
        self.health_y_range = (0.071, 0.10)
        self.player_x_range = (0.015, 0.41)
        self.opponent_x_range = (0.5925, 0.985)
        self.health_color = {
            'lower': [10, 61, 140],
            'upper': [50, 255, 255]
        }
        self.gray_color = {
            'lower': [0, 0, 0],
            'upper': [179, 20, 80]
        }
        
        # Instantiate the CNN for feature extraction.
        self.cnn = CNNFeatureExtractor(input_channels=self.frame_stack * 4, output_size=256, target_size=self.target_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn.to(self.device)
        self.cnn.eval()
    
    def preprocess(self, frame, is_player1=True):
        if not is_player1:
            frame = cv2.flip(frame, 1)
            
        # Convert frame to YUV and apply CLAHE on the Y channel.
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = self.clahe.apply(yuv[:, :, 0])
        
        # Compute Sobel edges
        sobelx = cv2.Sobel(yuv[:, :, 0], cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(yuv[:, :, 0], cv2.CV_64F, 0, 1, ksize=5)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = edges / (np.max(edges) + 1e-6)  # Avoid division by zero
        edges = np.clip(edges * 255, 0, 255).astype(np.uint8)
        
        # Resize all channels to the target size.
        def fast_resize(channel):
            return cv2.resize(channel, self.target_size, interpolation=cv2.INTER_AREA)

        resized_y, resized_u, resized_v = map(fast_resize, (yuv[:, :, 0], yuv[:, :, 1], yuv[:, :, 2]))
        resized_edges = fast_resize(edges)
        
        # Stack channels: Y, U, V, and edges.
        processed = np.dstack((resized_y, resized_u, resized_v, resized_edges))
        processed = (processed / 255.0).astype(np.float32)  # Normalize to [0, 1].
        return processed

    def update_frame_buffer(self, processed_frame):
        self.frame_buffer.append(processed_frame)

    def get_processed_input(self):
        
        while len(self.frame_buffer) < self.frame_stack:
            self.frame_buffer.appendleft(np.zeros((*self.target_size, 4), dtype=np.float32))
        return np.concatenate(self.frame_buffer, axis=-1)

    def get_processed_input_tensor(self):
        processed = self.get_processed_input()  # Shape: (H, W, channels)
        tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0)  # to (1, channels, H, W)
        return tensor.to(self.device)

    def extract_features(self):
        input_tensor = self.get_processed_input_tensor()
        with torch.no_grad():
            features = self.cnn(input_tensor)
        return features.cpu().numpy().squeeze()

    def get_health_regions(self, frame):
        h, w = frame.shape[:2]
        return {
            'player': (
                int(w * self.player_x_range[0]),
                int(h * self.health_y_range[0]),
                int(w * self.player_x_range[1]),
                int(h * self.health_y_range[1])
            ),
            'opponent': (
                int(w * self.opponent_x_range[0]),
                int(h * self.health_y_range[0]),
                int(w * self.opponent_x_range[1]),
                int(h * self.health_y_range[1])
            )
        }

    def detect_health(self, frame):
        regions = self.get_health_regions(frame)
        health_data = {'player': 0.0, 'opponent': 0.0}
        
        for role in ['player', 'opponent']:
            x1, y1, x2, y2 = regions[role]
            if x2 <= x1 or y2 <= y1:
                continue
                
            roi = frame[y1:y2, x1:x2]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            health_mask = cv2.inRange(hsv, 
                                    np.array(self.health_color['lower']),
                                    np.array(self.health_color['upper']))
            gray_mask = cv2.inRange(hsv, 
                                np.array(self.gray_color['lower']),
                                np.array(self.gray_color['upper']))
            
            combined_mask = cv2.bitwise_and(health_mask, cv2.bitwise_not(gray_mask))
            
            if role == 'player':
                filled_columns = []
                for row in combined_mask:
                    if np.any(row == 255):
                        first_health = np.argmax(row == 255)
                        filled_columns.append(combined_mask.shape[1] - first_health)
                    else:
                        filled_columns.append(0)
                
                health = np.mean(filled_columns) / combined_mask.shape[1]
            else:
                filled_columns = []
                for row in combined_mask:
                    if np.any(row == 255):
                        reversed_row = row[::-1]
                        first_health = np.argmax(reversed_row == 255)
                        filled_columns.append(combined_mask.shape[1] - first_health)
                    else:
                        filled_columns.append(0)
                
                health = np.mean(filled_columns) / combined_mask.shape[1]
            
            health_data[role] = np.clip(health, 0.0, 1.0)
            
        return health_data

    def create_channel_view(self, processed):
        """
        Creates a visualization of the individual channels (Y, U, V, edges)
        from the processed image.
        """
        y_channel = (processed[:, :, 0] * 255).astype(np.uint8)
        u_channel = (processed[:, :, 1] * 255).astype(np.uint8)
        v_channel = (processed[:, :, 2] * 255).astype(np.uint8)
        edges = (processed[:, :, 3] * 255).astype(np.uint8)
        combined = np.hstack([y_channel, u_channel, v_channel, edges])
        return cv2.resize(combined, (self.target_size[0]*4, self.target_size[1]))

###############################################################################
# Main Testing Block
###############################################################################
if __name__ == "__main__":
    game_region = get_game_region()
    if not game_region:
        print("Game region not found.")
        exit(1)
    
    processor = FrameProcessor(target_size=(128, 128), frame_stack=4)
    
    while True:
        raw_frame = capture_screen(game_region)
        if raw_frame is None:
            continue
        
        # Preprocess the captured frame.
        processed = processor.preprocess(raw_frame)
        processor.update_frame_buffer(processed)
        
        # Detect health from the raw frame.
        health_data = processor.detect_health(raw_frame)
        
        # Extract CNN features from the stacked frames.
        features = processor.extract_features()
        
        # Create a visualization of the processed channels.
        channels_view = processor.create_channel_view(processed)
        
        # Build a simple health display.
        health_display = np.zeros((100, 400, 3), dtype=np.uint8)
        cv2.putText(health_display, f"Player Health: {health_data['player']:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(health_display, f"Opponent Health: {health_data['opponent']:.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Display the raw frame, processed channels, and health info.
        cv2.imshow("Raw Frame", raw_frame)
        cv2.imshow("Processed Channels", channels_view)
        cv2.imshow("Health Monitoring", health_display)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

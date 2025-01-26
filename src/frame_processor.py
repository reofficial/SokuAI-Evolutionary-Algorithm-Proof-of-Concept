import cv2
import numpy as np
from collections import deque
from screen_capture import get_game_region, capture_screen

class FrameProcessor:
    def __init__(self, target_size=(128, 128), frame_stack=4):
        self.target_size = target_size
        self.frame_stack = frame_stack
        self.frame_buffer = deque(maxlen=frame_stack)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        self.health_y_range = (0.071, 0.109)
        self.player_x_range = (0.015, 0.41)
        self.opponent_x_range = (0.59, 0.985)

        self.health_color = {
            'lower': [10, 100, 150],
            'upper': [30, 255, 255]
        }
        
        self.gray_color = {
            'lower': [0, 0, 0],
            'upper': [179, 50, 100]
        }

    def preprocess(self, frame, is_player1=True):
        """Process frame with perspective flipping"""
        if not is_player1:
            frame = cv2.flip(frame, 1)
            
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = self.clahe.apply(yuv[:,:,0])
        
        median = np.median(yuv[:,:,0])
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        edges = cv2.Canny(yuv[:,:,0], lower, upper)
        
        resized_yuv = cv2.resize(yuv, self.target_size, interpolation=cv2.INTER_AREA)
        resized_edges = cv2.resize(edges, self.target_size, interpolation=cv2.INTER_AREA)
        
        processed = np.dstack((
            resized_yuv[:,:,0],
            resized_yuv[:,:,1],
            resized_yuv[:,:,2],
            resized_edges
        ))
        
        return (processed / 255.0).astype(np.float32)

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

    def update_frame_buffer(self, processed_frame):
        self.frame_buffer.append(processed_frame)

    def get_processed_input(self):
        while len(self.frame_buffer) < self.frame_stack:
            self.frame_buffer.appendleft(np.zeros((*self.target_size, 4), dtype=np.float32))
        return np.concatenate(self.frame_buffer, axis=-1)

    def create_channel_view(self, processed):
        """Create processed channels visualization"""
        # Scale channels for visibility
        y_channel = (processed[:,:,0] * 255).astype(np.uint8)
        u_channel = (processed[:,:,1] * 127 + 128).astype(np.uint8)  # U scaled
        v_channel = (processed[:,:,2] * 127 + 128).astype(np.uint8)  # V scaled
        edges = (processed[:,:,3] * 255).astype(np.uint8)
        
        # Combine and resize
        combined = np.hstack([y_channel, u_channel, v_channel, edges])
        return cv2.resize(combined, (512, 128))

def main():
    game_region = get_game_region()
    if not game_region:
        print("Failed to find game window")
        return
    
    processor = FrameProcessor(target_size=(128, 128), frame_stack=4)
    
    # Window setup
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Processed Channels", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Health Monitoring", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Diagnostics", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            raw_frame = capture_screen(game_region)
            if raw_frame is None:
                continue
                
            # Process frame
            processed = processor.preprocess(raw_frame)
            processor.update_frame_buffer(processed)
            health_data = processor.detect_health(raw_frame)
            
            # Create visualizations
            channels_view = processor.create_channel_view(processed)
            
            # Health display
            health_display = np.zeros((100, 400, 3), dtype=np.uint8)
            cv2.putText(health_display, f"Player: {health_data['player']:.2f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(health_display, f"Opponent: {health_data['opponent']:.2f}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            
            # Diagnostics overlay
            debug_frame = raw_frame.copy()
            regions = processor.get_health_regions(raw_frame)
            cv2.rectangle(debug_frame, 
                         (regions['player'][0], regions['player'][1]),
                         (regions['player'][2], regions['player'][3]),
                         (0,0,255), 2)
            cv2.rectangle(debug_frame, 
                         (regions['opponent'][0], regions['opponent'][1]),
                         (regions['opponent'][2], regions['opponent'][3]),
                         (255,0,0), 2)
            
            # Show all windows
            cv2.imshow("Original", raw_frame)
            cv2.imshow("Processed Channels", channels_view)
            cv2.imshow("Health Monitoring", health_display)
            cv2.imshow("Diagnostics", debug_frame)
            
            # Controls
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('c'):
                print("\n=== Calibration Values ===")
                print(f"Vertical Range: {processor.health_y_range}")
                print(f"Player X: {processor.player_x_range}")
                print(f"Opponent X: {processor.opponent_x_range}")
                print(f"Player Color: {processor.health_colors['player']}")
                print(f"Opponent Color: {processor.health_colors['opponent']}")
                
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
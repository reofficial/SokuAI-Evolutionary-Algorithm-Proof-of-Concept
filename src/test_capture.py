# test_capture_phase1.py
import cv2
import numpy as np
from mss import mss
from screen_capture import get_all_game_regions
from frame_processor import FrameProcessor

def main():
    # Initialize
    regions = get_all_game_regions()
    processor = FrameProcessor()
    sct = mss()
    
    print(f"Found {len(regions)} instances")
    
    # Create windows in a grid layout
    for i in range(len(regions)):
        cv2.namedWindow(f"Instance {i+1} - Original", cv2.WINDOW_NORMAL)
        cv2.namedWindow(f"Instance {i+1} - Processed", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"Instance {i+1} - Original", 400, 300)
        cv2.resizeWindow(f"Instance {i+1} - Processed", 400, 300)
    
    try:
        while True:
            frames = []
            processed_frames = []
            
            # Capture and process all instances
            for i, region in enumerate(regions):
                # Capture original frame
                raw_frame = np.array(sct.grab(region))
                frames.append(raw_frame)
                
                # Process frame
                processed = processor.preprocess(raw_frame)
                
                # Convert processed frame to display format
                processed_display = (processed * 255).astype(np.uint8)
                if processed_display.shape[-1] == 4:  # If alpha channel
                    processed_display = cv2.cvtColor(processed_display, cv2.COLOR_BGRA2BGR)
                processed_frames.append(processed_display)
            
            # Display results
            for i in range(len(regions)):
                # Show original frame
                cv2.imshow(
                    f"Instance {i+1} - Original", 
                    cv2.resize(frames[i], (400, 300))
                )
                
                # Show processed frame
                cv2.imshow(
                    f"Instance {i+1} - Processed", 
                    cv2.resize(processed_frames[i], (400, 300))
                )
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cv2.destroyAllWindows()
        sct.close()

if __name__ == "__main__":
    main()
import win32gui
import win32con
import win32process
import psutil
from mss import mss
import cv2
import numpy as np
import time

def get_all_game_regions(window_title_part="Touhou Hisoutensoku", process_name="th123.exe"):
    hwnds = []
    
    # Find by window title
    def callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd) and window_title_part in win32gui.GetWindowText(hwnd):
            hwnds.append(hwnd)
        return True
    win32gui.EnumWindows(callback, hwnds)
    
    # Sort by window position
    hwnds.sort(key=lambda h: (win32gui.GetWindowRect(h)[0], win32gui.GetWindowRect(h)[1]))
    
    regions = []
    for hwnd in hwnds[:4]:  # Max 4 instances
        try:
            rect = win32gui.GetWindowRect(hwnd)
            regions.append({
                'hwnd': hwnd,
                'left': rect[0],
                'top': rect[1],
                'width': rect[2]-rect[0],
                'height': rect[3]-rect[1]
            })
        except Exception as e:
            print(f"Error processing window {hwnd}: {str(e)}")
    
    return regions

def get_game_region(window_title_part="Touhou Hisoutensoku", process_name="th123.exe"):
    """Find window by partial title or process name"""
    try:
        def callback(hwnd, hwnds):
            if win32gui.IsWindowVisible(hwnd):
                text = win32gui.GetWindowText(hwnd)
                if window_title_part in text:
                    hwnds.append(hwnd)
            return True

        hwnds = []
        win32gui.EnumWindows(callback, hwnds)
        
        if not hwnds:
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'].lower() == process_name.lower():
                    hwnds = []
                    win32gui.EnumWindows(lambda h, p: p.append(h) if win32process.GetWindowThreadProcessId(h)[1] == proc.info['pid'] else True, hwnds)
                    if hwnds:
                        hwnd = hwnds[0]
                        break
            else:
                raise ValueError(f"No process found with name: {process_name}")
        else:
            hwnd = hwnds[0]

        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.5)

        client_rect = win32gui.GetClientRect(hwnd)
        client_left, client_top = win32gui.ClientToScreen(hwnd, (client_rect[0], client_rect[1]))
        client_right, client_bottom = win32gui.ClientToScreen(hwnd, (client_rect[2], client_rect[3]))
        
        return {
            'left': client_left,
            'top': client_top,
            'width': client_right - client_left,
            'height': client_bottom - client_top
        }
        
    except Exception as e:
        print(f"Window detection failed: {str(e)}")
        return None

def capture_screen(region):
    """Capture screen region using MSS"""
    with mss() as sct:
        img = np.array(sct.grab(region))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
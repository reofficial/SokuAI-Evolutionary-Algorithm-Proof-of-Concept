# input_controller.py
import time
import ctypes
import win32con
import win32gui
import pydirectinput

# Mapping from keys to virtual-key codes (for background input)
VK_CODE = {
    'w': 0x57,
    's': 0x53,
    'a': 0x41,
    'd': 0x44,
    'g': 0x47,
    'h': 0x48,
    'j': 0x4A,
    'y': 0x59,
    't': 0x54,
    'u': 0x55,
    # Add additional mappings as needed.
}

class InputController:
    KEY_MAPPINGS = {
        1: {  # Player 1 controls
            0: 'w',
            1: 's',
            2: 'a',
            3: 'd',
            4: 'g',
            5: 'h',
            6: 'j',
            7: 'y',
            8: 't',
            9: 'u'
        },
        2: {  # Player 2 controls (mirrored)
            0: 'z',
            1: 'x',
            2: 'c',
            3: 'v',
            4: 'b',
            5: 'n',
            6: 'm',
            7: ',',
            8: '.',
            9: '/'
        }
    }
    
    def __init__(self, player_number, hwnd=None):
        """
        If hwnd (window handle) is provided, inputs will be sent to that window using background methods.
        Otherwise, pydirectinput (which requires an active window) is used.
        """
        self.player_number = player_number
        self.key_map = self.KEY_MAPPINGS[player_number]
        self.current_keys = set()
        self.hwnd = hwnd  # Optional: target window handle for background input.

    def send_inputs(self, action_vector):
        new_keys = set()
        for i in range(10):
            if action_vector[i] > 0.5:  # Using threshold
                new_keys.add(self.key_map[i])
                
        # Release keys that are no longer pressed
        for key in self.current_keys - new_keys:
            self.release_key(key)
            
        # Press keys that are newly pressed
        for key in new_keys - self.current_keys:
            self.press_key(key)
            
        self.current_keys = new_keys

    def press_key(self, key):
        if self.hwnd:
            # Send a background key press to the specified window handle.
            vk = VK_CODE.get(key)
            if vk:
                # Post WM_KEYDOWN message
                win32gui.PostMessage(self.hwnd, win32con.WM_KEYDOWN, vk, 0)
        else:
            pydirectinput.keyDown(key)
            
    def release_key(self, key):
        if self.hwnd:
            vk = VK_CODE.get(key)
            if vk:
                # Post WM_KEYUP message
                win32gui.PostMessage(self.hwnd, win32con.WM_KEYUP, vk, 0)
        else:
            pydirectinput.keyUp(key)

    def release_all(self):
        for key in list(self.current_keys):
            self.release_key(key)
        self.current_keys = set()

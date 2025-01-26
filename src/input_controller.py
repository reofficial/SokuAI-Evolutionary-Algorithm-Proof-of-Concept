import pydirectinput
import time

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
    
    def __init__(self, player_number):
        self.player_number = player_number
        self.key_map = self.KEY_MAPPINGS[player_number]
        self.current_keys = set()

    def send_inputs(self, action_vector):
        new_keys = set()
        for i in range(10):
            if action_vector[i] > 0.5:  # Using threshold
                new_keys.add(self.key_map[i])
                
        # Release unused keys
        for key in self.current_keys - new_keys:
            pydirectinput.keyUp(key)
            
        # Press new keys
        for key in new_keys - self.current_keys:
            pydirectinput.keyDown(key)
            
        self.current_keys = new_keys

    def release_all(self):
        for key in self.current_keys:
            pydirectinput.keyUp(key)
        self.current_keys = set()
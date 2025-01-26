import keyboard
import time
import ctypes

SendInput = ctypes.windll.user32.SendInput

# Define Windows API structures
class KeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
    ]

class HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort)
    ]

class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
    ]

class Input_I(ctypes.Union):
    _fields_ = [
        ("ki", KeyBdInput),
        ("mi", MouseInput),
        ("hi", HardwareInput)
    ]

class Input(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("ii", Input_I)
    ]

class GameMacro:
    def __init__(self):
        self._key_map = {
            'esc': 0x01,
            'up': 0xC8,
            'z': 0x2C,
            'a': 0x1E
        }

    def _press_key(self, hex_key_code):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, hex_key_code, 0x0008, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def _release_key(self, hex_key_code):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, hex_key_code, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def reset_match(self):
        """Direct Windows API input for macros"""
        try:
            keys = [
                ('esc', 0.2),
                ('up', 0.2),
                ('up', 0.2),
                ('z', 1.0),
                ('a', 0.1),
                ('a', 0.1),
                ('a', 0.1),
                ('a', 0.1),
                ('a', 0.1),
                ('a', 1.0),
                ('z', 4)
            ]
            
            for key, delay in keys:
                hex_code = self._key_map[key]
                self._press_key(hex_code)
                time.sleep(0.05)  # Minimum press duration
                self._release_key(hex_code)
                time.sleep(delay)
                
        except Exception as e:
            print(f"Macro error: {str(e)}")

if __name__ == "__main__":
    mc = GameMacro()
    print("Testing reset macro...")
    mc.reset_match()
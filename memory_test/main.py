import ctypes
import ctypes.wintypes as wintypes
import win32gui
import win32process
import psutil

# Constants for process access
PROCESS_ALL_ACCESS = 0x1F0FFF
kernel32 = ctypes.windll.kernel32

def get_process_id_by_window(window_title_part):
    """Finds a window with the given title part and returns its process id."""
    hwnd = win32gui.FindWindow(None, window_title_part)
    if hwnd == 0:
        print("Window not found!")
        return None
    _, pid = win32process.GetWindowThreadProcessId(hwnd)
    return pid

def read_memory(process_handle, address, size):
    """Reads memory from the given process handle at the specified address."""
    buffer = ctypes.create_string_buffer(size)
    bytesRead = ctypes.c_size_t(0)
    if not kernel32.ReadProcessMemory(process_handle, ctypes.c_void_p(address),
                                      buffer, size, ctypes.byref(bytesRead)):
        raise ctypes.WinError()
    return buffer.raw[:bytesRead.value]

def main():
    # Adjust window title if necessary
    window_title = "Touhou Hisoutensoku + Giuroll 0.6.11"
    pid = get_process_id_by_window(window_title)
    if pid is None:
        return

    print("Process ID:", pid)

    # Open the process (you may need to run as Administrator)
    process_handle = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
    if not process_handle:
        print("Could not open process. Try running as Administrator.")
        return

    # Define the addresses to test
    addresses = {
        "WEATHER": 0x008971C0,
        "DISPLAY_WEATHER": 0x008971C4,
        "WEATHER_COUNTER": 0x008971CC,
        # You can add other addresses here as needed.
    }

    for name, addr in addresses.items():
        try:
            # Read 4 bytes (adjust size if you expect a different data type)
            data = read_memory(process_handle, addr, 4)
            # Interpret the bytes as a little-endian integer
            value = int.from_bytes(data, "little")
            print(f"{name} (0x{addr:08X}): {value}")
        except Exception as e:
            print(f"Error reading {name} at address 0x{addr:08X}: {e}")

    kernel32.CloseHandle(process_handle)

if __name__ == "__main__":
    main()

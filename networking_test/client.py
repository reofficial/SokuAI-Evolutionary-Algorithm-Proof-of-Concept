import socket

def client():
    host = 'PC1_IP_ADDRESS'  # Replace with PC1's actual IP
    port = 65432
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.send(b"Ready for work")
        
        # Get work range from server
        data = s.recv(1024).decode()
        start, end = map(int, data.split('-'))
        print(f"Received work: {start} to {end}")
        
        # Calculate sum
        result = sum(range(start, end + 1))
        print(f"Client calculated: {result}")
        
        # Send result back
        s.send(str(result).encode())

if __name__ == "__main__":
    client()
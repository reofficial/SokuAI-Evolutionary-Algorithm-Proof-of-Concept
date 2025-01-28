import socket
import threading
import time

def handle_client(conn, addr):
    print(f"Connected to {addr}")
    data = conn.recv(1024).decode()
    print(f"Received from client: {data}")
    
    # Split work (simple number range)
    total_work = 1000
    split_point = total_work // 2
    client_work = f"{split_point + 1}-{total_work}"
    server_work = (1, split_point)
    
    # Send work to client
    conn.send(client_work.encode())
    
    # Do server work
    server_result = sum(range(server_work[0], server_work[1] + 1))
    print(f"Server calculated: {server_result}")
    
    # Get client result
    client_result = int(conn.recv(1024).decode())
    total = server_result + client_result
    print(f"Final total: {total} (Server: {server_result} + Client: {client_result})")
    conn.close()

def server():
    host = '0.0.0.0'  # Listen on all interfaces
    port = 65432
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print("Server listening...")
        while True:
            conn, addr = s.accept()
            thread = threading.Thread(target=handle_client, args=(conn, addr))
            thread.start()

if __name__ == "__main__":
    server()
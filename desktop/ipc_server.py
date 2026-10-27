import socket
import json
import threading
import logging
from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)

class IPCServer(QObject):
    """
    Local socket server to receive commands from the CLI tool.
    Emits a Qt signal when a command is received so the GUI thread can handle it.
    """
    command_received = Signal(dict)

    def __init__(self, host='127.0.0.1', port=5555):
        super().__init__()
        self.host = host
        self.port = port
        self.running = False
        self.server_socket = None
        self.thread = None

    def start(self):
        """Start the socket server in a background thread."""
        if self.running:
            return

        self.running = True
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Allow port reuse to avoid "Address already in use" errors on restart
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            logger.info(f"IPC Server listening on {self.host}:{self.port}")
            
            self.thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.thread.start()
        except Exception as e:
            logger.error(f"Failed to start IPC Server: {e}")
            self.running = False

    def stop(self):
        """Stop the server."""
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
            self.server_socket = None

    def _listen_loop(self):
        """Background loop to accept connections."""
        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                # Handle client in a separate thread or just process quickly here
                # Since commands are small, processing here is fine for now
                self._handle_client(client_socket)
            except OSError:
                # Socket closed
                break
            except Exception as e:
                logger.error(f"Error accepting connection: {e}")

    def _handle_client(self, client_socket):
        """Read data from client and emit signal."""
        try:
            with client_socket:
                data = client_socket.recv(4096)
                if not data:
                    return
                
                try:
                    command = json.loads(data.decode('utf-8'))
                    logger.info(f"Received command: {command}")
                    # Emit signal to Main GUI Thread
                    self.command_received.emit(command)
                    
                    # Send success response
                    response = {"status": "ok", "received": command}
                    client_socket.sendall(json.dumps(response).encode('utf-8'))
                except json.JSONDecodeError:
                    err = {"status": "error", "message": "Invalid JSON"}
                    client_socket.sendall(json.dumps(err).encode('utf-8'))
                    
        except Exception as e:
            logger.error(f"Error handling client: {e}")

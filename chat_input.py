# Save this as chat_input.py in the same directory
import os
import tempfile
import time
import sys

def send_message(message):
    """Send a message to the running model"""
    temp_dir = tempfile.gettempdir()
    input_file = os.path.join(temp_dir, "gemma_memory_input.txt")
    input_ready_file = os.path.join(temp_dir, "gemma_memory_ready.txt")
    
    # Write message to input file
    with open(input_file, 'w') as f:
        f.write(message)
    
    # Create ready file to signal message is ready
    with open(input_ready_file, 'w') as f:
        pass
    
    print(f"Message sent: {message}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Get message from command line arguments
        message = " ".join(sys.argv[1:])
        send_message(message)
    else:
        # Interactive mode
        print("Gemma-Memory Chat Input Tool")
        print("Type your messages below. Type 'exit' to quit.")
        
        while True:
            message = input("> ")
            if message.lower() == "exit":
                send_message("exit")
                break
            send_message(message)
            # Wait a bit to avoid sending messages too quickly
            time.sleep(0.5)
import subprocess
import os
import sys
import time

def start_vllm_server():
    # Get the number of GPUs
    import torch
    gpu_num = torch.cuda.device_count()
    
    # Model path
    model_name = "pretrained_models/Qwen3-8B"
    
    # Construct the command
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--tensor-parallel-size", str(gpu_num),
        "--port", "8088",
        "--host", "0.0.0.0"
    ]
    
    # Start the server in a subprocess
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Wait a bit for the server to start
    time.sleep(5)
    
    # Check if the process is still running
    if process.poll() is None:
        print("vLLM server started successfully on port 8088")
        return process
    else:
        stdout, stderr = process.communicate()
        print("Failed to start vLLM server:")
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        return None

if __name__ == "__main__":
    server_process = start_vllm_server()
    if server_process:
        try:
            # Keep the script running
            server_process.wait()
        except KeyboardInterrupt:
            print("\nShutting down vLLM server...")
            server_process.terminate()
            server_process.wait() 
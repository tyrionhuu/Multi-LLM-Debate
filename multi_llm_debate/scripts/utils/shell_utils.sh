#!/bin/bash

# This file contains reusable shell functions for scripts

# Cleanup function for VLLM server processes
cleanup_vllm_server() {
    local server_pid=$1
    echo "Cleaning up..."
    if [[ -n "$server_pid" ]]; then
        echo "Terminating VLLM server (PID: $server_pid)..."
        kill $server_pid 2>/dev/null || true
        # Wait a moment and force kill if still running
        sleep 2
        if kill -0 $server_pid 2>/dev/null; then
            echo "Server still running, force killing..."
            kill -9 $server_pid 2>/dev/null || true
        fi
    fi
    echo "Cleanup complete."
}

# Function to wait for server to be ready
wait_for_server() {
    local port=$1
    local max_attempts=${2:-30}
    local sleep_time=${3:-6}
    
    echo "Waiting for server to start..."
    sleep 30  # Initial wait
    
    local attempt=2
    while ! curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; do
        if [ $attempt -ge $max_attempts ]; then
            echo "Server did not start after $max_attempts attempts. Exiting."
            return 1
        fi
        echo "Attempt $attempt: Server not ready yet. Waiting..."
        sleep $sleep_time
        attempt=$((attempt+1))
    done
    
    echo "Server is ready!"
    return 0
}

# Activate conda environment if not already active
activate_conda_env() {
    local env_name=$1
    
    if [[ "$CONDA_DEFAULT_ENV" != "$env_name" ]]; then
        echo "Activating $env_name conda environment..."
        eval "$(conda shell.bash hook)"
        conda activate $env_name
        return $?
    else
        echo "$env_name conda environment is already activated."
        return 0
    fi
}

import argparse
import logging
import subprocess
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def serve_model_with_vllm(
    model_name_or_path: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    host: str = "0.0.0.0",
    port: int = 8000,
    max_model_len: int = 4096,
    dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    quantization: Optional[str] = None,
    additional_args: Optional[Dict[str, Any]] = None,
) -> subprocess.Popen:
    """Serves a model using VLLM server.
    
    Args:
        model_name_or_path: Path to the model or model name from HuggingFace.
        tensor_parallel_size: Number of GPUs to use for tensor parallelism.
        gpu_memory_utilization: Fraction of GPU memory to use.
        host: Host address to bind the server to.
        port: Port to bind the server to.
        max_model_len: Maximum sequence length for the model.
        dtype: Data type to use for the model (float16, bfloat16, float32).
        trust_remote_code: Whether to trust remote code in the model.
        quantization: Quantization method to use (awq, squeezellm, gptq).
        additional_args: Additional arguments to pass to the VLLM server.
        
    Returns:
        A subprocess.Popen object representing the running server.
    
    Raises:
        FileNotFoundError: If the VLLM package is not installed.
        RuntimeError: If the server fails to start.
    """
    try:
        cmd = [
            "python", "-m", "vllm.entrypoints.api_server",
            "--model", model_name_or_path,
            "--tensor-parallel-size", str(tensor_parallel_size),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--host", host,
            "--port", str(port),
            "--max-model-len", str(max_model_len),
            "--dtype", dtype,
        ]
        
        if trust_remote_code:
            cmd.append("--trust-remote-code")
            
        if quantization:
            cmd.extend(["--quantization", quantization])
            
        # Add any additional arguments
        if additional_args:
            for key, value in additional_args.items():
                if isinstance(value, bool) and value:
                    cmd.append(f"--{key}")
                else:
                    cmd.extend([f"--{key}", str(value)])
        
        logger.info(f"Starting VLLM server with command: {' '.join(cmd)}")
        
        # Start the server as a subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Log that the server is starting
        logger.info(f"VLLM server starting with PID {process.pid}")
        return process
        
    except FileNotFoundError:
        logger.error("Failed to start VLLM server. Is VLLM installed?")
        raise FileNotFoundError(
            "VLLM package not found. Install it with: pip install vllm"
        )

def main():
    """Command line interface for serving models with VLLM."""
    parser = argparse.ArgumentParser(description="Serve a model with VLLM")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Path to the model or model name from HuggingFace"
    )
    parser.add_argument(
        "--tensor-parallel-size", 
        type=int, 
        default=1,
        help="Number of GPUs to use for tensor parallelism"
    )
    parser.add_argument(
        "--gpu-memory-utilization", 
        type=float, 
        default=0.9,
        help="Fraction of GPU memory to use"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0",
        help="Host address to bind the server to"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port to bind the server to"
    )
    parser.add_argument(
        "--max-model-len", 
        type=int, 
        default=4096,
        help="Maximum sequence length for the model"
    )
    parser.add_argument(
        "--dtype", 
        type=str, 
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type to use for the model"
    )
    parser.add_argument(
        "--trust-remote-code", 
        action="store_true",
        help="Whether to trust remote code in the model"
    )
    parser.add_argument(
        "--quantization", 
        type=str, 
        choices=["awq", "squeezellm", "gptq"],
        help="Quantization method to use"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Start the server
    process = serve_model_with_vllm(
        model_name_or_path=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        host=args.host,
        port=args.port,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        quantization=args.quantization
    )
    
    try:
        # Wait for the process to complete (or be interrupted)
        process.wait()
    except KeyboardInterrupt:
        logger.info("Stopping VLLM server...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("VLLM server did not terminate gracefully, killing...")
            process.kill()

if __name__ == "__main__":
    main()

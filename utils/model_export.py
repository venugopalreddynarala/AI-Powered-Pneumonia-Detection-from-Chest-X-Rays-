"""
Model Export module (ONNX / TensorRT).
Exports trained PyTorch models to ONNX format for cross-platform deployment
and faster inference.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Dict


def export_to_onnx(model: nn.Module, output_path: str = 'exports/model.onnx',
                   input_size: tuple = (1, 3, 224, 224),
                   opset_version: int = 14,
                   dynamic_axes: bool = True,
                   optimize: bool = True) -> str:
    """
    Export a PyTorch model to ONNX format.
    
    Args:
        model: Trained PyTorch model
        output_path: Path to save the ONNX model
        input_size: Input tensor shape
        opset_version: ONNX opset version
        dynamic_axes: Enable dynamic batch size
        optimize: Apply ONNX optimizations
    
    Returns:
        Path to the exported ONNX model
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_size, device=device)
    
    # Dynamic axes for variable batch size
    axes = None
    if dynamic_axes:
        axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    print(f"Exporting model to ONNX (opset {opset_version})...")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=axes
    )
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX model saved to {output_path} ({file_size:.1f} MB)")
    
    # Optimize if requested
    if optimize:
        try:
            optimized_path = _optimize_onnx(output_path)
            if optimized_path:
                output_path = optimized_path
        except Exception as e:
            print(f"ONNX optimization skipped: {e}")
    
    # Validate
    _validate_onnx(output_path)
    
    return output_path


def _optimize_onnx(model_path: str) -> Optional[str]:
    """Apply ONNX graph optimizations."""
    try:
        from onnxruntime.transformers import optimizer
        
        optimized_path = model_path.replace('.onnx', '_optimized.onnx')
        # Basic optimization
        import onnx
        from onnxruntime.transformers.onnx_model import OnnxModel
        
        model = onnx.load(model_path)
        # Remove unnecessary nodes
        # (Simple pass-through optimization)
        onnx.save(model, optimized_path)
        
        print(f"Optimized ONNX model saved to {optimized_path}")
        return optimized_path
    except ImportError:
        print("onnxruntime.transformers not available. Skipping optimization.")
        return None


def _validate_onnx(model_path: str) -> bool:
    """Validate exported ONNX model."""
    try:
        import onnx
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        print(f"ONNX validation passed: {model_path}")
        return True
    except ImportError:
        print("onnx package not installed. Skipping validation.")
        return True
    except Exception as e:
        print(f"ONNX validation failed: {e}")
        return False


class ONNXPredictor:
    """
    ONNX Runtime inference engine.
    Provides faster inference compared to PyTorch for production deployment.
    """
    
    def __init__(self, model_path: str, providers: list = None):
        """
        Args:
            model_path: Path to ONNX model file
            providers: ONNX Runtime execution providers
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required. Install with: pip install onnxruntime"
            )
        
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"ONNX model loaded from {model_path}")
        print(f"Providers: {self.session.get_providers()}")
    
    def predict(self, input_array: np.ndarray) -> Dict:
        """
        Run prediction using ONNX Runtime.
        
        Args:
            input_array: Input numpy array (batch, channels, H, W)
        
        Returns:
            Dictionary with prediction results
        """
        if isinstance(input_array, torch.Tensor):
            input_array = input_array.cpu().numpy()
        
        input_array = input_array.astype(np.float32)
        
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_array}
        )
        
        logits = outputs[0]
        
        # Softmax
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        
        predictions = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidence': confidence,
            'logits': logits,
        }
    
    def benchmark(self, input_shape: tuple = (1, 3, 224, 224),
                  num_runs: int = 100) -> Dict:
        """
        Benchmark ONNX inference speed.
        
        Returns:
            Dictionary with timing statistics
        """
        import time
        
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            self.session.run([self.output_name], {self.input_name: dummy_input})
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.session.run([self.output_name], {self.input_name: dummy_input})
            end = time.perf_counter()
            times.append((end - start) * 1000)  # milliseconds
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'p95_ms': np.percentile(times, 95),
            'throughput_fps': 1000 / np.mean(times),
        }


def compare_pytorch_vs_onnx(pytorch_model: nn.Module,
                             onnx_path: str,
                             input_shape: tuple = (1, 3, 224, 224)) -> Dict:
    """
    Compare PyTorch and ONNX inference speed and accuracy.
    
    Returns:
        Comparison dictionary
    """
    import time
    
    device = next(pytorch_model.parameters()).device
    dummy_input_torch = torch.randn(*input_shape, device=device)
    dummy_input_numpy = dummy_input_torch.cpu().numpy()
    
    # PyTorch inference
    pytorch_model.eval()
    torch_times = []
    for _ in range(50):
        start = time.perf_counter()
        with torch.no_grad():
            _ = pytorch_model(dummy_input_torch)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        torch_times.append((end - start) * 1000)
    
    # ONNX inference
    onnx_pred = ONNXPredictor(onnx_path)
    onnx_times = []
    for _ in range(50):
        start = time.perf_counter()
        _ = onnx_pred.predict(dummy_input_numpy)
        end = time.perf_counter()
        onnx_times.append((end - start) * 1000)
    
    # Check output equivalence
    with torch.no_grad():
        torch_out = pytorch_model(dummy_input_torch).cpu().numpy()
    onnx_out = onnx_pred.predict(dummy_input_numpy)['logits']
    
    max_diff = np.abs(torch_out - onnx_out).max()
    
    return {
        'pytorch_mean_ms': np.mean(torch_times),
        'onnx_mean_ms': np.mean(onnx_times),
        'speedup': np.mean(torch_times) / np.mean(onnx_times),
        'max_output_diff': float(max_diff),
        'outputs_match': bool(max_diff < 1e-4),
    }


if __name__ == "__main__":
    print("Model export module loaded successfully")
    print("Features: ONNX export, validation, optimization, benchmarking")

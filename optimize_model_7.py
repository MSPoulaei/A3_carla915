import os
import time
import numpy as np
import onnxruntime as ort
from pathlib import Path
from ultralytics import YOLO

def benchmark_model(model_path: Path, img_path: Path, num_runs=100):
    """Benchmarks a given model and returns average inference time in ms."""
    if model_path.suffix == '.pt':
        # Benchmark PyTorch model
        model = YOLO(model_path)
        # Warm-up runs
        for _ in range(10):
            model(img_path, verbose=False)
        
        start_time = time.time()
        for _ in range(num_runs):
            model(img_path, verbose=False)
        end_time = time.time()
        
    elif model_path.suffix == '.onnx':
        # Benchmark ONNX model
        session = ort.InferenceSession(str(model_path), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        # Warm-up runs
        for _ in range(10):
            session.run(None, {input_name: dummy_input})
            
        start_time = time.time()
        for _ in range(num_runs):
            session.run(None, {input_name: dummy_input})
        end_time = time.time()
    
    else:
        raise ValueError("Unsupported model format for benchmarking")

    return (end_time - start_time) / num_runs * 1000

def quantize_and_compare(model_path: Path, sample_image_path: Path):
    """
    Quantizes a PyTorch model to INT8 ONNX and compares size and performance.
    """
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    print(f"--- Optimizing Model: {model_path.name} ---")
    model = YOLO(model_path)
    
    # 1. Export to FP32 ONNX (standard)
    print("Exporting to FP32 ONNX...")
    fp32_model_path = model.export(format="onnx", dynamic=True, half=False)
    fp32_model_path = Path(fp32_model_path)
    
    # 2. Export to INT8 ONNX (quantized)
    # The `int8=True` flag enables post-training static quantization
    print("Exporting to INT8 ONNX...")
    int8_model_path = model.export(format="onnx", dynamic=True, int8=True)
    int8_model_path = Path(int8_model_path)
    
    # 3. Compare file sizes
    fp32_size = os.path.getsize(fp32_model_path) / (1024 * 1024)
    int8_size = os.path.getsize(int8_model_path) / (1024 * 1024)
    
    # 4. Compare inference speeds
    print("Benchmarking... (this may take a minute)")
    pytorch_speed = benchmark_model(model_path, sample_image_path)
    fp32_onnx_speed = benchmark_model(fp32_model_path, sample_image_path)
    int8_onnx_speed = benchmark_model(int8_model_path, sample_image_path)
    
    # 5. Report results
    print("\n--- Optimization Summary ---")
    print(f"{'Model':<20} | {'Size (MB)':<12} | {'Inference (ms)':<15}")
    print("-" * 55)
    print(f"{'PyTorch (FP32)':<20} | {'N/A':<12} | {pytorch_speed:<15.2f}")
    print(f"{'ONNX (FP32)':<20} | {fp32_size:<12.2f} | {fp32_onnx_speed:<15.2f}")
    print(f"{'ONNX (INT8)':<20} | {int8_size:<12.2f} | {int8_onnx_speed:<15.2f}")
    
    size_reduction = (1 - int8_size / fp32_size) * 100
    speed_improvement = (1 - int8_onnx_speed / fp32_onnx_speed) * 100
    print("\n--- Conclusion ---")
    print(f"INT8 quantization reduced model size by {size_reduction:.1f}%.")
    print(f"INT8 quantization improved ONNX inference speed by {speed_improvement:.1f}%.")

if __name__ == '__main__':
    # Path to your best trained PyTorch model
    trained_model_path = Path("runs/train/carla_segmentation_nano/weights/best.pt")
    
    # A sample image for benchmarking
    sample_image = Path("yolo_dataset/test/images/day_00001.png")
    
    if not sample_image.exists():
        print(f"Error: Sample image not found at {sample_image}. Please provide a valid path.")
    else:
        quantize_and_compare(trained_model_path, sample_image)
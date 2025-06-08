import cv2
import time
import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression, scale_boxes

# --- TensorRT Helper Class ---
class TRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = [], [], [], cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def infer(self, image):
        np.copyto(self.inputs[0]['host'], image.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()
        return [out['host'] for out in self.outputs]

# --- Pre and Post-processing Functions ---
def preprocess(img, target_shape=(512, 512)):
    h, w, _ = img.shape
    r = min(target_shape[0] / h, target_shape[1] / w)
    unpad_w, unpad_h = int(round(w * r)), int(round(h * r))
    resized_img = cv2.resize(img, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to target shape
    dw, dh = (target_shape[1] - unpad_w) / 2, (target_shape[0] - unpad_h) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # To float32 and CHW format
    blob = cv2.dnn.blobFromImage(padded_img, 1.0/255.0, target_shape, swapRB=True, crop=False)
    return blob, (h, w), (unpad_h, unpad_w), (top, left)

def postprocess(preds, original_shape, resized_shape, pad_info):
    # For YOLOv8/v11, preds tensor might be [batch, 4 + num_classes + num_mask_coeffs, num_proposals]
    # We transpose it to [batch, num_proposals, 4 + num_classes + num_mask_coeffs]
    preds = torch.from_numpy(preds).transpose(1, 2)
    
    # NMS
    results = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45)
    
    # Scale boxes back to original image size
    # This is a simplified version; full post-processing might involve mask processing too.
    for r in results:
        if r is not None and len(r):
            r[:, :4] = scale_boxes(resized_shape, r[:, :4], original_shape).round()
    return results

def benchmark_pipeline(model_or_engine, image_path, num_runs=50):
    is_trt = isinstance(model_or_engine, TRTInference)
    timings = {'preprocess': [], 'inference': [], 'postprocess': []}
    
    original_img = cv2.imread(str(image_path))
    
    for i in range(num_runs + 10): # +10 for warm-up
        
        # Pre-processing
        start_pre = time.perf_counter()
        blob, original_shape, resized_shape, pad_info = preprocess(original_img)
        end_pre = time.perf_counter()

        # Inference
        start_inf = time.perf_counter()
        if is_trt:
            raw_preds = model_or_engine.infer(blob)
            # Assuming first output is the main prediction tensor
            preds_np = raw_preds[0].reshape(1, 84, -1) # Adjust shape based on model output
        else: # PyTorch model
            raw_preds = model_or_engine(blob, verbose=False)
            preds_np = raw_preds[0].cpu().numpy()
        end_inf = time.perf_counter()
        
        # Post-processing
        start_post = time.perf_counter()
        _ = postprocess(preds_np, original_shape, resized_shape, pad_info)
        end_post = time.perf_counter()

        if i >= 10: # Store timings after warm-up
            timings['preprocess'].append((end_pre - start_pre) * 1000)
            timings['inference'].append((end_inf - start_inf) * 1000)
            timings['postprocess'].append((end_post - start_post) * 1000)

    avg_timings = {k: np.mean(v) for k, v in timings.items()}
    avg_timings['total'] = sum(avg_timings.values())
    return avg_timings

if __name__ == '__main__':
    # --- Config ---
    PT_MODEL_PATH = Path("runs/train/carla_segmentation_nano/weights/best.pt")
    TRT_ENGINE_PATH = Path("runs/train/carla_segmentation_nano/weights/best.engine")
    SAMPLE_IMAGE = Path("yolo_dataset/test/images/day_00001.png")

    # --- Benchmark PyTorch ---
    print("--- Benchmarking PyTorch Pipeline ---")
    if PT_MODEL_PATH.exists():
        pt_model = YOLO(PT_MODEL_PATH)
        pt_timings = benchmark_pipeline(pt_model, SAMPLE_IMAGE)
        print(f"Pre-processing: {pt_timings['preprocess']:.2f} ms")
        print(f"Inference:      {pt_timings['inference']:.2f} ms")
        print(f"Post-processing:{pt_timings['postprocess']:.2f} ms")
        print(f"Total:          {pt_timings['total']:.2f} ms")
    else:
        print(f"PyTorch model not found at {PT_MODEL_PATH}")

    # --- Benchmark TensorRT ---
    print("\n--- Benchmarking TensorRT Pipeline ---")
    if TRT_ENGINE_PATH.exists():
        trt_engine = TRTInference(str(TRT_ENGINE_PATH))
        trt_timings = benchmark_pipeline(trt_engine, SAMPLE_IMAGE)
        print(f"Pre-processing: {trt_timings['preprocess']:.2f} ms")
        print(f"Inference:      {trt_timings['inference']:.2f} ms")
        print(f"Post-processing:{trt_timings['postprocess']:.2f} ms")
        print(f"Total:          {trt_timings['total']:.2f} ms")
    else:
        print(f"TensorRT engine not found at {TRT_ENGINE_PATH}. Please generate it with trtexec.")
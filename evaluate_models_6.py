# evaluate_models.py
import os
import time
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

def evaluate_all_models():
    """
    Evaluates trained YOLOv11 models on different weather conditions.
    Reports mAP50-95, mAP50, and inference time.
    """
    # --- Configuration ---
    # Path to the base directory where training runs are saved
    runs_path = Path("runs/train")
    
    # Path to the weather-specific dataset configurations
    weather_test_path = Path("yolo_dataset/weather_test_sets")
    
    # Models to evaluate (assuming they are in subfolders of runs_path)
    model_variants = {
        "YOLOv11n-seg": runs_path / "carla_segmentation_nano/weights/best.pt",
        "YOLOv11m-seg": runs_path / "carla_segmentation_medium/weights/best.pt",
        "YOLOv11l-seg": runs_path / "carla_segmentation_large/weights/best.pt",
    }
    
    weather_conditions = ["day", "night", "rain", "fog"]
    results = []

    print("Starting model evaluation...")

    for model_name, model_path in model_variants.items():
        if not model_path.exists():
            print(f"Warning: Model not found at {model_path}. Skipping.")
            continue

        print(f"\n--- Loading Model: {model_name} ---")
        model = YOLO(model_path)

        for weather in weather_conditions:
            print(f"  - Evaluating on weather: {weather.capitalize()}")
            weather_yaml = weather_test_path / f"{weather}_config.yaml"
            if not weather_yaml.exists():
                print(f"    Warning: YAML file not found at {weather_yaml}. Skipping.")
                continue

            # 1. Evaluate mAP
            metrics = model.val(data=str(weather_yaml), verbose=False)
            map50_95 = metrics.seg.map
            map50 = metrics.seg.map50
            
            # 2. Evaluate Inference Time
            weather_images_dir = weather_test_path / weather / "images"
            sample_images = list(weather_images_dir.glob("*.png"))[:50] # Use 50 images for stable timing
            
            if not sample_images:
                print(f"    Warning: No sample images found for {weather}. Skipping inference time test.")
                avg_inference_time = -1
            else:
                start_time = time.time()
                for img_path in sample_images:
                    model(img_path, verbose=False)
                end_time = time.time()
                avg_inference_time = (end_time - start_time) / len(sample_images) * 1000  # in ms

            results.append({
                "Model": model_name,
                "Weather": weather.capitalize(),
                "mAP@50-95": f"{map50_95:.4f}",
                "mAP@50": f"{map50:.4f}",
                "Inference Time (ms)": f"{avg_inference_time:.2f}"
            })

    # --- Display Results ---
    if not results:
        print("\nNo results to display. Please check your model and data paths.")
        return
        
    df = pd.DataFrame(results)
    print("\n\n--- Overall Evaluation Summary ---")
    print(df.to_string(index=False))
    
    # Save results to a CSV file
    output_csv = "evaluation_summary.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")

if __name__ == '__main__':
    evaluate_all_models()
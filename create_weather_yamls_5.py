import os
import shutil
import yaml
from pathlib import Path

def create_weather_specific_yamls(dataset_path: Path):
    """
    Creates separate test sets and YAML files for each weather condition.
    Assumes the original image filenames are in the format: weather_frameid.png
    """
    print("Creating weather-specific test sets and YAML files...")
    
    main_yaml_path = dataset_path / "dataset.yaml"
    if not main_yaml_path.exists():
        print(f"Error: Main dataset YAML file not found at {main_yaml_path}")
        return

    with open(main_yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    test_images_path = dataset_path / config['test']
    weather_conditions = ["day", "night", "rain", "fog"]
    
    # Create a base path for weather-specific test sets
    weather_test_base = dataset_path / "weather_test_sets"
    weather_test_base.mkdir(exist_ok=True)
    
    # Create subdirectories for each weather condition
    for weather in weather_conditions:
        (weather_test_base / weather / "images").mkdir(parents=True, exist_ok=True)
        (weather_test_base / weather / "labels").mkdir(parents=True, exist_ok=True)

    print(f"Copying files to weather-specific directories from {test_images_path}...")
    # Copy images and labels to their respective weather folders
    for img_file in test_images_path.glob("*.png"):
        for weather in weather_conditions:
            if img_file.name.startswith(weather):
                # Copy image
                shutil.copy(img_file, weather_test_base / weather / "images" / img_file.name)
                
                # Copy corresponding label
                label_file = img_file.with_suffix('.txt').name
                src_label_path = test_images_path.parent / "labels" / label_file
                if src_label_path.exists():
                    shutil.copy(src_label_path, weather_test_base / weather / "labels" / label_file)
                break
    
    print("Creating YAML files...")
    # Create a new YAML file for each weather condition
    for weather in weather_conditions:
        weather_config = config.copy()
        weather_config['path'] = str(weather_test_base.absolute())
        # The 'val' path in the new YAML points to the specific weather test set
        weather_config['val'] = f"{weather}/images" 
        # Remove train and test paths as they are not needed for validation
        weather_config.pop('train', None)
        weather_config.pop('test', None)
        
        weather_yaml_path = weather_test_base / f"{weather}_config.yaml"
        with open(weather_yaml_path, 'w') as f:
            yaml.dump(weather_config, f, default_flow_style=False)
        
        print(f"Created {weather_yaml_path}")

if __name__ == '__main__':
    # Path to your main YOLO dataset directory (the one with dataset.yaml)
    yolo_dataset_path = Path("yolo_dataset") 
    create_weather_specific_yamls(yolo_dataset_path)
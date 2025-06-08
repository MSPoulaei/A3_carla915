import os
import shutil
import yaml
from pathlib import Path

def create_weather_specific_yamls(dataset_path: Path):
    """
    Creates separate test sets and YAML configuration files for different weather
    conditions based on image filenames.

    This function reads the main dataset configuration, identifies images in the
    test set belonging to each weather condition, and copies them and their
    corresponding labels into new directories. It then generates a new YAML
    file for each weather condition.

    Args:
        dataset_path (Path): The root path of the YOLO dataset. This directory
                             should contain 'train', 'val', 'test' folders and a
                             'dataset.yaml' file.
    """
    print("🚀 Starting to create weather-specific test sets and YAML files...")

    # --- 1. Load Main Configuration ---
    main_yaml_path = dataset_path / "dataset.yaml"
    if not main_yaml_path.exists():
        print(f"❌ Error: Main dataset YAML file not found at '{main_yaml_path}'")
        return

    with open(main_yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- 2. Define Paths and Weather Conditions ---
    # The original test images path, e.g., 'yolo_dataset/test/images'
    original_test_img_path = dataset_path / config['test']
    original_test_lbl_path = original_test_img_path.parent / "labels"
    
    # Define the weather prefixes to look for in filenames
    weather_conditions = ["day", "night", "rain", "fog"]
    
    # Define the base path for our new weather-specific test sets
    weather_test_base = dataset_path / "weather_test_sets"
    print(f"📁 New test sets will be created in: '{weather_test_base}'")

    # --- 3. Create Directories and Copy Files ---
    for weather in weather_conditions:
        # Create subdirectories for each weather condition's images and labels
        (weather_test_base / weather / "images").mkdir(parents=True, exist_ok=True)
        (weather_test_base / weather / "labels").mkdir(parents=True, exist_ok=True)

    print(f"🔄 Copying test files from '{original_test_img_path}'...")
    
    # Handle multiple image extensions gracefully
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.webp']
    
    # Iterate through all files in the original test images directory
    for img_file in original_test_img_path.iterdir():
        if img_file.suffix.lower() not in image_extensions:
            continue # Skip non-image files

        for weather in weather_conditions:
            if img_file.name.startswith(weather):
                # Define source and destination paths
                src_img_path = img_file
                dest_img_path = weather_test_base / weather / "images" / img_file.name
                
                src_label_path = original_test_lbl_path / img_file.with_suffix('.txt').name
                dest_label_path = weather_test_base / weather / "labels" / src_label_path.name

                # Copy image file
                shutil.copy(src_img_path, dest_img_path)

                # Copy corresponding label file if it exists
                if src_label_path.exists():
                    shutil.copy(src_label_path, dest_label_path)
                else:
                    print(f"⚠️ Warning: Label for '{img_file.name}' not found at '{src_label_path}'")
                
                break # Move to the next image file once weather is matched

    # --- 4. Create Weather-Specific YAML Files ---
    print("📝 Creating new YAML configuration files...")
    for weather in weather_conditions:
        # Start with a copy of the original configuration
        weather_config = config.copy()

        # The 'path' should be the main dataset directory, relative to the new YAML file
        # weather_config['path'] = '..'
        
        # Keep the original 'train' and 'val' paths
        # They are already relative to the dataset root, which is now '..'
        
        # Update the 'test' path to point to the new weather-specific test set
        # This path is relative to the 'path' key
        weather_config['test'] = f"weather_test_sets/{weather}/images"
        
        # Define the path for the new YAML file
        weather_yaml_path = weather_test_base / f"{weather}_dataset.yaml"
        
        # Write the new configuration to the YAML file
        with open(weather_yaml_path, 'w') as f:
            yaml.dump(weather_config, f, default_flow_style=False, sort_keys=False)
              
        print(f"✅ Successfully created '{weather_yaml_path}'")

    print("\n🎉 All done!")


if __name__ == '__main__':
    # Define the path to your main YOLO dataset directory
    # This is the folder that contains train/, val/, test/, and dataset.yaml
    yolo_dataset_path = Path("yolo_dataset") 
    
    # Run the function
    create_weather_specific_yamls(yolo_dataset_path)
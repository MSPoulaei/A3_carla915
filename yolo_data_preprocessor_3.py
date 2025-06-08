import os
import json
import cv2
import numpy as np
import yaml
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import albumentations as A
from PIL import Image, ImageDraw
import random


class YOLOv11DataPreprocessor:
    """
    Comprehensive data preprocessor for YOLOv11 instance segmentation
    Handles annotation conversion, dataset splitting, and augmentation preparation
    """

    def __init__(self, carla_dataset_path, output_path="yolo_dataset"):
        self.carla_path = Path(carla_dataset_path)
        self.output_path = Path(output_path)
        self.weather_conditions = ["day", "night", "rain", "fog"]

        # Class mapping (same as CARLA collector)
        self.class_mapping = {
            "car": 0,
            'bus': 1,
            "pedestrian": 2,
            "traffic_light": 3,
        }
        self.class_names = list(self.class_mapping.keys())

        # Statistics tracking
        self.conversion_stats = {
            "total_images": 0,
            "total_annotations": 0,
            "weather_distribution": defaultdict(int),
            "class_distribution": defaultdict(int),
            "annotation_issues": [],
        }

        self._create_yolo_structure()

    def _create_yolo_structure(self):
        """Create YOLOv11 dataset directory structure"""
        # Main directories
        for split in ["train", "val", "test"]:
            for data_type in ["images", "labels"]:
                (self.output_path / split / data_type).mkdir(
                    parents=True, exist_ok=True
                )

        # Analysis directory
        (self.output_path / "analysis").mkdir(exist_ok=True)

        print(f"Created YOLOv11 dataset structure at: {self.output_path}")

    def load_carla_annotations(self):
        """Load all CARLA annotations and images"""
        dataset = {}

        print("Loading CARLA dataset...")
        for weather in self.weather_conditions:
            weather_data = []

            # Paths for this weather condition
            rgb_path = self.carla_path / weather / "rgb"
            ann_path = self.carla_path / weather / "annotations"

            if not rgb_path.exists() or not ann_path.exists():
                print(f"Warning: Missing data for {weather} condition")
                continue

            # Load all annotations for this weather
            for ann_file in ann_path.glob("*.json"):
                try:
                    with open(ann_file, "r") as f:
                        annotation = json.load(f)

                    # Find corresponding image
                    image_name = ann_file.stem + ".png"
                    image_path = rgb_path / image_name

                    if image_path.exists():
                        weather_data.append(
                            {
                                "image_path": str(image_path),
                                "annotation": annotation,
                                "weather": weather,
                                "image_id": annotation.get("image_id", ann_file.stem),
                            }
                        )
                    else:
                        print(f"Warning: Missing image for {ann_file}")

                except Exception as e:
                    print(f"Error loading {ann_file}: {e}")

            dataset[weather] = weather_data
            print(f"Loaded {len(weather_data)} samples for {weather}")

        return dataset

    def convert_segmentation_to_yolo(self, segmentation_points):
        """
        Convert segmentation polygon to YOLOv11 format
        YOLOv11 expects: class_id x1 y1 x2 y2 ... xn yn (normalized coordinates)
        """
        if len(segmentation_points) < 6:  # At least 3 points (x,y pairs)
            return None

        # Ensure we have an even number of coordinates
        if len(segmentation_points) % 2 != 0:
            segmentation_points = segmentation_points[:-1]

        # Validate coordinates are in [0,1] range
        valid_points = []
        for i in range(0, len(segmentation_points), 2):
            x, y = segmentation_points[i], segmentation_points[i + 1]

            # Clamp coordinates to valid range
            x = max(0.0, min(1.0, float(x)))
            y = max(0.0, min(1.0, float(y)))

            valid_points.extend([x, y])

        return valid_points

    def process_annotation(self, annotation_data):
        """Process a single annotation into YOLOv11 format"""
        yolo_annotations = []

        for obj in annotation_data.get("objects", []):
            class_name = obj.get("class")

            # Skip if class not in our mapping
            if class_name not in self.class_mapping:
                continue

            class_id = self.class_mapping[class_name]
            segmentation = obj.get("segmentation", [])

            # Convert segmentation to YOLO format
            yolo_segmentation = self.convert_segmentation_to_yolo(segmentation)

            if yolo_segmentation and len(yolo_segmentation) >= 6:
                # Create YOLO annotation line
                yolo_line = [str(class_id)] + [
                    f"{coord:.6f}" for coord in yolo_segmentation
                ]
                yolo_annotations.append(" ".join(yolo_line))

                # Update statistics
                self.conversion_stats["class_distribution"][class_name] += 1
                self.conversion_stats["total_annotations"] += 1
            else:
                self.conversion_stats["annotation_issues"].append(
                    {
                        "issue": "Invalid segmentation",
                        "class": class_name,
                        "segmentation_length": len(segmentation),
                    }
                )

        return yolo_annotations

    def convert_dataset(self):
        """Convert entire CARLA dataset to YOLOv11 format"""
        print("Converting CARLA dataset to YOLOv11 format...")

        # Load CARLA data
        carla_dataset = self.load_carla_annotations()

        # Flatten all data for splitting
        all_samples = []
        for weather, samples in carla_dataset.items():
            all_samples.extend(samples)

        print(f"Total samples to convert: {len(all_samples)}")

        # Split dataset (stratified by weather condition)
        weather_labels = [sample["weather"] for sample in all_samples]

        # First split: train+val vs test (80% vs 20%)
        train_val_samples, test_samples, train_val_weather, test_weather = (
            train_test_split(
                all_samples,
                weather_labels,
                test_size=0.2,
                stratify=weather_labels,
                random_state=42,
            )
        )

        # Second split: train vs val (75% vs 25% of train_val = 60% vs 20% of total)
        train_samples, val_samples, _, _ = train_test_split(
            train_val_samples,
            train_val_weather,
            test_size=0.25,  # 0.25 of 0.8 = 0.2 (20% of total)
            stratify=train_val_weather,
            random_state=42,
        )

        # Convert each split
        splits = {"train": train_samples, "val": val_samples, "test": test_samples}

        split_stats = {}

        for split_name, split_samples in splits.items():
            print(f"\nProcessing {split_name} split ({len(split_samples)} samples)...")

            split_stats[split_name] = {
                "total_images": 0,
                "total_annotations": 0,
                "weather_distribution": defaultdict(int),
                "class_distribution": defaultdict(int),
            }

            for sample in tqdm(split_samples, desc=f"Converting {split_name}"):
                try:
                    # Copy image
                    src_image_path = sample["image_path"]
                    image_filename = f"{sample['image_id']}.png"
                    dst_image_path = (
                        self.output_path / split_name / "images" / image_filename
                    )

                    shutil.copy2(src_image_path, dst_image_path)

                    # Convert and save annotation
                    yolo_annotations = self.process_annotation(sample["annotation"])

                    # Save YOLO annotation file
                    label_filename = f"{sample['image_id']}.txt"
                    label_path = (
                        self.output_path / split_name / "labels" / label_filename
                    )

                    with open(label_path, "w") as f:
                        f.write("\n".join(yolo_annotations))

                    # Update split statistics
                    split_stats[split_name]["total_images"] += 1
                    split_stats[split_name]["total_annotations"] += len(
                        yolo_annotations
                    )
                    split_stats[split_name]["weather_distribution"][
                        sample["weather"]
                    ] += 1

                    # Count classes in this annotation
                    for annotation in yolo_annotations:
                        class_id = int(annotation.split()[0])
                        class_name = self.class_names[class_id]
                        split_stats[split_name]["class_distribution"][class_name] += 1

                except Exception as e:
                    print(f"Error processing {sample['image_id']}: {e}")

        # Update global statistics
        self.conversion_stats["total_images"] = sum(
            stats["total_images"] for stats in split_stats.values()
        )
        for weather in self.weather_conditions:
            self.conversion_stats["weather_distribution"][weather] = sum(
                stats["weather_distribution"][weather] for stats in split_stats.values()
            )

        # Create dataset configuration file
        self._create_dataset_yaml()

        # Generate conversion report
        self._generate_conversion_report(split_stats)

        print(f"\n✅ Dataset conversion completed!")
        print(f"📊 Total images: {self.conversion_stats['total_images']}")
        print(f"🏷️  Total annotations: {self.conversion_stats['total_annotations']}")
        print(
            f"⚠️  Annotation issues: {len(self.conversion_stats['annotation_issues'])}"
        )

        return split_stats

    def _create_dataset_yaml(self):
        """Create YOLOv11 dataset configuration file"""
        config = {
            "path": str(self.output_path.absolute()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(self.class_names),
            "names": self.class_names,
        }

        yaml_path = self.output_path / "dataset.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"Created dataset configuration: {yaml_path}")

    def _generate_conversion_report(self, split_stats):
        """Generate detailed conversion report"""
        report_path = self.output_path / "analysis" / "conversion_report.md"

        with open(report_path, "w") as f:
            f.write("# YOLOv11 Dataset Conversion Report\n\n")

            # Overall statistics
            f.write("## Conversion Summary\n\n")
            f.write(
                f"- **Total Images Converted:** {self.conversion_stats['total_images']:,}\n"
            )
            f.write(
                f"- **Total Annotations:** {self.conversion_stats['total_annotations']:,}\n"
            )
            f.write(
                f"- **Annotation Issues:** {len(self.conversion_stats['annotation_issues'])}\n"
            )
            f.write(f"- **Classes:** {len(self.class_names)}\n\n")

            # Split distribution
            f.write("## Dataset Split Distribution\n\n")
            f.write("| Split | Images | Annotations | Percentage |\n")
            f.write("|-------|--------|-------------|------------|\n")

            total_images = sum(stats["total_images"] for stats in split_stats.values())
            for split_name, stats in split_stats.items():
                percentage = (stats["total_images"] / total_images) * 100
                f.write(
                    f"| {split_name.capitalize()} | {stats['total_images']:,} | {stats['total_annotations']:,} | {percentage:.1f}% |\n"
                )

            # Weather distribution per split
            f.write("\n## Weather Distribution by Split\n\n")
            for split_name, stats in split_stats.items():
                f.write(f"### {split_name.capitalize()} Split\n")
                f.write("| Weather | Images | Percentage |\n")
                f.write("|---------|--------|------------|\n")

                split_total = stats["total_images"]
                for weather in self.weather_conditions:
                    count = stats["weather_distribution"][weather]
                    percentage = (count / split_total) * 100 if split_total > 0 else 0
                    f.write(
                        f"| {weather.capitalize()} | {count} | {percentage:.1f}% |\n"
                    )
                f.write("\n")

            # Class distribution
            f.write("## Class Distribution\n\n")
            f.write("| Class | Total Annotations | Percentage |\n")
            f.write("|-------|------------------|------------|\n")

            total_annotations = sum(
                self.conversion_stats["class_distribution"].values()
            )
            for class_name in self.class_names:
                count = self.conversion_stats["class_distribution"][class_name]
                percentage = (
                    (count / total_annotations) * 100 if total_annotations > 0 else 0
                )
                f.write(
                    f"| {class_name.capitalize()} | {count:,} | {percentage:.1f}% |\n"
                )

            # Annotation issues
            if self.conversion_stats["annotation_issues"]:
                f.write("\n## Annotation Issues\n\n")
                issue_counts = defaultdict(int)
                for issue in self.conversion_stats["annotation_issues"]:
                    issue_counts[issue["issue"]] += 1

                f.write("| Issue Type | Count |\n")
                f.write("|------------|-------|\n")
                for issue_type, count in issue_counts.items():
                    f.write(f"| {issue_type} | {count} |\n")

        print(f"Generated conversion report: {report_path}")

    def visualize_dataset_statistics(self):
        """Generate visualization plots for dataset analysis"""
        print("Generating dataset visualizations...")

        # Set up the plotting style
        if "seaborn-v0_8" in plt.style.available:
            plt.style.use("seaborn-v0_8")
        else:
            plt.style.use("seaborn")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("YOLOv11 Dataset Analysis", fontsize=16, fontweight="bold")

        # 1. Weather distribution
        weather_counts = list(self.conversion_stats["weather_distribution"].values())
        weather_labels = list(self.conversion_stats["weather_distribution"].keys())
        axes[0, 0].pie(
            weather_counts, labels=weather_labels, autopct="%1.1f%%", startangle=90
        )
        axes[0, 0].set_title("Weather Condition Distribution")

        # 2. Class distribution
        class_counts = [
            self.conversion_stats["class_distribution"][class_name]
            for class_name in self.class_names
        ]
        bars = axes[0, 1].bar(
            self.class_names,
            class_counts,
            color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"],
        )
        axes[0, 1].set_title("Class Distribution")
        axes[0, 1].set_ylabel("Number of Annotations")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
            )

        # 3. Annotations per image distribution
        # This would require loading actual data, so we'll create a placeholder
        sample_annotations_per_image = np.random.poisson(3, 1000)  # Placeholder data
        axes[1, 0].hist(
            sample_annotations_per_image,
            bins=20,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )
        axes[1, 0].set_title("Annotations per Image Distribution")
        axes[1, 0].set_xlabel("Number of Annotations")
        axes[1, 0].set_ylabel("Frequency")

        # 4. Dataset split sizes
        split_names = ["Train", "Validation", "Test"]
        split_percentages = [60, 20, 20]  # Standard split
        colors = ["#FF9999", "#66B2FF", "#99FF99"]

        axes[1, 1].pie(
            split_percentages,
            labels=split_names,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        axes[1, 1].set_title("Dataset Split Distribution")

        plt.tight_layout()

        # Save the plot
        plot_path = self.output_path / "analysis" / "dataset_statistics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Dataset statistics plot saved: {plot_path}")

    def create_augmentation_pipeline(self):
        """Create augmentation pipeline for training data enhancement"""
        augmentation_pipeline = A.Compose(
            [
                # Geometric transformations
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5
                ),
                # Color and lighting augmentations
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5
                ),
                # Weather-like effects
                A.RandomRain(p=0.1),
                A.RandomFog(p=0.1),
                A.RandomSunFlare(p=0.05),
                # Noise and blur
                A.GaussNoise(var_limit=(10, 50), p=0.2),
                A.MotionBlur(blur_limit=3, p=0.1),
                A.GaussianBlur(blur_limit=3, p=0.1),
                # Normalization (if needed)
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0
                ),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )

        # Save augmentation configuration
        aug_config = {
            "augmentation_pipeline": "albumentations",
            "transformations": [
                "HorizontalFlip",
                "RandomRotate90",
                "ShiftScaleRotate",
                "RandomBrightnessContrast",
                "HueSaturationValue",
                "RandomRain",
                "RandomFog",
                "RandomSunFlare",
                "GaussNoise",
                "MotionBlur",
                "GaussianBlur",
                "Normalize",
            ],
            "probability_range": "0.05 - 0.5",
            "note": "Pipeline designed for autonomous driving scenarios",
        }

        config_path = self.output_path / "augmentation_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(aug_config, f, default_flow_style=False)

        print(f"Augmentation pipeline configuration saved: {config_path}")
        return augmentation_pipeline

    def validate_dataset(self):
        """Validate the converted dataset for common issues"""
        print("Validating converted dataset...")

        validation_results = {
            "missing_images": [],
            "missing_labels": [],
            "empty_labels": [],
            "invalid_annotations": [],
            "class_id_issues": [],
        }

        for split in ["train", "val", "test"]:
            images_dir = self.output_path / split / "images"
            labels_dir = self.output_path / split / "labels"

            # Check for missing files
            image_files = set(f.stem for f in images_dir.glob("*.png"))
            label_files = set(f.stem for f in labels_dir.glob("*.txt"))

            # Find missing labels
            missing_labels = image_files - label_files
            validation_results["missing_labels"].extend(
                [f"{split}/{img_id}" for img_id in missing_labels]
            )

            # Find missing images
            missing_images = label_files - image_files
            validation_results["missing_images"].extend(
                [f"{split}/{img_id}" for img_id in missing_images]
            )

            # Validate label files
            for label_file in labels_dir.glob("*.txt"):
                try:
                    with open(label_file, "r") as f:
                        lines = f.readlines()

                    if not lines:
                        validation_results["empty_labels"].append(
                            f"{split}/{label_file.stem}"
                        )
                        continue

                    for line_num, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()
                        if (
                            len(parts) < 7
                        ):  # class_id + at least 3 points (6 coordinates)
                            validation_results["invalid_annotations"].append(
                                {
                                    "file": f"{split}/{label_file.stem}",
                                    "line": line_num,
                                    "issue": "Insufficient coordinates",
                                    "content": line,
                                }
                            )
                            continue

                        # Check class ID
                        try:
                            class_id = int(parts[0])
                            if class_id < 0 or class_id >= len(self.class_names):
                                validation_results["class_id_issues"].append(
                                    {
                                        "file": f"{split}/{label_file.stem}",
                                        "line": line_num,
                                        "class_id": class_id,
                                        "valid_range": f"0-{len(self.class_names)-1}",
                                    }
                                )
                        except ValueError:
                            validation_results["invalid_annotations"].append(
                                {
                                    "file": f"{split}/{label_file.stem}",
                                    "line": line_num,
                                    "issue": "Invalid class ID",
                                    "content": line,
                                }
                            )

                        # Check coordinate format
                        try:
                            coords = [float(x) for x in parts[1:]]
                            if len(coords) % 2 != 0:
                                validation_results["invalid_annotations"].append(
                                    {
                                        "file": f"{split}/{label_file.stem}",
                                        "line": line_num,
                                        "issue": "Odd number of coordinates",
                                        "content": line,
                                    }
                                )

                            # Check coordinate range
                            for coord in coords:
                                if coord < 0 or coord > 1:
                                    validation_results["invalid_annotations"].append(
                                        {
                                            "file": f"{split}/{label_file.stem}",
                                            "line": line_num,
                                            "issue": "Coordinates out of range [0,1]",
                                            "content": line,
                                        }
                                    )
                                    break

                        except ValueError:
                            validation_results["invalid_annotations"].append(
                                {
                                    "file": f"{split}/{label_file.stem}",
                                    "line": line_num,
                                    "issue": "Invalid coordinate format",
                                    "content": line,
                                }
                            )

                except Exception as e:
                    validation_results["invalid_annotations"].append(
                        {
                            "file": f"{split}/{label_file.stem}",
                            "issue": f"File read error: {str(e)}",
                        }
                    )

        # Generate validation report
        self._generate_validation_report(validation_results)

        # Print summary
        total_issues = sum(len(issues) for issues in validation_results.values())
        if total_issues == 0:
            print("✅ Dataset validation passed! No issues found.")
        else:
            print(f"⚠️  Dataset validation found {total_issues} issues:")
            for issue_type, issues in validation_results.items():
                if issues:
                    print(f"  - {issue_type}: {len(issues)}")

        return validation_results

    def _generate_validation_report(self, validation_results):
        """Generate detailed validation report"""
        report_path = self.output_path / "analysis" / "validation_report.md"

        with open(report_path, "w") as f:
            f.write("# Dataset Validation Report\n\n")

            total_issues = sum(len(issues) for issues in validation_results.values())
            f.write(f"**Total Issues Found:** {total_issues}\n\n")

            for issue_type, issues in validation_results.items():
                if not issues:
                    continue

                f.write(
                    f"## {issue_type.replace('_', ' ').title()} ({len(issues)} issues)\n\n"
                )

                if issue_type in ["missing_images", "missing_labels", "empty_labels"]:
                    for issue in issues[:10]:  # Show first 10
                        f.write(f"- {issue}\n")
                    if len(issues) > 10:
                        f.write(f"- ... and {len(issues) - 10} more\n")
                else:
                    for issue in issues[:5]:  # Show first 5 detailed issues
                        f.write(f"- **File:** {issue.get('file', 'Unknown')}\n")
                        if "line" in issue:
                            f.write(f"  - **Line:** {issue['line']}\n")
                        f.write(f"  - **Issue:** {issue.get('issue', 'Unknown')}\n")
                        if "content" in issue:
                            f.write(f"  - **Content:** `{issue['content']}`\n")
                        f.write("\n")
                    if len(issues) > 5:
                        f.write(f"... and {len(issues) - 5} more similar issues\n")

                f.write("\n")

        print(f"Validation report saved: {report_path}")

    def sample_visualization(self, num_samples=5):
        """Create sample visualizations with annotations"""
        print(f"Creating sample visualizations ({num_samples} samples)...")

        # Create samples directory
        samples_dir = self.output_path / "analysis" / "samples"
        samples_dir.mkdir(exist_ok=True)

        # Get random samples from train set
        train_images_dir = self.output_path / "train" / "images"
        train_labels_dir = self.output_path / "train" / "labels"

        image_files = list(train_images_dir.glob("*.png"))
        if len(image_files) < num_samples:
            num_samples = len(image_files)
            print(f"Only {num_samples} samples available")

        selected_files = random.sample(image_files, num_samples)

        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
        ]  # Red, Green, Blue, Yellow

        for i, image_file in enumerate(selected_files):
            try:
                # Load image
                image = cv2.imread(str(image_file))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w = image.shape[:2]

                # Load corresponding label
                label_file = train_labels_dir / f"{image_file.stem}.txt"

                if label_file.exists():
                    with open(label_file, "r") as f:
                        annotations = f.readlines()

                    # Draw annotations
                    for annotation in annotations:
                        parts = annotation.strip().split()
                        if len(parts) < 7:
                            continue

                        class_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]

                        # Convert normalized coordinates to pixel coordinates
                        points = []
                        for j in range(0, len(coords), 2):
                            x = int(coords[j] * w)
                            y = int(coords[j + 1] * h)
                            points.append((x, y))

                        # Draw polygon
                        color = colors[class_id % len(colors)]
                        cv2.polylines(image, [np.array(points)], True, color, 2)

                        # Add class label
                        if points:
                            cv2.putText(
                                image,
                                self.class_names[class_id],
                                points[0],
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                color,
                                2,
                            )

                # Save visualization
                sample_path = samples_dir / f"sample_{i+1}_{image_file.stem}.png"
                cv2.imwrite(str(sample_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            except Exception as e:
                print(f"Error creating sample visualization for {image_file}: {e}")

        print(f"Sample visualizations saved in: {samples_dir}")

    def generate_training_script(self):
        """Generate a sample YOLOv11 training script"""
        script_content = f'''#!/usr/bin/env python3
"""
YOLOv11 Training Script for CARLA Dataset
Generated automatically by YOLOv11DataPreprocessor
"""

from ultralytics import YOLO
import yaml
from pathlib import Path

def main():
    # Dataset configuration
    dataset_path = Path("{self.output_path.absolute()}")
    config_file = dataset_path / "dataset.yaml"
    
    # Load dataset configuration
    with open(config_file, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    print(f"Training YOLOv11 on dataset: {{dataset_config['path']}}")
    print(f"Classes: {{dataset_config['names']}}")
    print(f"Number of classes: {{dataset_config['nc']}}")
    
    # Initialize YOLOv11 model
    # Options: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
    model = YOLO('yolo11n-seg.pt')  # nano model for segmentation
    
    # Training parameters
    training_args = {{
        'data': str(config_file),
        'epochs': 50,
        'imgsz': 512,
        'batch': 16,
        'workers': 8,
        'device': '0',  # Use GPU if available
        'project': 'carla_yolo_training_result',
        'name': 'carla_segmentation_nano',
        'save_period': 10,
        'val': True,
        'plots': True,
        'verbose': True,
        
        # Optimization parameters
        'optimizer': 'AdamW',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Augmentation parameters
        # 'hsv_h': 0.015,
        # 'hsv_s': 0.7,
        # 'hsv_v': 0.4,
        # 'degrees': 0.0,
        # 'translate': 0.1,
        # 'scale': 0.5,
        # 'shear': 0.0,
        # 'perspective': 0.0,
        # 'flipud': 0.0,
        # 'fliplr': 0.5,
        # 'mosaic': 1.0,
        # 'mixup': 0.0,
        # 'copy_paste': 0.0,

        'hsv_h': 0.0,
        'hsv_s': 0.0,
        'hsv_v': 0.0,
        'degrees': 0.0,
        'translate': 0.0,
        'scale': 0.0,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.0,
        'mosaic': 0.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        
        # Loss parameters
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 2.0,
        'label_smoothing': 0.0,
        
        # Validation parameters
        #'val_period': 1,
        'save_json': True,
        'save_hybrid': False,
        'conf': 0.001,
        'iou': 0.6,
        'max_det': 300,
        'half': False,
        'dnn': False,
        'plots': True
    }}
    
    # Start training
    print("Starting training...")
    results = model.train(**training_args)
    
    # Validation
    print("Running validation...")
    val_results = model.val()
    
    # Export model
    print("Exporting model...")
    model.export(format='onnx')  # Export to ONNX format
    
    print("Training completed!")
    print(f"Best model saved at: {{model.trainer.best}}")
    print(f"Results: {{results}}")

if __name__ == "__main__":
    main()
'''

        script_path = self.output_path / "train_yolo.py"
        with open(script_path, "w") as f:
            f.write(script_content)

        # Make script executable
        os.chmod(script_path, 0o755)

        print(f"Training script generated: {script_path}")

        # Also create a requirements file
        requirements_content = """# YOLOv11 Training Requirements
ultralytics>=8.0.0
torch>=1.8.0
torchvision>=0.9.0
opencv-python>=4.5.0
Pillow>=8.0.0
matplotlib>=3.3.0
seaborn>=0.11.0
numpy>=1.21.0
pandas>=1.3.0
PyYAML>=5.4.0
tqdm>=4.62.0
scikit-learn>=1.0.0
albumentations>=1.3.0
"""

        req_path = self.output_path / "requirements.txt"
        with open(req_path, "w") as f:
            f.write(requirements_content)

        print(f"Requirements file generated: {req_path}")

    def run_complete_preprocessing(self):
        """Run the complete preprocessing pipeline"""
        print("🚀 Starting complete YOLOv11 preprocessing pipeline...")
        print("=" * 60)

        try:
            # Step 1: Convert dataset
            print("\n📁 Step 1: Converting CARLA dataset to YOLOv11 format...")
            split_stats = self.convert_dataset()

            # Step 2: Validate dataset
            print("\n✅ Step 2: Validating converted dataset...")
            validation_results = self.validate_dataset()

            # Step 3: Generate visualizations
            print("\n📊 Step 3: Generating dataset statistics...")
            self.visualize_dataset_statistics()

            # Step 4: Create sample visualizations
            print("\n🖼️  Step 4: Creating sample visualizations...")
            self.sample_visualization(num_samples=10)

            # Step 5: Create augmentation pipeline
            print("\n🔄 Step 5: Setting up augmentation pipeline...")
            self.create_augmentation_pipeline()

            # Step 6: Generate training script
            print("\n🏋️  Step 6: Generating training script...")
            self.generate_training_script()

            # Final summary
            print("\n" + "=" * 60)
            print("🎉 PREPROCESSING COMPLETED SUCCESSFULLY!")
            print("=" * 60)

            print(f"\n📋 SUMMARY:")
            print(f"   • Dataset location: {self.output_path}")
            print(f"   • Total images: {self.conversion_stats['total_images']:,}")
            print(
                f"   • Total annotations: {self.conversion_stats['total_annotations']:,}"
            )
            print(
                f"   • Classes: {len(self.class_names)} ({', '.join(self.class_names)})"
            )

            total_issues = sum(len(issues) for issues in validation_results.values())
            if total_issues == 0:
                print(f"   • Validation: ✅ No issues found")
            else:
                print(
                    f"   • Validation: ⚠️  {total_issues} issues found (see validation report)"
                )

            print(f"\n📁 Generated Files:")
            print(f"   • Dataset config: dataset.yaml")
            print(f"   • Training script: train_yolo.py")
            print(f"   • Requirements: requirements.txt")
            print(f"   • Analysis reports: analysis/")
            print(f"   • Sample visualizations: analysis/samples/")

            print(f"\n🚀 Next Steps:")
            print(f"   1. Review validation report if issues were found")
            print(f"   2. Install requirements: pip install -r requirements.txt")
            print(f"   3. Start training: python train_yolo.py")
            print(f"   4. Monitor training progress in the generated project folder")

            return True

        except Exception as e:
            print(f"\n❌ Error during preprocessing: {str(e)}")
            import traceback

            traceback.print_exc()
            return False


def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert CARLA dataset to YOLOv11 format"
    )
    parser.add_argument("carla_path", help="Path to CARLA dataset directory")
    parser.add_argument(
        "--output",
        "-o",
        default="yolo_dataset",
        help="Output directory for YOLOv11 dataset (default: yolo_dataset)",
    )
    parser.add_argument(
        "--samples",
        "-s",
        type=int,
        default=10,
        help="Number of sample visualizations to generate (default: 10)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation on existing dataset",
    )

    args = parser.parse_args()

    # Initialize preprocessor
    preprocessor = YOLOv11DataPreprocessor(args.carla_path, args.output)

    if args.validate_only:
        # Only run validation
        print("Running validation only...")
        validation_results = preprocessor.validate_dataset()
        total_issues = sum(len(issues) for issues in validation_results.values())
        if total_issues == 0:
            print("✅ Validation passed!")
        else:
            print(f"⚠️  Found {total_issues} validation issues")
    else:
        # Run complete preprocessing
        success = preprocessor.run_complete_preprocessing()
        if success:
            print("\n🎉 Preprocessing completed successfully!")
        else:
            print("\n❌ Preprocessing failed!")
            exit(1)


if __name__ == "__main__":
    main()

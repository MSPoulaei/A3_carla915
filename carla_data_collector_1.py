# carla_data_collector.py

# import sys

# sys.path.append(
#     'E:\\eLearning\\code\\TermArshad2\\ML\\Carla\\WindowsNoEditor\\PythonAPI\\carla\\dist\\carla-0.9.11-py%d.%d-win-amd64.egg' % (sys.version_info.major,
#                                                              sys.version_info.minor))

import carla
import numpy as np
import cv2
import os
import json
import time
from datetime import datetime
import random
from collections import defaultdict
import queue
import threading
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd


class CarlaDataCollector:
    def __init__(self, host="localhost", port=2000, timeout=10.0):
        """Initialize CARLA client and world"""
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.world = self.client.get_world()
        # self.world = self.client.load_world("Town05")
        self.blueprint_library = self.world.get_blueprint_library()

        # Data collection settings
        self.output_dir = "carla_dataset"
        self.image_width = 512
        self.image_height = 512
        self.fov = 90

        # Synchronization queues
        self.rgb_queue = queue.Queue()
        self.seg_queue = queue.Queue()

        # Frame counter
        self.frame_id = 0

        # Weather presets
        self.weather_presets = {
            "day": carla.WeatherParameters.ClearNoon,
            "night": carla.WeatherParameters.ClearNight,
            "rain": carla.WeatherParameters.HardRainNoon,
            "fog": carla.WeatherParameters.CloudyNoon,
        }

        # Modify fog weather to have actual fog
        self.weather_presets["fog"].fog_density = 100
        self.weather_presets["fog"].fog_distance = 10

        # Class mappings for instance segmentation
        # self.class_mapping = {
        #     7: "traffic_light",
        #     12: "pedestrian",
        #     13: "pedestrian",
        #     14: "car",
        #     15: "car",
        #     16: "bus",
        # }
        self.class_mapping = {
            7: "traffic_light",
            12: "pedestrian",
            13: "pedestrian",
            14: "car",
            15: "car",
            16: "bus",
        }
        # 1 road
        # 2 kerare road
        # 3 building
        # 4 wall
        # 5 fence
        # 6 pole
        # 7
        # 8 traffic sign
        # 9 Vegetation
        # 10 Grass
        # 11 sky
        # 12 pedestrian
        # 13
        # 14 car
        # 15
        # 16 bus
        # 17
        # 18 motor
        # 19

        # self.class_mapping[15]="traffic_light"
        # self.class_mapping[19]="pedestrian"
        # self.class_mapping[17]="car"
        # self.class_mapping[18]="bus"
        # for i in range(6):
        #     self.class_mapping[i]="traffic_light"
        # for i in range(6,11):
        #     self.class_mapping[i]="pedestrian"
        # for i in range(11,16):
        #     self.class_mapping[i]="car"
        # for i in range(16,21):
        #     self.class_mapping[i]="bus"

        self.class_mapping_to_catid = {
            "car": 0,
            "bus": 1,
            "pedestrian": 2,
            "traffic_light": 3,
        }

        # Dataset statistics
        self.dataset_stats = {
            "weather_conditions": {},
            "class_instances": defaultdict(int),
            "total_frames": 0,
            "total_instances": 0,
        }

        # Create output directories
        self._create_directories()

    def _create_directories(self):
        """Create directory structure for dataset"""
        for weather in self.weather_presets.keys():
            for data_type in ["rgb", "segmentation", "annotations", "visualizations"]:
                path = os.path.join(self.output_dir, weather, data_type)
                os.makedirs(path, exist_ok=True)

        # Create analysis directory
        os.makedirs(os.path.join(self.output_dir, "analysis"), exist_ok=True)

    def setup_synchronous_mode(self):
        """Enable synchronous mode for consistent data capture"""
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)

    def spawn_ego_vehicle(self):
        """Spawn ego vehicle with cameras"""
        # Get vehicle blueprint
        vehicle_bp = self.blueprint_library.filter("vehicle.tesla.model3")[0]

        # Get spawn point
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)

        # Spawn vehicle
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        # Setup cameras
        self._setup_cameras()

        return self.ego_vehicle

    def _setup_cameras(self):
        """Setup RGB and instance segmentation cameras"""
        # Camera transform (relative to vehicle)
        camera_transform = carla.Transform(
            carla.Location(x=2.5, z=1.5),  # Front of vehicle, elevated
            carla.Rotation(pitch=-10),  # Slight downward angle
        )

        # RGB Camera
        rgb_bp = self.blueprint_library.find("sensor.camera.rgb")
        rgb_bp.set_attribute("image_size_x", str(self.image_width))
        rgb_bp.set_attribute("image_size_y", str(self.image_height))
        rgb_bp.set_attribute("fov", str(self.fov))

        self.rgb_camera = self.world.spawn_actor(
            rgb_bp, camera_transform, attach_to=self.ego_vehicle
        )
        self.rgb_camera.listen(lambda image: self.rgb_queue.put(image))

        # Instance Segmentation Camera
        seg_bp = self.blueprint_library.find("sensor.camera.instance_segmentation")
        seg_bp.set_attribute("image_size_x", str(self.image_width))
        seg_bp.set_attribute("image_size_y", str(self.image_height))
        seg_bp.set_attribute("fov", str(self.fov))

        self.seg_camera = self.world.spawn_actor(
            seg_bp, camera_transform, attach_to=self.ego_vehicle
        )
        self.seg_camera.listen(lambda image: self.seg_queue.put(image))

    def spawn_traffic(self, num_vehicles=100, num_pedestrians=50):
        """Spawn vehicles and pedestrians to create congested traffic"""
        spawn_points = self.world.get_map().get_spawn_points()

        # Spawn vehicles
        vehicles = []

        # Car Blueprints
        car_bps_ids = [
            "vehicle.audi.a2",
            "vehicle.audi.etron",
            "vehicle.audi.tt",
            "vehicle.bmw.grandtourer",
            "vehicle.chevrolet.impala",
            "vehicle.citroen.c3",
            "vehicle.dodge.charger_2020",
            "vehicle.dodge.charger_police",
            "vehicle.dodge.charger_police_2020",
            "vehicle.ford.crown",
            "vehicle.ford.mustang",
            "vehicle.jeep.wrangler_rubicon",
            "vehicle.lincoln.mkz_2017",
            "vehicle.lincoln.mkz_2020",
            "vehicle.mercedes.coupe",
            "vehicle.mercedes.coupe_2020",
            "vehicle.micro.microlino",
            "vehicle.mini.cooper_s",
            "vehicle.mini.cooper_s_2021",
            "vehicle.nissan.micra",
            "vehicle.nissan.patrol",
            "vehicle.nissan.patrol_2021",
            "vehicle.seat.leon",
            "vehicle.tesla.model3",
            "vehicle.toyota.prius",
        ]

        # Bus Blueprints
        bus_bps_ids = [
            "vehicle.mitsubishi.fusorosa",
        ]

        vehicle_bps = self.blueprint_library.filter("vehicle.*")
        car_bps = [bp for bp in vehicle_bps if bp.id in car_bps_ids]
        bus_bps = [bp for bp in vehicle_bps if bp.id in bus_bps_ids]

        for i in range(num_vehicles):
            # 80% cars, 20% buses
            if random.random() < 0.8:
                blueprint = random.choice(car_bps)
            else:
                blueprint = (
                    random.choice(bus_bps) if bus_bps else random.choice(car_bps)
                )

            # Try to spawn vehicle
            spawn_point = random.choice(spawn_points)
            vehicle = self.world.try_spawn_actor(blueprint, spawn_point)
            if vehicle:
                vehicle.set_autopilot(True)
                vehicles.append(vehicle)

        # Spawn pedestrians
        pedestrians = []
        walker_bps = self.blueprint_library.filter("walker.pedestrian.*")
        walker_controller_bp = self.blueprint_library.find("controller.ai.walker")

        # Get spawn points for pedestrians
        walker_spawn_points = []
        for i in range(num_pedestrians):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc:
                spawn_point.location = loc
                walker_spawn_points.append(spawn_point)

        # Spawn walker actors
        batch = []
        for spawn_point in walker_spawn_points:
            walker_bp = random.choice(walker_bps)
            batch.append(carla.command.SpawnActor(walker_bp, spawn_point))

        # Apply batch
        results = self.client.apply_batch_sync(batch, True)
        walkers_list = []
        for i in range(len(results)):
            if results[i].error:
                continue
            walkers_list.append({"id": results[i].actor_id})

        # Spawn walker controllers
        batch = []
        for i in range(len(walkers_list)):
            batch.append(
                carla.command.SpawnActor(
                    walker_controller_bp, carla.Transform(), walkers_list[i]["id"]
                )
            )

        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                continue
            walkers_list[i]["con"] = results[i].actor_id

        # Initialize controllers
        for i in range(len(walkers_list)):
            self.world.tick()

        # Set walker destinations
        all_actors = self.world.get_actors()
        for walker_info in walkers_list:
            walker = all_actors.find(walker_info["id"])
            controller = all_actors.find(walker_info["con"])
            if walker and controller:
                controller.start()
                controller.go_to_location(
                    self.world.get_random_location_from_navigation()
                )
                controller.set_max_speed(1 + random.random())
                pedestrians.append(walker)

        return vehicles, pedestrians

    def process_segmentation_image(self, seg_image, weather_condition, rgb_image):
        """Process instance segmentation image and create annotations"""
        # Convert segmentation image to numpy array
        seg_array = np.frombuffer(seg_image.raw_data, dtype=np.dtype("uint8"))
        seg_array = seg_array.reshape((self.image_height, self.image_width, 4))

        # Instance segmentation uses R channel for instance ID and G for class ID
        instance_ids = seg_array[:, :, 0].astype(np.uint16) + (
            seg_array[:, :, 1].astype(np.uint16) * 256
        )
        class_ids = seg_array[:, :, 2]

        # Create annotation data
        annotations = {
            "image_id": f"{weather_condition}_{self.frame_id:06d}",
            "image_width": self.image_width,
            "image_height": self.image_height,
            "objects": [],
        }

        # Find unique instances
        unique_instances = np.unique(instance_ids)

        for instance_id in unique_instances:
            if instance_id == 0:  # Background
                continue

            # Get mask for this instance
            instance_mask = instance_ids == instance_id

            # Get class ID for this instance
            class_id = class_ids[instance_mask][0]

            # Check if this class is one we're interested in
            if class_id not in self.class_mapping:
                continue

            # Find contours
            contours, _ = cv2.findContours(
                instance_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            if not contours:
                continue

            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Skip very small objects
            if cv2.contourArea(largest_contour) < 100:
                continue

            # Simplify contour
            epsilon = 0.005 * cv2.arcLength(largest_contour, True)
            approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

            # Convert to normalized coordinates
            segmentation = []
            for point in approx_contour:
                x = point[0][0] / self.image_width
                y = point[0][1] / self.image_height
                segmentation.extend([x, y])

            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Create annotation object
            obj = {
                "class": self.class_mapping[class_id],
                "class_id": self.class_mapping_to_catid[self.class_mapping[class_id]],
                "instance_id": int(instance_id),
                "bbox": [x, y, w, h],
                "bbox_normalized": [
                    x / self.image_width,
                    y / self.image_height,
                    w / self.image_width,
                    h / self.image_height,
                ],
                "segmentation": segmentation,
                "area": cv2.contourArea(largest_contour),
            }

            annotations["objects"].append(obj)

            # Update statistics
            self.dataset_stats["class_instances"][self.class_mapping[class_id]] += 1
            self.dataset_stats["total_instances"] += 1

        return annotations, seg_array

    def create_merged_visualization(
        self, rgb_image, seg_array, annotations, weather_condition, frame_name
    ):
        """Create merged visualization of RGB and segmentation with annotations"""
        # Convert RGB image to numpy array
        rgb_array = np.frombuffer(rgb_image.raw_data, dtype=np.uint8).reshape(
            (self.image_height, self.image_width, 4)
        )
        rgb_bgr = rgb_array[:, :, :3].copy()  # Keep BGR format for OpenCV

        # Create segmentation visualization
        seg_vis = self.create_segmentation_visualization(seg_array)

        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f'{weather_condition.capitalize()} - Frame {self.frame_id:06d} - Objects: {len(annotations["objects"])}',
            fontsize=16,
            fontweight="bold",
        )

        # Convert BGR to RGB for matplotlib
        rgb_for_plt = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        # 1. Original RGB image
        axes[0, 0].imshow(rgb_for_plt)
        axes[0, 0].set_title("RGB Image", fontweight="bold")
        axes[0, 0].axis("off")

        # 2. Segmentation visualization
        axes[0, 1].imshow(seg_vis)
        axes[0, 1].set_title("Instance Segmentation", fontweight="bold")
        axes[0, 1].axis("off")

        # 3. RGB with bounding boxes
        rgb_with_boxes = rgb_for_plt.copy()
        class_colors = {
            "car": (255, 0, 0),
            "bus": (0, 255, 0),
            "pedestrian": (0, 0, 255),
            "traffic_light": (255, 255, 0),
        }

        # Draw bounding boxes and labels on RGB image
        for obj in annotations["objects"]:
            bbox = obj["bbox"]
            class_name = obj["class"]
            x, y, w, h = bbox

            color = class_colors.get(class_name, (255, 255, 255))

            # Draw rectangle
            cv2.rectangle(rgb_with_boxes, (x, y), (x + w, y + h), color, 2)

            # Add label
            label = f"{class_name} ({obj['instance_id']})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

            # Background for text
            cv2.rectangle(
                rgb_with_boxes, (x, y - 20), (x + label_size[0], y), color, -1
            )
            cv2.putText(
                rgb_with_boxes,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        axes[1, 0].imshow(rgb_with_boxes)
        axes[1, 0].set_title("RGB with Bounding Boxes", fontweight="bold")
        axes[1, 0].axis("off")

        # 4. Combined overlay (RGB + Segmentation)
        # Create semi-transparent overlay
        alpha = 0.6
        overlay = cv2.addWeighted(rgb_for_plt, alpha, seg_vis, 1 - alpha, 0)

        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title("RGB + Segmentation Overlay", fontweight="bold")
        axes[1, 1].axis("off")

        # Add statistics text
        class_counts = {}
        for obj in annotations["objects"]:
            class_name = obj["class"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        stats_text = "Objects detected:\n"
        for class_name, count in class_counts.items():
            stats_text += f"• {class_name.replace('_', ' ').title()}: {count}\n"

        fig.text(
            0.02,
            0.02,
            stats_text,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
        )

        # Save the merged visualization
        vis_path = os.path.join(
            self.output_dir,
            weather_condition,
            "visualizations",
            f"{frame_name}_merged.png",
        )
        plt.tight_layout()
        plt.savefig(vis_path, dpi=100, bbox_inches="tight")
        plt.close()

        # print(f"   💾 Saved merged visualization: {frame_name}_merged.png")

    def save_frame_data(self, rgb_image, seg_image, annotations, weather_condition):
        """Save RGB image, segmentation, and annotations"""
        frame_name = f"{weather_condition}_{self.frame_id:06d}"

        # Save RGB image
        rgb_array = np.frombuffer(rgb_image.raw_data, dtype=np.uint8).reshape(
            (self.image_height, self.image_width, 4)
        )
        rgb_bgr = rgb_array[:, :, :3]

        rgb_path = os.path.join(
            self.output_dir, weather_condition, "rgb", f"{frame_name}.png"
        )
        cv2.imwrite(rgb_path, rgb_bgr)

        # Save segmentation visualization
        seg_vis = self.create_segmentation_visualization(seg_image)
        seg_path = os.path.join(
            self.output_dir, weather_condition, "segmentation", f"{frame_name}.png"
        )
        cv2.imwrite(seg_path, seg_vis)

        # Save annotations as JSON
        ann_path = os.path.join(
            self.output_dir, weather_condition, "annotations", f"{frame_name}.json"
        )
        with open(ann_path, "w") as f:
            json.dump(annotations, f, indent=2)

        # Save YOLOv11 format annotation
        self.save_yolo_annotation(annotations, weather_condition, frame_name)

        # Create merged visualization every 10 frames
        if self.frame_id % 10 == 0:
            # print(f"   🎨 Creating merged visualization for frame {self.frame_id}")
            self.create_merged_visualization(
                rgb_image, seg_image, annotations, weather_condition, frame_name
            )

    def save_yolo_annotation(self, annotations, weather_condition, frame_name):
        """Save annotations in YOLOv11 format"""
        yolo_path = os.path.join(
            self.output_dir, weather_condition, "annotations", f"{frame_name}.txt"
        )

        with open(yolo_path, "w") as f:
            for obj in annotations["objects"]:
                # YOLOv11 format: class_id x1 y1 x2 y2 ... xn yn (normalized)
                line = f"{obj['class_id']}"
                for coord in obj["segmentation"]:
                    line += f" {coord:.6f}"
                f.write(line + "\n")

    def create_segmentation_visualization(self, seg_array):
        """Create colored visualization of segmentation"""
        # colors = {
        #     14: [255, 0, 0],      # Car - Red
        #     15: [255, 0, 0],      # Car - Red
        #     16: [0, 255, 0],       # Bus - Green
        #     12: [0, 0, 255],       # Pedestrian - Blue
        #     13: [0, 0, 255],       # Pedestrian - Blue
        #     7: [255, 255, 0]     # Traffic light - Yellow
        # }
        colors = {
            "car": [255, 0, 0],
            "bus": [0, 255, 0],
            "pedestrian": [0, 0, 255],
            "traffic_light": [255, 255, 0],
        }

        vis = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)

        # Color each instance
        instance_ids = seg_array[:, :, 0].astype(np.uint16) + (
            seg_array[:, :, 1].astype(np.uint16) * 256
        )
        class_ids = seg_array[:, :, 2]

        for instance_id in np.unique(instance_ids):
            if instance_id == 0:
                continue

            mask = instance_ids == instance_id
            class_id = class_ids[mask][0]

            if class_id in self.class_mapping:
                vis[mask] = colors[self.class_mapping[class_id]]

        return vis

    def collect_data(self, weather_condition, num_frames=1000, ego_speed=30):
        """Main data collection loop for a specific weather condition"""
        print(f"\n🌤️  Collecting data for {weather_condition} condition...")

        # Set weather
        self.world.set_weather(self.weather_presets[weather_condition])

        # Reset frame counter
        self.frame_id = 0

        # Statistics for this weather condition
        weather_stats = {
            "frames_collected": 0,
            "objects_per_class": defaultdict(int),
            "total_objects": 0,
            "visualizations_created": 0,
        }

        try:
            # Set ego vehicle to autopilot
            self.ego_vehicle.set_autopilot(True)

            # Warm up
            for _ in range(50):
                self.world.tick()

            # Clear queues
            while not self.rgb_queue.empty():
                self.rgb_queue.get()
            while not self.seg_queue.empty():
                self.seg_queue.get()

            # Main collection loop
            start_time = time.time()

            for frame in range(num_frames):
                # Tick the world
                self.world.tick()

                # Get synchronized frames
                rgb_image = None
                seg_image = None

                # Wait for both images with timeout
                try:
                    rgb_image = self.rgb_queue.get(timeout=1.0)
                    seg_image = self.seg_queue.get(timeout=1.0)
                except queue.Empty:
                    print(f"Timeout waiting for synchronized frames at frame {frame}")
                    continue

                # Process segmentation and create annotations
                annotations, seg_array = self.process_segmentation_image(
                    seg_image, weather_condition, rgb_image
                )

                # Save frame data (includes merged visualization every 10 frames)
                self.save_frame_data(
                    rgb_image, seg_array, annotations, weather_condition
                )

                # Update statistics
                weather_stats["frames_collected"] += 1
                weather_stats["total_objects"] += len(annotations["objects"])

                if self.frame_id % 10 == 0:
                    weather_stats["visualizations_created"] += 1

                for obj in annotations["objects"]:
                    weather_stats["objects_per_class"][obj["class"]] += 1

                self.frame_id += 1

                # Progress update
                if frame % 100 == 0:
                    elapsed = time.time() - start_time
                    fps = frame / elapsed if elapsed > 0 else 0
                    print(
                        f"   📊 Frame {frame}/{num_frames} - {fps:.2f} FPS - Objects: {len(annotations['objects'])} - Visualizations: {weather_stats['visualizations_created']}"
                    )

            # Store weather condition statistics
            self.dataset_stats["weather_conditions"][weather_condition] = weather_stats
            self.dataset_stats["total_frames"] += weather_stats["frames_collected"]

            print(f"✅ Completed {weather_condition} data collection:")
            print(f"   📁 Frames: {weather_stats['frames_collected']}")
            print(f"   🏷️  Total objects: {weather_stats['total_objects']}")
            print(
                f"   🎨 Visualizations created: {weather_stats['visualizations_created']}"
            )
            print(
                f"   📈 Objects per class: {dict(weather_stats['objects_per_class'])}"
            )

        except Exception as e:
            print(f"❌ Error during data collection: {e}")
            raise

    def cleanup_actors(self):
        """Clean up all spawned actors"""
        try:
            # Destroy cameras
            if hasattr(self, "rgb_camera"):
                self.rgb_camera.destroy()
            if hasattr(self, "seg_camera"):
                self.seg_camera.destroy()

            # Destroy ego vehicle
            if hasattr(self, "ego_vehicle"):
                self.ego_vehicle.destroy()

            # Destroy all other actors
            actors = self.world.get_actors()
            for actor in actors:
                if actor.type_id.startswith("vehicle.") or actor.type_id.startswith(
                    "walker."
                ):
                    actor.destroy()

            # Disable synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

        except Exception as e:
            print(f"⚠️  Error during cleanup: {e}")

    def run_full_data_collection(self, frames_per_condition=1000):
        """Run complete data collection for all weather conditions"""
        print("🚀 Starting CARLA Data Collection with Merged Visualizations...")
        print(f"🎯 Target frames per condition: {frames_per_condition}")
        print("🎨 Merged visualizations will be created every 10 frames")

        try:
            # Setup synchronous mode
            self.setup_synchronous_mode()

            # Spawn ego vehicle
            self.spawn_ego_vehicle()

            # Spawn traffic
            print("🚗 Spawning traffic...")
            vehicles, pedestrians = self.spawn_traffic(
                num_vehicles=100, num_pedestrians=60
            )
            print(
                f"✅ Spawned {len(vehicles)} vehicles and {len(pedestrians)} pedestrians"
            )

            # Collect data for each weather condition
            for weather_condition in self.weather_presets.keys():
                self.collect_data(weather_condition, frames_per_condition)
                time.sleep(2)  # Brief pause between conditions

            # Generate dataset analysis
            self.generate_dataset_analysis()

            total_visualizations = sum(
                [
                    stats.get("visualizations_created", 0)
                    for stats in self.dataset_stats["weather_conditions"].values()
                ]
            )

            print("\n🎉 DATA COLLECTION COMPLETED SUCCESSFULLY! 🎉")
            print(f"📊 Total frames collected: {self.dataset_stats['total_frames']}")
            print(
                f"🏷️  Total instances annotated: {self.dataset_stats['total_instances']}"
            )
            print(f"🎨 Total merged visualizations created: {total_visualizations}")

        finally:
            self.cleanup_actors()

    def generate_dataset_analysis(self):
        """Generate comprehensive dataset analysis and visualizations"""
        print("\nGenerating dataset analysis...")

        # Create analysis directory
        analysis_dir = os.path.join(self.output_dir, "analysis")

        # 1. Dataset Summary Statistics
        summary_stats = {
            "dataset_overview": {
                "total_frames": self.dataset_stats["total_frames"],
                "total_instances": self.dataset_stats["total_instances"],
                "weather_conditions": len(self.weather_presets),
                "object_classes": len(self.class_mapping),
            },
            "weather_distribution": {},
            "class_distribution": dict(self.dataset_stats["class_instances"]),
            "detailed_weather_stats": self.dataset_stats["weather_conditions"],
        }

        # Calculate weather distribution
        for weather, stats in self.dataset_stats["weather_conditions"].items():
            summary_stats["weather_distribution"][weather] = {
                "frames": stats["frames_collected"],
                "objects": stats["total_objects"],
                "objects_per_frame": (
                    stats["total_objects"] / stats["frames_collected"]
                    if stats["frames_collected"] > 0
                    else 0
                ),
            }

        # Save summary statistics
        with open(os.path.join(analysis_dir, "dataset_summary.json"), "w") as f:
            json.dump(summary_stats, f, indent=2)

        # 2. Generate Visualizations
        self._create_visualizations(analysis_dir, summary_stats)

        # 3. Generate detailed report
        self._generate_analysis_report(analysis_dir, summary_stats)

        print(f"Dataset analysis saved to: {analysis_dir}")

    def _create_visualizations(self, analysis_dir, summary_stats):
        """Create comprehensive visualizations for dataset analysis"""
        # plt.style.use('seaborn-v0_8')
        if "seaborn-v0_8" in plt.style.available:
            plt.style.use("seaborn-v0_8")
        else:
            plt.style.use("seaborn")

        # 1. Weather Condition Distribution
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Frames per weather condition
        weather_frames = [
            stats["frames"] for stats in summary_stats["weather_distribution"].values()
        ]
        weather_labels = list(summary_stats["weather_distribution"].keys())

        ax1.bar(
            weather_labels, weather_frames, color=["gold", "navy", "skyblue", "gray"]
        )
        ax1.set_title("Frames per Weather Condition", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Number of Frames")
        ax1.grid(axis="y", alpha=0.3)

        # Objects per weather condition
        weather_objects = [
            stats["objects"] for stats in summary_stats["weather_distribution"].values()
        ]
        ax2.bar(
            weather_labels, weather_objects, color=["gold", "navy", "skyblue", "gray"]
        )
        ax2.set_title("Objects per Weather Condition", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Number of Objects")
        ax2.grid(axis="y", alpha=0.3)

        # Class distribution (overall)
        classes = list(summary_stats["class_distribution"].keys())
        class_counts = list(summary_stats["class_distribution"].values())

        colors = ["red", "green", "blue", "orange"]
        ax3.pie(
            class_counts,
            labels=classes,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        ax3.set_title("Overall Class Distribution", fontsize=14, fontweight="bold")

        # Objects per frame by weather
        weather_obj_per_frame = [
            summary_stats["weather_distribution"][w]["objects_per_frame"]
            for w in weather_labels
        ]
        ax4.bar(
            weather_labels,
            weather_obj_per_frame,
            color=["gold", "navy", "skyblue", "gray"],
        )
        ax4.set_title(
            "Average Objects per Frame by Weather", fontsize=14, fontweight="bold"
        )
        ax4.set_ylabel("Objects per Frame")
        ax4.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(analysis_dir, "dataset_overview.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # 2. Detailed Class Distribution by Weather
        fig, ax = plt.subplots(figsize=(12, 8))

        # Prepare data for stacked bar chart
        weather_conditions = list(self.dataset_stats["weather_conditions"].keys())
        class_names = list(set(self.class_mapping.values()))

        # Create matrix of class counts per weather condition
        class_matrix = np.zeros((len(weather_conditions), len(class_names)))

        for i, weather in enumerate(weather_conditions):
            weather_data = self.dataset_stats["weather_conditions"][weather]
            for j, class_name in enumerate(class_names):
                class_matrix[i, j] = weather_data["objects_per_class"].get(
                    class_name, 0
                )

        # Create stacked bar chart
        bottom = np.zeros(len(weather_conditions))
        colors = ["red", "green", "blue", "orange"]

        for j, class_name in enumerate(class_names):
            ax.bar(
                weather_conditions,
                class_matrix[:, j],
                bottom=bottom,
                label=class_name,
                color=colors[j],
                alpha=0.8,
            )
            bottom += class_matrix[:, j]

        ax.set_title(
            "Class Distribution by Weather Condition", fontsize=16, fontweight="bold"
        )
        ax.set_ylabel("Number of Instances")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(analysis_dir, "class_distribution_by_weather.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # 3. Dataset Quality Metrics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Class balance visualization
        total_instances = sum(summary_stats["class_distribution"].values())
        class_percentages = [
            count / total_instances * 100
            for count in summary_stats["class_distribution"].values()
        ]

        ax1.bar(classes, class_percentages, color=colors)
        ax1.set_title("Class Balance (%)", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Percentage of Total Instances")
        ax1.grid(axis="y", alpha=0.3)

        # Add percentage labels on bars
        for i, v in enumerate(class_percentages):
            ax1.text(
                i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold"
            )

        # Weather condition balance
        total_frames = sum(weather_frames)
        weather_percentages = [frames / total_frames * 100 for frames in weather_frames]

        ax2.bar(
            weather_labels,
            weather_percentages,
            color=["gold", "navy", "skyblue", "gray"],
        )
        ax2.set_title("Weather Condition Balance (%)", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Percentage of Total Frames")
        ax2.grid(axis="y", alpha=0.3)

        # Add percentage labels on bars
        for i, v in enumerate(weather_percentages):
            ax2.text(
                i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold"
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(analysis_dir, "dataset_balance.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _generate_analysis_report(self, analysis_dir, summary_stats):
        """Generate a detailed analysis report in Markdown format."""
        # Use pathlib for modern, object-oriented path handling
        report_path = Path(analysis_dir) / "dataset_analysis_report.md"

        with open(report_path, "w", encoding="UTF-8") as f:
            f.write("# CARLA Dataset Analysis Report\n\n")
            f.write(
                f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            # --- Section: Dataset Overview ---
            f.write("## Dataset Overview\n\n")
            overview = summary_stats.get("dataset_overview", {})
            total_frames = overview.get("total_frames", 0)
            total_instances = overview.get("total_instances", 0)

            # Safely calculate average instances per frame
            avg_instances_per_frame = (
                (total_instances / total_frames) if total_frames > 0 else 0.0
            )

            f.write(f"- **Total Frames:** {total_frames:,}\n")
            f.write(f"- **Total Annotated Instances:** {total_instances:,}\n")
            f.write(
                f"- **Weather Conditions:** {overview.get('weather_conditions', 'N/A')}\n"
            )
            f.write(f"- **Object Classes:** {overview.get('object_classes', 'N/A')}\n")
            f.write(
                f"- **Average Instances per Frame:** {avg_instances_per_frame:.2f}\n\n"
            )

            # --- Section: Weather Condition Distribution ---
            f.write("## Weather Condition Distribution\n\n")
            f.write("| Weather | Frames | Objects | Objects/Frame |\n")
            f.write("|---------|--------|---------|---------------|\n")

            for weather, stats in summary_stats.get("weather_distribution", {}).items():
                frames = stats.get("frames", 0)
                objects = stats.get("objects", 0)
                # Safely calculate objects per frame for each weather condition
                obj_per_frame = (objects / frames) if frames > 0 else 0.0
                f.write(
                    f"| {weather.capitalize()} | {frames:,} | {objects:,} | {obj_per_frame:.2f} |\n"
                )

            # --- Section: Class Distribution ---
            f.write("\n## Class Distribution\n\n")
            f.write("| Class | Instances | Percentage |\n")
            f.write("|-------|-----------|------------|\n")

            class_dist = summary_stats.get("class_distribution", {})
            total_class_instances = sum(class_dist.values())

            for class_name, count in class_dist.items():
                # Safely calculate percentage
                percentage = (
                    (count / total_class_instances * 100)
                    if total_class_instances > 0
                    else 0.0
                )
                formatted_name = class_name.replace("_", " ").title()
                f.write(f"| {formatted_name} | {count:,} | {percentage:.1f}% |\n")

            # --- Section: Detailed Weather Analysis ---
            f.write("\n## Detailed Weather Analysis\n\n")
            for weather, detailed_stats in summary_stats.get(
                "detailed_weather_stats", {}
            ).items():
                frames_collected = detailed_stats.get("frames_collected", 0)
                total_objects = detailed_stats.get("total_objects", 0)
                avg_obj_per_frame = (
                    (total_objects / frames_collected) if frames_collected > 0 else 0.0
                )

                f.write(f"### {weather.capitalize()} Conditions\n")
                f.write(f"- **Frames Collected:** {frames_collected:,}\n")
                f.write(f"- **Total Objects:** {total_objects:,}\n")
                f.write(f"- **Average Objects per Frame:** {avg_obj_per_frame:.2f}\n")
                f.write("- **Class Distribution:**\n")

                for class_name, count in detailed_stats.get(
                    "objects_per_class", {}
                ).items():
                    f.write(f"  - {class_name.replace('_', ' ').title()}: {count:,}\n")
                f.write("\n")

            # --- Section: Dataset Quality Assessment ---
            f.write("## Dataset Quality Assessment\n\n")

            # Class balance assessment
            f.write("### Class Balance\n")
            class_counts = list(class_dist.values())
            if class_counts:  # Ensure list is not empty
                min_class_count = min(class_counts)
                max_class_count = max(class_counts)
                balance_ratio = (
                    (min_class_count / max_class_count) if max_class_count > 0 else 0.0
                )

                f.write(
                    f"- **Balance Ratio:** {balance_ratio:.3f} (1.0 = perfect balance)\n"
                )
                f.write(f"- **Min Class Count:** {min_class_count:,}\n")
                f.write(f"- **Max Class Count:** {max_class_count:,}\n")

                if balance_ratio > 0.7:
                    f.write("- **Assessment:** Good class balance ✓\n")
                elif balance_ratio > 0.4:
                    f.write("- **Assessment:** Moderate class imbalance ⚠️\n")
                else:
                    f.write("- **Assessment:** Significant class imbalance ❌\n")
            else:
                f.write("- **Assessment:** No class data to assess.\n")

            # Weather balance assessment
            f.write("\n### Weather Condition Balance\n")
            weather_counts = [
                stats.get("frames", 0)
                for stats in summary_stats.get("weather_distribution", {}).values()
            ]
            if weather_counts:  # Ensure list is not empty
                min_weather_frames = min(weather_counts)
                max_weather_frames = max(weather_counts)
                weather_balance = (
                    (min_weather_frames / max_weather_frames)
                    if max_weather_frames > 0
                    else 0.0
                )

                f.write(f"- **Balance Ratio:** {weather_balance:.3f}\n")
                f.write(f"- **Min Weather Frames:** {min_weather_frames:,}\n")
                f.write(f"- **Max Weather Frames:** {max_weather_frames:,}\n")

                if weather_balance > 0.8:
                    f.write("- **Assessment:** Excellent weather balance ✓\n")
                else:
                    f.write("- **Assessment:** Good weather distribution ✓\n")
            else:
                f.write("- **Assessment:** No weather data to assess.\n")

            # Sample sufficiency assessment
            f.write("\n### Sample Sufficiency for Deep Learning\n")
            f.write(f"- **Total Training Samples:** {total_frames:,}\n")

            if total_frames > 8000:
                f.write("- **Assessment:** Excellent sample size for deep learning ✓\n")
            elif total_frames > 4000:
                f.write("- **Assessment:** Good sample size for deep learning ✓\n")
            elif total_frames > 2000:
                f.write(
                    "- **Assessment:** Moderate sample size, augmentation recommended ⚠️\n"
                )
            else:
                f.write(
                    "- **Assessment:** Small sample size, extensive augmentation required ❌\n"
                )

            # --- Section: Recommendations ---
            f.write("\n## Recommendations\n\n")
            f.write("### Data Augmentation Strategy\n")

            # This recommendation is now also safer
            if "balance_ratio" in locals() and balance_ratio < 0.5:
                f.write("- Apply **class-weighted sampling** during training.\n")
                f.write(
                    "- Use **targeted augmentation** for underrepresented classes.\n"
                )

            f.write(
                "- **General Augmentations:** Rotation, horizontal flipping, brightness/contrast adjustments, and noise addition are recommended.\n"
            )

            f.write("\n### Training Strategy\n")
            f.write(
                "- **Dataset Split:** A 60/20/20 split (Train/Val/Test), stratified by weather, is a solid baseline.\n"
            )
            f.write(
                "- **Evaluation:** Test the final model on each weather condition separately to identify performance gaps.\n"
            )

            f.write("\n---\n")
            f.write(
                "*This analysis provides the foundation for YOLOv11 model training and evaluation.*\n"
            )


# Dataset Analysis and Visualization Script
class DatasetAnalyzer:
    """Separate class for comprehensive dataset analysis"""

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.analysis_data = {}

    def load_annotations(self):
        """Load all annotations from dataset"""
        annotations = {}

        for weather in ["day", "night", "rain", "fog"]:
            weather_path = os.path.join(self.dataset_path, weather, "annotations")
            annotations[weather] = []

            if os.path.exists(weather_path):
                for ann_file in os.listdir(weather_path):
                    if ann_file.endswith(".json"):
                        with open(os.path.join(weather_path, ann_file), "r") as f:
                            ann_data = json.load(f)
                            annotations[weather].append(ann_data)

        return annotations

    def analyze_dataset_statistics(self):
        """Perform comprehensive statistical analysis"""
        annotations = self.load_annotations()

        stats = {
            "weather_stats": {},
            "class_stats": defaultdict(int),
            "size_distribution": [],
            "aspect_ratios": [],
            "object_density": [],
        }

        for weather, weather_annotations in annotations.items():
            weather_data = {
                "frame_count": len(weather_annotations),
                "total_objects": 0,
                "class_distribution": defaultdict(int),
                "avg_objects_per_frame": 0,
                "object_sizes": [],
                "bbox_areas": [],
            }

            for ann in weather_annotations:
                objects = ann.get("objects", [])
                weather_data["total_objects"] += len(objects)

                for obj in objects:
                    class_name = obj["class"]
                    weather_data["class_distribution"][class_name] += 1
                    stats["class_stats"][class_name] += 1

                    # Analyze bounding box properties
                    bbox = obj["bbox_normalized"]
                    width, height = bbox[2], bbox[3]
                    area = width * height
                    aspect_ratio = width / height if height > 0 else 1

                    weather_data["bbox_areas"].append(area)
                    weather_data["object_sizes"].append((width, height))
                    stats["size_distribution"].append(area)
                    stats["aspect_ratios"].append(aspect_ratio)

            if weather_data["frame_count"] > 0:
                weather_data["avg_objects_per_frame"] = (
                    weather_data["total_objects"] / weather_data["frame_count"]
                )

            stats["weather_stats"][weather] = weather_data
            stats["object_density"].extend(
                [len(ann.get("objects", [])) for ann in weather_annotations]
            )

        return stats

    def create_advanced_visualizations(self, stats, output_dir):
        """Create advanced visualizations for dataset analysis"""

        # 1. Object Size Distribution Analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Size distribution histogram
        sizes = np.array(stats["size_distribution"])
        ax1.hist(sizes, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
        ax1.set_xlabel("Normalized Bounding Box Area")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Object Size Distribution", fontweight="bold")
        ax1.grid(alpha=0.3)

        # Aspect ratio distribution
        ax2.hist(
            stats["aspect_ratios"],
            bins=50,
            alpha=0.7,
            color="lightcoral",
            edgecolor="black",
        )
        ax2.set_xlabel("Aspect Ratio (Width/Height)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Object Aspect Ratio Distribution", fontweight="bold")
        ax2.grid(alpha=0.3)

        # Objects per frame distribution
        ax3.hist(
            stats["object_density"],
            bins=30,
            alpha=0.7,
            color="lightgreen",
            edgecolor="black",
        )
        ax3.set_xlabel("Objects per Frame")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Object Density Distribution", fontweight="bold")
        ax3.grid(alpha=0.3)

        # Class frequency comparison
        classes = list(stats["class_stats"].keys())
        counts = list(stats["class_stats"].values())
        colors = ["red", "green", "blue", "orange"][: len(classes)]

        bars = ax4.bar(classes, counts, color=colors, alpha=0.8, edgecolor="black")
        ax4.set_ylabel("Total Instances")
        ax4.set_title("Class Frequency Distribution", fontweight="bold")
        ax4.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(counts) * 0.01,
                f"{count:,}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "advanced_dataset_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # 2. Weather-specific Analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        weather_conditions = list(stats["weather_stats"].keys())

        for i, (weather, ax) in enumerate(zip(weather_conditions, axes.flat)):
            weather_data = stats["weather_stats"][weather]
            classes = list(weather_data["class_distribution"].keys())
            counts = list(weather_data["class_distribution"].values())

            if counts:  # Only plot if there's data
                colors = ["red", "green", "blue", "orange"][: len(classes)]
                bars = ax.bar(classes, counts, color=colors, alpha=0.8)
                ax.set_title(
                    f"{weather.capitalize()} - Class Distribution", fontweight="bold"
                )
                ax.set_ylabel("Instances")
                ax.grid(axis="y", alpha=0.3)

                # Add value labels
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + max(counts) * 0.01,
                        f"{count}",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "weather_specific_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # 3. Correlation Analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Weather vs Average Objects per Frame
        weather_names = []
        avg_objects = []
        for weather, data in stats["weather_stats"].items():
            weather_names.append(weather.capitalize())
            avg_objects.append(data["avg_objects_per_frame"])

        bars = ax1.bar(
            weather_names,
            avg_objects,
            color=["gold", "navy", "skyblue", "gray"],
            alpha=0.8,
        )
        ax1.set_ylabel("Average Objects per Frame")
        ax1.set_title("Object Density by Weather Condition", fontweight="bold")
        ax1.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar, avg in zip(bars, avg_objects):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(avg_objects) * 0.01,
                f"{avg:.1f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Size vs Weather (box plot style visualization)
        weather_sizes = {}
        for weather, data in stats["weather_stats"].items():
            weather_sizes[weather.capitalize()] = data["bbox_areas"]

        if weather_sizes:
            ax2.boxplot(weather_sizes.values(), labels=weather_sizes.keys())
            ax2.set_ylabel("Normalized Bounding Box Area")
            ax2.set_title("Object Size Distribution by Weather", fontweight="bold")
            ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "correlation_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        return stats


# Main execution script
def main():
    """Main function to run complete data collection and analysis"""

    # Configuration
    FRAMES_PER_CONDITION = 1000  # Adjust based on your needs
    HOST = "localhost"
    PORT = 2000

    try:
        print("=" * 60)
        print("CARLA YOLOv11 Dataset Collection & Analysis")
        print("=" * 60)

        # Initialize data collector
        collector = CarlaDataCollector(host=HOST, port=PORT)

        # Run full data collection
        collector.run_full_data_collection(frames_per_condition=FRAMES_PER_CONDITION)

        print("\n" + "=" * 60)
        print("Running Advanced Dataset Analysis...")
        print("=" * 60)

        # Run advanced analysis
        analyzer = DatasetAnalyzer(collector.output_dir)
        stats = analyzer.analyze_dataset_statistics()

        analysis_dir = os.path.join(collector.output_dir, "analysis")
        analyzer.create_advanced_visualizations(stats, analysis_dir)

        # Generate comprehensive summary
        total_frames = sum(
            [data["frame_count"] for data in stats["weather_stats"].values()]
        )
        total_objects = sum(stats["class_stats"].values())

        print(f"\n🎉 DATASET COLLECTION COMPLETED! 🎉")
        print(f"📊 Total Frames: {total_frames:,}")
        print(f"🏷️  Total Annotated Objects: {total_objects:,}")
        print(f"🌤️  Weather Conditions: {len(stats['weather_stats'])}")
        print(f"📱 Object Classes: {len(stats['class_stats'])}")
        print(f"📁 Dataset saved to: {collector.output_dir}")
        print(f"📈 Analysis saved to: {analysis_dir}")

        print("\n📋 Next Steps:")
        print("1. Review the generated analysis report")
        print("2. Implement data augmentation strategies")
        print("3. Convert annotations to YOLOv11 format")
        print("4. Split dataset for training/validation/testing")
        print("5. Begin YOLOv11 model training")

    except KeyboardInterrupt:
        print("\n⚠️  Data collection interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        raise
    finally:
        print("\n🧹 Cleaning up CARLA environment...")


if __name__ == "__main__":
    main()

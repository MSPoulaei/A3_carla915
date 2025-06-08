pip install tensorrt pycuda 
trtexec --onnx=carla_yolo_training_result/carla_segmentation_nano/weights/best.onnx \
        --saveEngine=carla_yolo_training_result/carla_segmentation_nano/weights/best.engine \
        --minShapes=images:1x3x512x512 \
        --optShapes=images:1x3x512x512 \
        --maxShapes=images:1x3x512x512 \
        --fp16
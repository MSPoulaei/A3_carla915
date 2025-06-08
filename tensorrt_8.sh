pip install tensorrt pycuda 
trtexec --onnx=path/to/your/model.onnx \
        --saveEngine=path/to/your/model.engine \
        --minShapes=images:1x3x512x512 \
        --optShapes=images:1x3x512x512 \
        --maxShapes=images:1x3x512x512 \
        --fp16
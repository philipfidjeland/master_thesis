import os
import cv2
import numpy as np
from pynq_dpu import DpuOverlay
from pathlib import Path
from time import time
import pynq
import matplotlib.pyplot as plt
from pynq import DataRecorder
# Configuration
model_path = "SimpleSSIM.xmodel"  # Replace with your xmodel path
input_folder = "test"
output_folder = "test2"
IMG_SIZE = (512, 512, 3)  # Image dimensions with channel last

# Create output directory
Path(output_folder).mkdir(parents=True, exist_ok=True)

# Load DPU Overlay (required for DPU functionality)
overlay = DpuOverlay("dpu.bit")  # Replace with your DPU bitstream file
overlay.load_model(model_path)


rails = pynq.get_rails()
print(rails)
dpu = overlay.runner  # Get the DPU runner
recorder = pynq.DataRecorder(rails['INT'].power) #Record power consumption

# Get input/output tensor information
input_tensors = dpu.get_input_tensors()
output_tensors = dpu.get_output_tensors()

# Print tensor shapes for debugging
print("Input Tensor Shape:", input_tensors[0].dims)
print("Output Tensor Shape:", output_tensors[0].dims)

# Ensure the output shape matches (1, 512, 512, 3)
shape_output = (1, 512, 512, 3)
output = [np.empty(shape_output, dtype=np.float32, order="C")]
total_img=0
total_time=0

# Process images
recorder.reset()
with recorder.record(0.2):
    for img_name in os.listdir(input_folder):
        if not img_name.lower().endswith(('.jpg', '.jpeg')):
            continue
        total_img+=1
        # Load and preprocess image
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, IMG_SIZE[:2])  # Resize to (512, 512)
    
        # Convert BGR to RGB (if your model expects RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        # Normalize to [0,1] and prepare input tensor
    
        normalized_data = np.asarray(img/255, dtype=np.float32)
        input_data = normalized_data.reshape(1, *IMG_SIZE)  # Add batch dimension
        start = time()
        # Run inference
        job_id = dpu.execute_async(input_data, output)
        dpu.wait(job_id)
        stop = time()
        total_time+=(stop-start)
        # Extract output and reshape
        output_img = output[0].reshape(512, 512, 3)  # Remove batch dimension
    
        # Denormalize and convert to uint8
        output_img = (output_img * 255).clip(0, 255).astype(np.float32)
    
        # Convert RGB to BGR if needed
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        # Save result

    
        output_path = os.path.join(output_folder, f"processed_{img_name}")
        cv2.imwrite(output_path, output_img)
        
print(recorder.frame)

recorder.frame['INT_power'].plot()
plt.savefig('power.png', dpi=150, bbox_inches='tight')

    

print("Throughput: {:.4f}FPS".format(total_img/total_time))

print("Processing completed!")

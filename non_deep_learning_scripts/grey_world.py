import os
import cv2
import numpy as np
import natsort
import datetime

def grey_world(image):
    # Convert image to float32 for calculations
    image = image.astype(np.float32)
    
    # Split color channels
    B, G, R = cv2.split(image)
    
    # Calculate average intensities for each channel
    avg_B = np.mean(B)
    avg_G = np.mean(G)
    avg_R = np.mean(R)
    
    # Calculate average gray value
    avg_gray = (avg_B + avg_G + avg_R) / 3.0
    
    # Avoid division by zero
    if avg_B == 0: avg_B = 1e-6
    if avg_G == 0: avg_G = 1e-6
    if avg_R == 0: avg_R = 1e-6
    
    # Compute scaling factors
    scale_B = avg_gray / avg_B
    scale_G = avg_gray / avg_G
    scale_R = avg_gray / avg_R
    
    # Apply scaling to each channel
    B = np.clip(B * scale_B, 0, 255)
    G = np.clip(G * scale_G, 0, 255)
    R = np.clip(R * scale_R, 0, 255)
    
    # Merge channels back and convert to uint8
    balanced = cv2.merge([B, G, R]).astype(np.uint8)
    return balanced

if __name__ == '__main__':
    starttime = datetime.datetime.now()
    
    # Configure paths
    input_folder = "MasterNetverk/LSUI/test"
    output_folder = "OutputImagesGreyWorld"
    
    # Create output directory if needed
    os.makedirs(output_folder, exist_ok=True)
    
    # Process images
    files = natsort.natsorted(os.listdir(input_folder))
    
    for file in files:
        file_path = os.path.join(input_folder, file)
        if os.path.isfile(file_path):
            print(f'Processing: {file}')
            img = cv2.imread(file_path)
            if img is not None:
                result = grey_world(img)
                output_path = os.path.join(output_folder, 
                                         f"{os.path.splitext(file)[0]}_GreyWorld.jpg")
                cv2.imwrite(output_path, result)
    
    # Calculate and print execution time
    endtime = datetime.datetime.now()
    print(f'Total processing time: {endtime - starttime}')
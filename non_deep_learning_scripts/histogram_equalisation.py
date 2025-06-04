import os
import cv2
import numpy as np
import natsort
import datetime
from skimage import exposure, color

def sk_hist_equalization(img):
    # Convert to RGB and float [0,1]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    
    # Convert to HSV and equalize V channel
    hsv = color.rgb2hsv(img_float)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    
    # Convert back to RGB
    equalized = color.hsv2rgb(hsv) * 255
    return cv2.cvtColor(equalized.astype(np.uint8), cv2.COLOR_RGB2BGR)

def process_image(input_path, output_path):
    img = cv2.imread(input_path)
    if img is not None:
        processed = sk_hist_equalization(img)
        cv2.imwrite(output_path, processed)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    
    input_folder = "MasterNetverk/LSUI/test"
    output_folder = "OutputHistEqualization"
    os.makedirs(output_folder, exist_ok=True)

    files = natsort.natsorted(os.listdir(input_folder))
    
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, 
                                     f"{os.path.splitext(file)[0]}_HE.jpg")
            process_image(input_path, output_path)
    
    print(f"Processing time: {datetime.datetime.now() - start_time}")
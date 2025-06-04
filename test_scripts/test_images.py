import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers
import os
import numpy as np
import matplotlib.pyplot as plt

import time
import numpy as np
from pynvml import *
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from skimage.metrics import structural_similarity
from skimage import io, color, filters
import math
from scipy import ndimage
import cv2 

def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    """
      Calculates the asymetric alpha-trimmed mean
    """
    # sort pixels by intensity - for clipping
    x = sorted(x)
    # get number of pixels
    K = len(x)
    # calculate T alpha L and T alpha R
    T_a_L = math.ceil(alpha_L*K)
    T_a_R = math.floor(alpha_R*K)
    # calculate mu_alpha weight
    weight = (1/(K-T_a_L-T_a_R))
    # loop through flattened image starting at T_a_L+1 and ending at K-T_a_R
    s   = int(T_a_L+1)
    e   = int(K-T_a_R)
    val = sum(x[s:e])
    val = weight*val
    return val


def s_a(x, mu):
    val = 0
    for pixel in x:
        val += math.pow((pixel-mu), 2)
    return val/len(x)


def _uicm(x):
    R = x[:,:,0].flatten()
    G = x[:,:,1].flatten()
    B = x[:,:,2].flatten()
    RG = R-G
    YB = ((R+G)/2)-B
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    l = math.sqrt( (math.pow(mu_a_RG,2)+math.pow(mu_a_YB,2)) )
    r = math.sqrt(s_a_RG+s_a_YB)
    return (-0.0268*l)+(0.1586*r)


def sobel(x):
    dx = ndimage.sobel(x,0)
    dy = ndimage.sobel(x,1)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag) 
    return mag


def eme(x, window_size):
    """
      Enhancement measure estimation
      x.shape[0] = height
      x.shape[1] = width
    """
    # if 4 blocks, then 2x2...etc.
    k1 = int(x.shape[1]/window_size)
    k2 = int(x.shape[0]/window_size)

    # weight
    w = 2./(k1*k2)

    blocksize_x = window_size
    blocksize_y = window_size

    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[:blocksize_y*int(k2), :blocksize_x*int(k1)]

    val = 0
    for l in range(k1):
        for k in range(k2):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1)]
            max_ = np.max(block)
            min_ = np.min(block)

            # bound checks, can't do log(0)
            if min_ == 0.0: val += 0
            elif max_ == 0.0: val += 0
            else: val += math.log(max_/min_)
    return w*val


def _uism(x):
    """
      Underwater Image Sharpness Measure
    """
    # get image channels
    R = x[:,:,0]
    G = x[:,:,1]
    B = x[:,:,2]

    # first apply Sobel edge detector to each RGB component
    Rs = sobel(R)
    Gs = sobel(G)
    Bs = sobel(B)

    # multiply the edges detected for each channel by the channel itself
    R_edge_map = np.multiply(Rs, R)
    G_edge_map = np.multiply(Gs, G)
    B_edge_map = np.multiply(Bs, B)

    # get eme for each channel
    r_eme = eme(R_edge_map, 10)
    g_eme = eme(G_edge_map, 10)
    b_eme = eme(B_edge_map, 10)

    # coefficients
    lambda_r = 0.299
    lambda_g = 0.587
    lambda_b = 0.144

    return (lambda_r*r_eme) + (lambda_g*g_eme) + (lambda_b*b_eme)


def plip_g(x,mu=1026.0):
    return mu-x


def plip_theta(g1, g2, k):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return k*((g1-g2)/(k-g2))


def plip_cross(g1, g2, gamma):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return g1+g2-((g1*g2)/(gamma))


def plip_diag(c, g, gamma):
    g = plip_g(g)
    return gamma - (gamma * math.pow((1 - (g/gamma) ), c) )


def plip_multiplication(g1, g2):
    return plip_phiInverse(plip_phi(g1) * plip_phi(g2))
    #return plip_phiInverse(plip_phi(plip_g(g1)) * plip_phi(plip_g(g2)))


def plip_phiInverse(g):
    plip_lambda = 1026.0
    plip_beta   = 1.0
    return plip_lambda * (1 - math.pow(math.exp(-g / plip_lambda), 1 / plip_beta));


def plip_phi(g):
    plip_lambda = 1026.0
    plip_beta   = 1.0
    return -plip_lambda * math.pow(math.log(1 - g / plip_lambda), plip_beta)


def _uiconm(x, window_size):
    """
      Underwater image contrast measure
      https://github.com/tkrahn108/UIQM/blob/master/src/uiconm.cpp
      https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5609219
    """

    plip_lambda = 1026.0
    plip_gamma  = 1026.0
    plip_beta   = 1.0
    plip_mu     = 1026.0
    plip_k      = 1026.0

    # if 4 blocks, then 2x2...etc.
    k1 = int(x.shape[1]/window_size)
    k2 = int(x.shape[0]/window_size)

    # weight
    w = -1./(k1*k2)

    blocksize_x = window_size
    blocksize_y = window_size

    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[:blocksize_y*k2, :blocksize_x*k1]

    # entropy scale - higher helps with randomness
    alpha = 1

    val = 0
    for l in range(k1):
        for k in range(k2):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1), :]
            max_ = np.max(block)
            min_ = np.min(block)

            top = max_-min_
            bot = max_+min_

            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0: val += 0.0
            else: val += alpha*math.pow((top/bot),alpha) * math.log(top/bot)

            #try: val += plip_multiplication((top/bot),math.log(top/bot))
    return w*val


def getUIQM(x):
    """
      Function to return UIQM to be called from other programs
      x: image
    """
    x = x.astype(np.float32)
    ### from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7300447
    #c1 = 0.4680; c2 = 0.2745; c3 = 0.2576
    ### from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7300447
    c1 = 0.0282; c2 = 0.2953; c3 = 3.5753

    uicm   = _uicm(x)
    uism   = _uism(x)
    uiconm = _uiconm(x, 10)
    uiqm = (c1*uicm) + (c2*uism) + (c3*uiconm)
    return uiqm

def getUCIQE(img):
    img_BGR = cv2.cvtColor(img*255, cv2.COLOR_RGB2BGR)
    img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB) 
    img_LAB = np.array(img_LAB,dtype=np.float64)
    # Trained coefficients are c1=0.4680, c2=0.2745, c3=0.2576 according to paper.
    coe_Metric = [0.4680, 0.2745, 0.2576]
    img_lum = img_LAB[:,:,0]/100
    img_a = (img_LAB[:,:,1]+128)/255
    img_b = (img_LAB[:,:,2]+128)/255

    # item-1
    chroma = np.sqrt(np.square(img_a)+np.square(img_b))
    sigma_c = np.std(chroma)

    # item-2
    img_lum = img_lum.flatten()
    sorted_index = np.argsort(img_lum)
    top_index = sorted_index[int(len(img_lum)*0.99)]
    bottom_index = sorted_index[int(len(img_lum)*0.01)]
    con_lum = img_lum[top_index] - img_lum[bottom_index]

    # item-3
    chroma = chroma.flatten()
    sat = np.divide(chroma, img_lum, out=np.zeros_like(chroma, dtype=np.float64), where=img_lum!=0)
    avg_sat = np.mean(sat)

    uciqe = sigma_c*coe_Metric[0] + con_lum*coe_Metric[1] + avg_sat*coe_Metric[2]
    return uciqe


def calculate_psnr(ground_truth, prediction):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between the ground truth and prediction.
    
    Args:
        ground_truth: Ground truth image (NumPy array or Tensor) with values in range [0, 1].
        prediction: Predicted image (NumPy array or Tensor) with values in range [0, 1].
    
    Returns:
        PSNR value in decibels (dB).
    """
    # Ensure inputs are NumPy arrays
    ground_truth = np.array(ground_truth)
    prediction = np.array(prediction)
    # Mean Squared Error (MSE)
    mse = np.mean((ground_truth - prediction) ** 2)
    
    # Avoid division by zero
    if mse == 0:
        return float('inf')  # Perfect prediction
    
    # PSNR calculation
    max_pixel_value = 1.0  # Pixel values are normalized to [0, 1]
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return tf.image.psnr(ground_truth, prediction, max_val=max_pixel_value)

def evaluate_image_quality_dataset(prediction_dir, ground_truth_dir, num_images=None):
    """
    Evaluate image quality metrics using precomputed images from specified directories.
    
    Args:
        prediction_dir: Directory containing predicted images.
        ground_truth_dir: Directory containing ground truth images.
        num_images: Number of images to evaluate. If None, uses all available images.
    
    Returns:
        Lists of PSNR, SSIM, UIQM, and UCIQE values for each image.
    """
    psnr_values, ssim_values, uiqm_values, uciqe_values = [], [], [], []
    prediction_files = sorted(os.listdir(prediction_dir))
    ground_truth_files = sorted(os.listdir(ground_truth_dir))
    
    # Determine the number of images to process
    max_images = min(len(prediction_files), len(ground_truth_files))
    if num_images is None:
        num_images = max_images
    else:
        num_images = min(num_images, max_images)
    
    for i in range(num_images):
        # Load prediction and ground truth images
        pred_path = os.path.join(prediction_dir, prediction_files[i])
        gt_path = os.path.join(ground_truth_dir, ground_truth_files[i])
        
        pred_img = load_and_preprocess_image(pred_path).numpy()
        gt_img = load_and_preprocess_image(gt_path).numpy()
        
        # Compute metrics
        psnr = calculate_psnr(gt_img, pred_img)
        ssim = structural_similarity(gt_img, pred_img, channel_axis=2, data_range=gt_img.max() - gt_img.min())
        uiqm = getUIQM(pred_img)
        uciqe = getUCIQE(pred_img)
        
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        uiqm_values.append(uiqm)
        uciqe_values.append(uciqe)
    
    return psnr_values, ssim_values, uiqm_values, uciqe_values


def load_and_preprocess_image(file_path):
    """
    Load and preprocess an image from a file path.
    """
    image = tf.io.read_file(file_path)  # Read the file
    image = tf.image.decode_jpeg(image, channels=3)  # Decode the JPEG
    image = tf.image.resize(image, [512, 512])  # Resize to model's input size
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image
def visualize_and_save_predictions(model, dataset, save_dir="predictions256", num_images=1048):
    """
    Visualizes predictions on the provided dataset and saves the images.
    
    Args:
        model: The model used for making predictions.
        dataset: The dataset containing input images.
        save_dir: Directory to save the visualized predictions.
        num_images: Number of images to visualize and save.
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Take one batch from the dataset
    for images in dataset.take(1):
        # Convert TensorFlow tensors to NumPy arrays
        images_np = images.numpy()
        predictions = model.predict(images_np)

        # Ensure pixel values are scaled between [0, 1] for saving
        images_np = (images_np - images_np.min()) / (images_np.max() - images_np.min())
        predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())

        # Plot input images and predictions
        plt.figure(figsize=(15, num_images * 5))
        for i in range(num_images):
            #Plot input image
            plt.subplot(num_images, 2, i * 2 + 1)
            plt.imshow(images_np[i])
            plt.title("Input Image")

            # Save input image
            input_image_path = os.path.join(save_dir, f"input_image_{i}.png")
            plt.imsave(input_image_path, images_np[i])
            
            # Plot predicted image
            plt.subplot(num_images, 2, i * 2 + 2)
            plt.imshow(predictions[i])
            plt.title("Predicted Image")
            plt.axis("off")

            # Save predicted image
            predicted_image_path = os.path.join(save_dir, f"predicted_image_{i}.png")
            plt.imsave(predicted_image_path, predictions[i])

        # Save the full visualization as a single figure
        combined_path = os.path.join(save_dir, "combined_visualization.png")
        plt.savefig(combined_path)
        plt.show()

    print(f"Images and visualizations saved in '{save_dir}'.")

# Visualize on a custom directory of images
vis_dir = 'MasterNetverk/LSUI/test'  # Directory containing validation images
vis_images = [os.path.join(vis_dir, fname) for fname in sorted(os.listdir(vis_dir))]

ground_truth_dir = 'MasterNetverk/LSUI/GT_test'
prediction_dir= 'MasterNetverk/FPGA_predictions/OutputImagesGreyWorld'
# Prepare the dataset for visualization
vis_dataset = tf.data.Dataset.from_tensor_slices(vis_images) \
    .map(load_and_preprocess_image) \
    .batch(8)  # Batch size can be adjusted


psnr_values, ssim_values, uiqm_values, uciqe_values = evaluate_image_quality_dataset(prediction_dir, ground_truth_dir)

# Print average and variance for PSNR
average_psnr = np.mean(psnr_values)
variance_psnr = np.var(psnr_values)
print (max(psnr_values))
print (min(psnr_values))
print(f"\nAverage PSNR: {average_psnr:.2f} dB")
print(f"Variance of PSNR: {variance_psnr:.2f}")
# Print average and variance for SSIM
average_ssim = np.mean(ssim_values)
variance_ssim = np.var(ssim_values)
print(f"Average SSIM: {average_ssim:.4f}")
print(f"Variance of SSIM: {variance_ssim:.4f}")

# Print average and variance for UIQM
average_uiqm = np.mean(uiqm_values)
variance_uiqm = np.var(uiqm_values)
print(f"Average UIQM: {average_uiqm:.4f}")
print(f"Variance of UIQM: {variance_uiqm:.4f}")

# Print average and variance for UCIQE
average_uciqe = np.mean(uciqe_values)
variance_uciqe = np.var(uciqe_values)
print(f"Average UCIQE: {average_uciqe:.4f}")
print(f"Variance of UCIQE: {variance_uciqe:.4f}\n")


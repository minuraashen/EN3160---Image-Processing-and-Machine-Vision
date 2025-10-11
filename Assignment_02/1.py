"""
Question 1: Blob Detection using Laplacian of Gaussian (LoG)
Detects circular blobs (sunflowers) using scale-space extrema detection
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def laplacian_of_gaussian(img, sigma):
    """
    Compute Laplacian of Gaussian for blob detection
    
    Args:
        img: Grayscale input image
        sigma: Standard deviation of Gaussian kernel
    
    Returns:
        LoG response normalized by sigma^2
    """
    # Apply Gaussian smoothing
    smoothed = ndimage.gaussian_filter(img, sigma)
    
    # Compute Laplacian using convolution
    laplacian = ndimage.laplace(smoothed)
    
    # Normalize by sigma^2 for scale invariance
    return sigma**2 * laplacian

def detect_blobs_log(image, sigma_min=5, sigma_max=30, num_sigma=10, threshold=0.01):
    """
    Detect blobs using LoG across multiple scales
    
    Args:
        image: Input grayscale image
        sigma_min: Minimum sigma value
        sigma_max: Maximum sigma value
        num_sigma: Number of sigma values to test
        threshold: Threshold for blob detection (fraction of max response)
    
    Returns:
        List of detected blobs as (y, x, sigma, response)
    """
    # Convert to float and normalize
    img_float = image.astype(float) / 255.0
    
    # Create scale space
    sigma_list = np.linspace(sigma_min, sigma_max, num_sigma)
    
    # Store LoG responses for each scale
    log_stack = np.zeros((len(sigma_list), img_float.shape[0], img_float.shape[1]))
    
    print("Computing LoG across scales...")
    for idx, sigma in enumerate(sigma_list):
        log_response = laplacian_of_gaussian(img_float, sigma)
        log_stack[idx] = np.abs(log_response)  # Take absolute value
        print(f"Sigma {sigma:.2f}: Max response = {log_stack[idx].max():.6f}")
    
    # Find local maxima in scale space
    # A point is a maximum if it's greater than all 26 neighbors (3x3x3 cube)
    blobs = []
    
    # Set threshold based on maximum response
    max_response = log_stack.max()
    thresh_value = threshold * max_response
    
    print(f"\nThreshold value: {thresh_value:.6f}")
    
    # Check each scale (avoid boundaries)
    for s in range(1, len(sigma_list) - 1):
        for i in range(3, img_float.shape[0] - 3):
            for j in range(3, img_float.shape[1] - 3):
                # Current point response
                current = log_stack[s, i, j]
                
                # Check if above threshold
                if current < thresh_value:
                    continue
                
                # Check if local maximum in 3x3x3 neighborhood
                neighborhood = log_stack[s-1:s+2, i-3:i+4, j-3:j+4]
                
                if current == neighborhood.max():
                    # Store blob: (y, x, sigma, response)
                    blobs.append((i, j, sigma_list[s], current))
    
    print(f"\nDetected {len(blobs)} blobs before non-maximum suppression")
    
    # Non-maximum suppression: remove overlapping detections
    blobs = sorted(blobs, key=lambda x: x[3], reverse=True)
    final_blobs = []
    
    for blob in blobs:
        y, x, sigma, response = blob
        radius = sigma * np.sqrt(2)  # Blob radius
        
        # Check if too close to existing blob
        too_close = False
        for existing in final_blobs:
            ey, ex, esigma, _ = existing
            eradius = esigma * np.sqrt(2)
            dist = np.sqrt((y - ey)**2 + (x - ex)**2)
            
            # If overlapping significantly, skip
            if dist < 0.5 * (radius + eradius):
                too_close = True
                break
        
        if not too_close:
            final_blobs.append(blob)
    
    print(f"Final blob count after NMS: {len(final_blobs)}")
    
    return final_blobs, sigma_list

def visualize_blobs(image, blobs, title="Detected Blobs"):
    """
    Visualize detected blobs with circles
    
    Args:
        image: Original image
        blobs: List of (y, x, sigma, response) tuples
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Display image
    if len(image.shape) == 3:
        ax.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    else:
        ax.imshow(image, cmap='gray')
    
    # Draw circles for each blob
    for idx, (y, x, sigma, response) in enumerate(blobs):
        radius = sigma * np.sqrt(2)  # Convert sigma to radius
        circle = plt.Circle((x, y), radius, color='red', fill=False, linewidth=2)
        ax.add_patch(circle)
        
        # Annotate largest circles
        if idx < 10:  # Show info for top 10
            ax.text(x, y, f'{idx+1}', color='yellow', fontsize=8, 
                   ha='center', va='center', weight='bold')
    
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    return fig

# Main execution
if __name__ == "__main__":
    # Load image
    im = cv.imread('images/the_berry_farms_sunflower_field.jpeg', 
                   cv.IMREAD_REDUCED_COLOR_4)
    
    if im is None:
        print("Error: Could not load image. Check the path.")
        exit()
    
    print(f"Image shape: {im.shape}")
    
    # Convert to grayscale
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    
    # Detect blobs
    # Adjust these parameters based on sunflower sizes in your image
    blobs, sigma_range = detect_blobs_log(
        gray,
        sigma_min=3,      # Smaller sigma for small sunflowers
        sigma_max=25,     # Larger sigma for big sunflowers
        num_sigma=15,     # Number of scales to check
        threshold=0.005   # Lower threshold to detect more blobs
    )
    
    # Sort by response (strongest first)
    blobs_sorted = sorted(blobs, key=lambda x: x[3], reverse=True)
    
    # Report largest circles (top 10)
    print("\n" + "="*60)
    print("LARGEST DETECTED CIRCLES (Sunflowers)")
    print("="*60)
    print(f"{'Rank':<6} {'Y':>8} {'X':>8} {'Sigma':>8} {'Radius':>8} {'Response':>12}")
    print("-"*60)
    
    for idx, (y, x, sigma, response) in enumerate(blobs_sorted[:10]):
        radius = sigma * np.sqrt(2)
        print(f"{idx+1:<6} {y:>8.1f} {x:>8.1f} {sigma:>8.2f} {radius:>8.2f} {response:>12.6f}")
    
    print(f"\nSigma range used: {sigma_range[0]:.2f} to {sigma_range[-1]:.2f}")
    print(f"Total circles detected: {len(blobs)}")
    
    # Visualize
    visualize_blobs(im, blobs_sorted[:50], "Top 50 Detected Sunflowers")
    
    # Additional visualization: show top 10 only
    visualize_blobs(im, blobs_sorted[:10], "Top 10 Largest Sunflowers")
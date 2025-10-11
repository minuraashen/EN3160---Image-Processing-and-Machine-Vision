"""
Question 3: Homography Computation and Image Warping
Superimpose one image onto a planar surface in another image
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class HomographyWarper:
    """Class to handle homography computation and image warping"""
    
    def __init__(self):
        self.src_points = []
        self.dst_points = []
        self.current_image = None
        self.flag_image = None
        
    def compute_homography(self, src_pts, dst_pts):
        """
        Compute homography matrix from source to destination points
        
        Args:
            src_pts: 4x2 array of source points
            dst_pts: 4x2 array of destination points
        
        Returns:
            H: 3x3 homography matrix
        """
        # Need at least 4 point correspondences
        assert len(src_pts) >= 4 and len(dst_pts) >= 4
        
        # Build matrix A for homogeneous system Ah = 0
        A = []
        for i in range(len(src_pts)):
            x, y = src_pts[i]
            u, v = dst_pts[i]
            
            # Two equations per correspondence
            A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
            A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
        
        A = np.array(A)
        
        # Solve using SVD
        U, S, Vt = np.linalg.svd(A)
        
        # Homography is last column of V (last row of Vt)
        H = Vt[-1].reshape(3, 3)
        
        # Normalize so that H[2,2] = 1
        H = H / H[2, 2]
        
        return H
    
    def warp_image(self, src_img, dst_img, H):
        """
        Warp source image onto destination using homography
        
        Args:
            src_img: Source image to warp
            dst_img: Destination image
            H: 3x3 homography matrix
        
        Returns:
            result: Blended result image
        """
        h, w = dst_img.shape[:2]
        
        # Warp the source image
        warped = cv.warpPerspective(src_img, H, (w, h))
        
        # Create mask for blending
        mask = np.zeros((src_img.shape[0], src_img.shape[1]), dtype=np.uint8)
        mask[:, :] = 255
        warped_mask = cv.warpPerspective(mask, H, (w, h))
        
        # Blend images
        result = dst_img.copy()
        
        # Where mask is non-zero, use warped image
        mask_bool = warped_mask > 0
        result[mask_bool] = warped[mask_bool]
        
        return result, warped, warped_mask
    
    def click_points(self, event, x, y, flags, param):
        """Mouse callback for point selection"""
        if event == cv.EVENT_LBUTTONDOWN:
            self.dst_points.append([x, y])
            
            # Draw point
            cv.circle(self.current_image, (x, y), 5, (0, 255, 0), -1)
            cv.putText(self.current_image, f'{len(self.dst_points)}', 
                      (x+10, y+10), cv.FONT_HERSHEY_SIMPLEX, 
                      0.5, (0, 255, 0), 2)
            cv.imshow('Select 4 points', self.current_image)
            
            print(f"Point {len(self.dst_points)}: ({x}, {y})")
            
            if len(self.dst_points) == 4:
                print("4 points selected. Press any key to continue.")
    
    def select_points_interactive(self, dst_img):
        """
        Interactive point selection on destination image
        
        Args:
            dst_img: Destination image
        
        Returns:
            dst_points: Selected points
        """
        self.current_image = dst_img.copy()
        self.dst_points = []
        
        cv.namedWindow('Select 4 points')
        cv.setMouseCallback('Select 4 points', self.click_points)
        
        print("\nClick 4 points on the planar surface (in order: top-left, top-right, bottom-right, bottom-left)")
        print("Press any key after selecting 4 points")
        
        cv.imshow('Select 4 points', self.current_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        return np.array(self.dst_points, dtype=np.float32)


def demonstrate_homography(flag_path, target_path, output_path='result.jpg'):
    """
    Complete demonstration of homography warping
    
    Args:
        flag_path: Path to flag/source image
        target_path: Path to target architectural image
        output_path: Path to save result
    """
    # Load images
    flag = cv.imread(flag_path)
    target = cv.imread(target_path)
    
    if flag is None or target is None:
        print("Error loading images!")
        return
    
    print(f"Flag image shape: {flag.shape}")
    print(f"Target image shape: {target.shape}")
    
    # Define source points (corners of flag image)
    h_flag, w_flag = flag.shape[:2]
    src_points = np.array([
        [0, 0],              # Top-left
        [w_flag-1, 0],       # Top-right
        [w_flag-1, h_flag-1], # Bottom-right
        [0, h_flag-1]        # Bottom-left
    ], dtype=np.float32)
    
    # Interactive point selection or manual specification
    warper = HomographyWarper()
    
    # Option 1: Interactive selection
    # dst_points = warper.select_points_interactive(target)
    
    # Option 2: Manual specification (example coordinates)
    # Modify these for your specific images
    dst_points = np.array([
        [200, 150],   # Top-left
        [500, 180],   # Top-right
        [480, 400],   # Bottom-right
        [220, 380]    # Bottom-left
    ], dtype=np.float32)
    
    print(f"\nSource points (flag corners):\n{src_points}")
    print(f"\nDestination points (target plane):\n{dst_points}")
    
    # Compute homography
    H = warper.compute_homography(src_points, dst_points)
    print(f"\nComputed Homography Matrix:\n{H}")
    
    # Verify with OpenCV (for comparison)
    H_cv, _ = cv.findHomography(src_points, dst_points)
    print(f"\nOpenCV Homography (for verification):\n{H_cv}")
    
    # Warp and blen
"""
Question 2: RANSAC for Line and Circle Fitting
Implements RANSAC algorithm for robust fitting with outliers
"""

import numpy as np
from scipy.optimize import minimize
from scipy import linalg
import matplotlib.pyplot as plt

# Generate noisy point set (from assignment)
np.random.seed(42)  # For reproducibility
N = 100
half_n = N // 2

# Generate circle points
r = 10
x0_gt, y0_gt = 2, 3  # Ground truth center
s = r / 16
t = np.random.uniform(0, 2*np.pi, half_n)
n = s * np.random.randn(half_n)
x, y = x0_gt + (r + n) * np.cos(t), y0_gt + (r + n) * np.sin(t)
X_circ = np.hstack((x.reshape(half_n, 1), y.reshape(half_n, 1)))

# Generate line points
s = 1.
m, b = -1, 2
x = np.linspace(-12, 12, half_n)
y = m*x + b + s * np.random.randn(half_n)
X_line = np.hstack((x.reshape(half_n, 1), y.reshape(half_n, 1)))

# Combine all points
X = np.vstack((X_circ, X_line))

print(f"Total points: {len(X)}")
print(f"Line points: {len(X_line)}")
print(f"Circle points: {len(X_circ)}")


def fit_line_to_points(points):
    """
    Fit a line to points using Total Least Squares (TLS)
    Line parameterized as: a*x + b*y + d = 0, where [a,b] is unit normal
    
    Args:
        points: Nx2 array of points
    
    Returns:
        a, b, d: Line parameters with ||[a,b]|| = 1
    """
    # Center the points
    centroid = points.mean(axis=0)
    centered = points - centroid
    
    # SVD for TLS
    U, S, Vt = np.linalg.svd(centered)
    
    # Normal vector is last row of Vt (smallest singular value)
    normal = Vt[-1, :]
    a, b = normal
    
    # Ensure unit normal
    norm = np.sqrt(a**2 + b**2)
    a, b = a/norm, b/norm
    
    # Compute d from centroid
    d = -(a * centroid[0] + b * centroid[1])
    
    return a, b, d


def distance_point_to_line(points, a, b, d):
    """
    Compute perpendicular distance from points to line
    Line: a*x + b*y + d = 0 with ||[a,b]|| = 1
    
    Args:
        points: Nx2 array
        a, b, d: Line parameters
    
    Returns:
        Array of distances
    """
    return np.abs(a * points[:, 0] + b * points[:, 1] + d)


def ransac_line(points, threshold=0.5, min_inliers=30, max_iterations=1000):
    """
    RANSAC for line fitting
    
    Args:
        points: Nx2 array of points
        threshold: Distance threshold for inliers
        min_inliers: Minimum number of inliers
        max_iterations: Maximum RANSAC iterations
    
    Returns:
        best_a, best_b, best_d: Best line parameters
        best_inliers: Indices of inliers
        best_sample: Indices of best sample
    """
    best_inliers = []
    best_sample = None
    best_params = None
    
    for iteration in range(max_iterations):
        # Randomly sample 2 points
        sample_idx = np.random.choice(len(points), 2, replace=False)
        sample = points[sample_idx]
        
        # Fit line to sample
        try:
            a, b, d = fit_line_to_points(sample)
        except:
            continue
        
        # Compute distances for all points
        distances = distance_point_to_line(points, a, b, d)
        
        # Find inliers
        inliers = np.where(distances < threshold)[0]
        
        # Update best model if more inliers
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_sample = sample_idx
            best_params = (a, b, d)
    
    # Refit line using all inliers
    if len(best_inliers) >= min_inliers:
        a, b, d = fit_line_to_points(points[best_inliers])
        best_params = (a, b, d)
    
    print(f"Line RANSAC: {len(best_inliers)} inliers found")
    
    return best_params, best_inliers, best_sample


def fit_circle_to_points(points):
    """
    Fit circle to points using least squares optimization
    Circle: (x - x0)^2 + (y - y0)^2 = r^2
    
    Args:
        points: Nx2 array
    
    Returns:
        x0, y0, r: Circle parameters
    """
    def circle_residuals(params, points):
        """Compute sum of squared radial errors"""
        x0, y0, r = params
        distances = np.sqrt((points[:, 0] - x0)**2 + (points[:, 1] - y0)**2)
        return np.sum((distances - r)**2)
    
    # Initial guess: centroid and average distance
    centroid = points.mean(axis=0)
    distances = np.sqrt((points[:, 0] - centroid[0])**2 + 
                       (points[:, 1] - centroid[1])**2)
    r_init = distances.mean()
    
    initial_guess = [centroid[0], centroid[1], r_init]
    
    # Optimize
    result = minimize(circle_residuals, initial_guess, args=(points,),
                     method='Nelder-Mead')
    
    return result.x


def distance_point_to_circle(points, x0, y0, r):
    """
    Compute radial distance from points to circle
    
    Args:
        points: Nx2 array
        x0, y0, r: Circle parameters
    
    Returns:
        Array of radial errors
    """
    distances = np.sqrt((points[:, 0] - x0)**2 + (points[:, 1] - y0)**2)
    return np.abs(distances - r)


def ransac_circle(points, threshold=0.5, min_inliers=30, max_iterations=1000):
    """
    RANSAC for circle fitting
    
    Args:
        points: Nx2 array of points
        threshold: Radial error threshold
        min_inliers: Minimum inliers
        max_iterations: Max iterations
    
    Returns:
        best_params: (x0, y0, r)
        best_inliers: Inlier indices
        best_sample: Sample indices
    """
    best_inliers = []
    best_sample = None
    best_params = None
    
    for iteration in range(max_iterations):
        # Sample 3 points for circle
        sample_idx = np.random.choice(len(points), 3, replace=False)
        sample = points[sample_idx]
        
        # Fit circle to sample
        try:
            params = fit_circle_to_points(sample)
            x0, y0, r = params
            
            # Check validity
            if r <= 0 or r > 50:  # Reasonable radius constraint
                continue
        except:
            continue
        
        # Compute distances
        distances = distance_point_to_circle(points, x0, y0, r)
        
        # Find inliers
        inliers = np.where(distances < threshold)[0]
        
        # Update best
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_sample = sample_idx
            best_params = params
    
    # Refit with all inliers
    if len(best_inliers) >= min_inliers:
        best_params = fit_circle_to_points(points[best_inliers])
    
    print(f"Circle RANSAC: {len(best_inliers)} inliers found")
    
    return best_params, best_inliers, best_sample


# Part (a): Fit line using RANSAC
print("\n=== Part (a): Line Fitting ===")
line_params, line_inliers, line_sample = ransac_line(
    X, threshold=1.5, min_inliers=40, max_iterations=1000
)
a, b, d = line_params
print(f"Line parameters: a={a:.4f}, b={b:.4f}, d={d:.4f}")
print(f"Normal vector magnitude: {np.sqrt(a**2 + b**2):.6f}")


# Part (b): Remove line inliers and fit circle
print("\n=== Part (b): Circle Fitting ===")
remaining_points = np.delete(X, line_inliers, axis=0)
print(f"Remaining points after line removal: {len(remaining_points)}")

circle_params, circle_inliers_rel, circle_sample_rel = ransac_circle(
    remaining_points, threshold=1.0, min_inliers=35, max_iterations=1000
)
x0, y0, radius = circle_params
print(f"Circle parameters: center=({x0:.4f}, {y0:.4f}), radius={radius:.4f}")
print(f"Ground truth: center=({x0_gt}, {y0_gt}), radius={r}")


# Part (c): Visualization
print("\n=== Part (c): Creating Visualization ===")
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# Plot all points
ax.scatter(X[:, 0], X[:, 1], c='lightgray', s=30, alpha=0.6, 
          label='All points', zorder=1)

# Plot line inliers
ax.scatter(X[line_inliers, 0], X[line_inliers, 1], c='blue', s=40,
          label='Line inliers', zorder=2)

# Plot best sample for line
ax.scatter(X[line_sample, 0], X[line_sample, 1], c='cyan', s=100,
          marker='s', edgecolors='black', linewidths=2,
          label='Best sample for line', zorder=5)

# Plot circle inliers (map back to original indices)
remaining_indices = np.delete(np.arange(len(X)), line_inliers)
circle_inliers = remaining_indices[circle_inliers_rel]
ax.scatter(X[circle_inliers, 0], X[circle_inliers, 1], c='red', s=40,
          label='Circle inliers', zorder=3)

# Plot best sample for circle
circle_sample = remaining_indices[circle_sample_rel]
ax.scatter(X[circle_sample, 0], X[circle_sample, 1], c='orange', s=100,
          marker='s', edgecolors='black', linewidths=2,
          label='Best sample for circle', zorder=6)

# Plot ground truth circle
circle_gt = plt.Circle((x0_gt, y0_gt), r, color='green', fill=False,
                       linewidth=2, linestyle='--', label='Ground truth circle')
ax.add_patch(circle_gt)
ax.plot(x0_gt, y0_gt, 'g+', markersize=15, markeredgewidth=2)

# Plot RANSAC circle
circle_ransac = plt.Circle((x0, y0), radius, color='darkred', fill=False,
                          linewidth=2, label='RANSAC circle')
ax.add_patch(circle_ransac)
ax.plot(x0, y0, 'r+', markersize=15, markeredgewidth=2)

# Plot ground truth line
x_range = np.array([ax.get_xlim()[0], ax.get_xlim()[1]])
y_range = m * x_range + b
ax.plot(x_range, y_range, 'g--', linewidth=2, label='Ground truth line')

# Plot RANSAC line
# Line: a*x + b*y + d = 0 => y = -(a*x + d)/b
if abs(b) > 1e-6:
    y_ransac = -(a * x_range + d) / b
else:  # Vertical line
    x_ransac = -d / a * np.ones_like(x_range)
    y_ransac = x_range
    x_range = x_ransac

ax.plot(x_range, y_ransac, 'b-', linewidth=2, label='RANSAC line')

ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('RANSAC Line and Circle Fitting', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
ax.set_xlim(-15, 15)
ax.set_ylim(-13, 17)

plt.tight_layout()
plt.savefig('ransac_fitting.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== Part (d): Discussion ===")
print("""
If we fit the circle first, the algorithm would:
1. Incorrectly include some line points as circle inliers (since line points
   can be at similar distances from a potential circle center)
2. Result in a poorly fitted circle with larger error
3. Leave fewer and more scattered line points for the subsequent line fit
4. The line fit would then be less accurate due to contamination

The order matters because:
- Lines have fewer parameters (3) than circles (3), making them easier to fit
- Line fitting is more robust to outliers geometrically
- Fitting the simpler model first removes a clean subset of points
- This sequential approach works best when going from simpler to complex models
""")
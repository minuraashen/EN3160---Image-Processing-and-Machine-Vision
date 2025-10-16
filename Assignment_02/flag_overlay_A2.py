import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images
building_img_path = 'building.jpg'
flag_img_path = 'avengers.jpg'

building_img = cv2.imread(building_img_path)
building_copy = building_img.copy()
flag_img = cv2.imread(flag_img_path)

# Resize flag to match size with the building (optional, if flag needs resizing)
flag_img = cv2.resize(flag_img, (building_img.shape[1], building_img.shape[0]))

# Create a list to store points
points_building = []

# Define the function that will be called on mouse events
def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Store the clicked points
        points_building.append([x, y])
        print(f"Point selected: {x}, {y}")
        
        # Display the points on the image for feedback
        cv2.circle(building_copy, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select 4 points", building_copy)

# Show the building image and set the mouse callback
cv2.imshow("Select 4 points", building_copy)
cv2.setMouseCallback("Select 4 points", select_points)

# Wait until 4 points are selected
while len(points_building) < 4:
    cv2.waitKey(1)

# Close the image window
cv2.destroyAllWindows()

# Convert the selected points to a numpy array
pts_building = np.array(points_building, dtype=np.float32)

# Corresponding points in the flag image (covering the entire flag)
pts_flag = np.array([[0, 0], [flag_img.shape[1], 0], [flag_img.shape[1], flag_img.shape[0]], [0, flag_img.shape[0]]], dtype=np.float32)

# Compute the homography matrix
H, _ = cv2.findHomography(pts_flag, pts_building)

# Warp the flag image onto the building image
warped_flag = cv2.warpPerspective(flag_img, H, (building_img.shape[1], building_img.shape[0]))

# Create a mask for the warped flag
mask = np.zeros_like(building_img, dtype=np.uint8)
cv2.fillConvexPoly(mask, pts_building.astype(int), (255, 255, 255))

# Blend the warped flag with the building image
blended = cv2.addWeighted(building_img, 1, warped_flag, 0.5, 0)

# Display the final result
plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Optionally save the output
output_path = './flag_output.png'
cv2.imwrite(output_path, blended)
print(f"Blended image saved at {output_path}")

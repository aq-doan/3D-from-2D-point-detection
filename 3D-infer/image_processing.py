import cv2
import numpy as np
import os

# Function to capture mouse click events for mat corners
def click_event_corners(event, x, y, flags, params):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            cv2.circle(params, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("image", params)
            if len(points) == 4:
                cv2.setMouseCallback("image", lambda *args : None)

# Load the image
image_path = './sample/image_sample1.png'
original_img = cv2.imread(image_path)
if original_img is None:
    print("Error loading image.")
    exit()

img = original_img.copy()

# Step 1: Annotate the mat corners
points = []
cv2.imshow("image", img)
cv2.setMouseCallback("image", click_event_corners, img)
cv2.waitKey(0)

if len(points) != 4:
    print("You need to select exactly 4 corners.")
    cv2.destroyAllWindows()
    exit()

# Print the coordinates of the selected corners
for i, point in enumerate(points):
    print(f"Corner {i+1} (label {chr(65+i)}): {point}")

# Draw the rectangle and show the image again with annotations
labels = ['A', 'B', 'C', 'D']
for point, label in zip(points, labels):
    cv2.circle(img, point, 5, (0, 0, 255), -1)
    cv2.putText(img, label, point, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# Draw lines between the points to visualize the rectangle
cv2.line(img, points[0], points[1], (255, 0, 0), 2)
cv2.line(img, points[1], points[2], (255, 0, 0), 2)
cv2.line(img, points[2], points[3], (255, 0, 0), 2)
cv2.line(img, points[3], points[0], (255, 0, 0), 2)

# Define a standard size for the destination points (e.g., 1000x500 pixels)
#change this to ratio image based on the real distance

dst_width = 1000
dst_height = 500
dst_points = np.array([
    [0, 0],
    [dst_width, 0],
    [dst_width, dst_height],
    [0, dst_height]
], dtype='float32')

# Convert the annotated points to a numpy array
src_points = np.array(points, dtype='float32')

# Calculate the perspective transformation matrix
M = cv2.getPerspectiveTransform(src_points, dst_points)

# Apply the perspective transformation to warp the image using a better interpolation method
warped_img = cv2.warpPerspective(original_img, M, (dst_width, dst_height), flags=cv2.INTER_CUBIC)

# Display the warped image
cv2.imshow("Warped Image", warped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Create output directory if it doesn't exist
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the warped image
warped_output_path = os.path.join(output_dir, 'warped_image.png')
cv2.imwrite(warped_output_path, warped_img)
print(f"Warped image saved to {warped_output_path}")

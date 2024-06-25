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

# Function to capture mouse click events for the selected point
def click_event_point(event, x, y, flags, params):
    global selected_point
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_point = (x, y)
        cv2.circle(params, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow("image", params)
        cv2.setMouseCallback("image", lambda *args : None)

# Load the image
image_path = './sample/image_sample1.png'
original_img = cv2.imread(image_path)
if original_img is None:
    print("Error loading image.")
    exit()

# Resize for better viewing
scale_percent = 50  # percent of original size
width = int(original_img.shape[1] * scale_percent / 100)
height = int(original_img.shape[0] * scale_percent / 100)
dim = (width, height)
img = cv2.resize(original_img, dim, interpolation=cv2.INTER_AREA)

# Step 1: Annotate the mat corners
points = []
cv2.imshow("image", img)
cv2.setMouseCallback("image", click_event_corners, img)
cv2.waitKey(0)

if len(points) != 4:
    print("You need to select exactly 4 corners.")
    cv2.destroyAllWindows()
    exit()

# Step 2: Ask for real-world dimensions
real_length = float(input("Enter the real length of the mat: "))
real_width = float(input("Enter the real width of the mat: "))

# Draw the rectangle and show the image again with annotations
for point in points:
    cv2.circle(img, point, 5, (0, 0, 255), -1)

# Draw lines between the points to visualize the rectangle
cv2.line(img, points[0], points[1], (255, 0, 0), 2)
cv2.line(img, points[1], points[2], (255, 0, 0), 2)
cv2.line(img, points[2], points[3], (255, 0, 0), 2)
cv2.line(img, points[3], points[0], (255, 0, 0), 2)

# Display real-world dimensions on the image
cv2.putText(img, f"Length: {real_length} units", ((points[0][0] + points[1][0]) // 2, (points[0][1] + points[1][1]) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
cv2.putText(img, f"Width: {real_width} units", ((points[1][0] + points[2][0]) // 2, (points[1][1] + points[2][1]) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# Display the annotated mat with dimensions
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 3: Annotate a point within the mat plane
selected_point = None
cv2.imshow("image", img)
cv2.setMouseCallback("image", click_event_point, img)
cv2.waitKey(0)

if selected_point is None:
    print("You need to select a point within the mat.")
    cv2.destroyAllWindows()
    exit()

# Calculate real-world distances from the selected point to each corner
def calculate_real_distance(point1, point2, real_length, real_width):
    pixel_distance = np.linalg.norm(np.array(point1) - np.array(point2))
    avg_pixel_distance_length = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
    avg_pixel_distance_width = np.linalg.norm(np.array(points[1]) - np.array(points[2]))
    scale_factor_length = real_length / avg_pixel_distance_length
    scale_factor_width = real_width / avg_pixel_distance_width
    real_distance_length = pixel_distance * scale_factor_length
    real_distance_width = pixel_distance * scale_factor_width
    real_distance = (real_distance_length + real_distance_width) / 2
    return real_distance

distances = [calculate_real_distance(selected_point, corner, real_length, real_width) for corner in points]

# Draw lines from each corner to the selected point and annotate distances
for i, distance in enumerate(distances):
    cv2.line(img, selected_point, points[i], (0, 255, 0), 2)
    midpoint = ((selected_point[0] + points[i][0]) // 2, (selected_point[1] + points[i][1]) // 2)
    cv2.putText(img, f"{distance:.2f} units", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Draw the selected point
cv2.circle(img, selected_point, 5, (255, 0, 0), -1)

# Display the final annotated image
cv2.imshow("image", img)
cv2.waitKey(5000)
cv2.destroyAllWindows()

# Create output directory if it doesn't exist
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save annotated image
output_path = os.path.join(output_dir, 'annotated_image.png')
cv2.imwrite(output_path, img)
print(f"Annotated image saved to {output_path}")

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

# Ask for real-world dimensions
real_length = float(input("Enter the real length of the mat: "))
real_width = float(input("Enter the real width of the mat: "))

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

# Function to calculate the scale factor based on pixel and real-world distances
def calculate_real_distance(pixel_distance, real_distance):
    return real_distance / pixel_distance

# Calculate pixel distances and scale factors for length and width
pixel_distance_length = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
pixel_distance_width = np.linalg.norm(np.array(points[1]) - np.array(points[2]))
scale_factor_length = calculate_real_distance(pixel_distance_length, real_length)
scale_factor_width = calculate_real_distance(pixel_distance_width, real_width)

# Display pixel and real-world distances on the image
distances = [
    (pixel_distance_length, real_length),  # A to B
    (pixel_distance_width, real_width),   # B to C
    (pixel_distance_length, real_length),  # C to D
    (pixel_distance_width, real_width)    # D to A
]
midpoints = [
    ((points[0][0] + points[1][0]) // 2, (points[0][1] + points[1][1]) // 2),
    ((points[1][0] + points[2][0]) // 2, (points[1][1] + points[2][1]) // 2),
    ((points[2][0] + points[3][0]) // 2, (points[2][1] + points[3][1]) // 2),
    ((points[3][0] + points[0][0]) // 2, (points[3][1] + points[0][1]) // 2)
]
for (pixel_distance, real_distance), midpoint in zip(distances, midpoints):
    cv2.putText(img, f"{pixel_distance:.2f} px / {real_distance:.2f} units", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Function to interpolate points between two points
def interpolate_points(p1, p2, num_points):
    return [(int(p1[0] + i * (p2[0] - p1[0]) / num_points),
             int(p1[1] + i * (p2[1] - p1[1]) / num_points)) for i in range(num_points + 1)]

# Determine the number of grid lines based on real-world dimensions
num_rows = int(real_length)
num_cols = int(real_width)

# Generate the grid points
top_points = interpolate_points(points[0], points[1], num_cols)
bottom_points = interpolate_points(points[3], points[2], num_cols)
left_points = interpolate_points(points[0], points[3], num_rows)
right_points = interpolate_points(points[1], points[2], num_rows)

# Draw the vertical grid lines
for i in range(num_cols + 1):
    cv2.line(img, top_points[i], bottom_points[i], (0, 255, 255), 1)

# Draw the horizontal grid lines
for i in range(num_rows + 1):
    cv2.line(img, left_points[i], right_points[i], (0, 255, 255), 1)

# Display the grid
cv2.imshow("image", img)
cv2.waitKey(0)

# Function to calculate the distance between two points
def calculate_distances(point1, point2, scale_factor_length, scale_factor_width):
    pixel_distance = np.linalg.norm(np.array(point1) - np.array(point2))
    real_distance_length = pixel_distance * scale_factor_length
    real_distance_width = pixel_distance * scale_factor_width
    real_distance = (real_distance_length + real_distance_width) / 2
    return pixel_distance, real_distance

# Annotate a point within the mat plane  ***try different grid sizes
selected_point = None
def click_event_point(event, x, y, flags, params):
    global selected_point
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_point = (x, y)
        cv2.circle(params, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow("image", params)
        cv2.setMouseCallback("image", lambda *args : None)

cv2.imshow("image", img)
cv2.setMouseCallback("image", click_event_point, img)
cv2.waitKey(0)

if selected_point is None:
    print("You need to select a point within the mat.")
    cv2.destroyAllWindows()
    exit()

# Display the coordinates of the selected point
print(f"Selected Point: {selected_point}")
cv2.putText(img, f"({selected_point[0]}, {selected_point[1]})", selected_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Calculate and display the distances from the selected point to each corner
distances = [calculate_distances(selected_point, corner, scale_factor_length, scale_factor_width) for corner in points]
for i, (pixel_distance, real_distance) in enumerate(distances):
    print(f"Distance to {labels[i]}: {pixel_distance:.2f} px, {real_distance:.2f} units")
    midpoint = ((selected_point[0] + points[i][0]) // 2, (selected_point[1] + points[i][1]) // 2)
    cv2.line(img, selected_point, points[i], (0, 255, 0), 2)
    cv2.putText(img, f"{pixel_distance:.2f} px / {real_distance:.2f} units", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Draw the selected point
cv2.circle(img, selected_point, 5, (255, 0, 0), -1)

# Display the final annotated image
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Create output directory if it doesn't exist
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the annotated image with grid and distances
output_path = os.path.join(output_dir, 'annotated_image_with_grid_and_point.png')
cv2.imwrite(output_path, img)
print(f"Annotated image with grid and point saved to {output_path}")

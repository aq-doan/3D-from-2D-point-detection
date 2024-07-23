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
                cv2.setMouseCallback("image", lambda *args: None)

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

# Set the real-world dimensions
real_length = 120.0  # cm
real_width = 90.0  # cm

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
vertical_lines = []
for i in range(num_cols + 1):
    vertical_lines.append((top_points[i], bottom_points[i]))
    cv2.line(img, top_points[i], bottom_points[i], (0, 255, 255), 1)

# Draw the horizontal grid lines
horizontal_lines = []
for i in range(num_rows + 1):
    horizontal_lines.append((left_points[i], right_points[i]))
    cv2.line(img, left_points[i], right_points[i], (0, 255, 255), 1)

# Display the grid
cv2.imshow("image", img)
cv2.waitKey(0)

# Function to find the shortest distance from a point to a line
def point_to_line_dist(point, line):
    p1, p2 = np.array(line[0]), np.array(line[1])
    p3 = np.array(point)
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

# Function to find the closest grid line to the selected point
def find_closest_grid_line(point, lines):
    min_distance = float('inf')
    closest_line = None
    for line in lines:
        distance = point_to_line_dist(point, line)
        if distance < min_distance:
            min_distance = distance
            closest_line = line
    return closest_line

# Annotate a point within the mat plane
selected_point = None
def click_event_point(event, x, y, flags, params):
    global selected_point
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_point = (x, y)
        # Draw the selected point immediately
        cv2.circle(params, (x, y), 5, (255, 0, 0), -1)
        cv2.putText(params, f"({x, y})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Find the closest vertical and horizontal grid lines
        closest_vertical_line = find_closest_grid_line(selected_point, vertical_lines)
        closest_horizontal_line = find_closest_grid_line(selected_point, horizontal_lines)

        # Highlight the closest grid lines
        cv2.line(params, closest_vertical_line[0], closest_vertical_line[1], (0, 0, 255), 2)
        cv2.line(params, closest_horizontal_line[0], closest_horizontal_line[1], (0, 0, 255), 2)

        cv2.imshow("image", params)
        cv2.setMouseCallback("image", lambda *args: None)

cv2.imshow("image", img)
cv2.setMouseCallback("image", click_event_point, img)
cv2.waitKey(0)

if selected_point is None:
    print("You need to select a point within the mat.")
    cv2.destroyAllWindows()
    exit()

# Display the coordinates of the selected point
print(f"Selected Point: {selected_point}")

# Find the intersection of the closest vertical and horizontal lines
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

closest_vertical_line = find_closest_grid_line(selected_point, vertical_lines)
closest_horizontal_line = find_closest_grid_line(selected_point, horizontal_lines)

intersection_point = line_intersection(closest_vertical_line, closest_horizontal_line)
print(f"Intersection Point: {intersection_point}")

# Translate the points so that the intersection point is at the origin
translated_points = [(point[0] - intersection_point[0], point[1] - intersection_point[1]) for point in points]

# Calculate real-world distances
def pixel_to_real_distance(pixel_distance, real_length, pixel_length):
    return pixel_distance * real_length / pixel_length

real_distances = []
for point in translated_points:
    #Calculate the pixel distance from the intersection point (which is now the origin) to the current translated point
    pixel_distance = np.linalg.norm(np.array(point))
    #np.linalg.norm(np.array(points[0]) - np.array(points[1])) calculates the pixel distance
    # between two original corner points (points[0] and points[1]) in the image. 
    real_distances.append(pixel_to_real_distance(pixel_distance, real_length, np.linalg.norm(np.array(points[0]) - np.array(points[1]))))

# Print the real distances
for i, dist in enumerate(real_distances):
    print(f"Real distance to corner {labels[i]}: {dist:.2f} cm")

# Draw the selected point and intersection point
cv2.circle(img, selected_point, 5, (255, 0, 0), -1)
cv2.circle(img, tuple(map(int, intersection_point)), 5, (0, 255, 0), -1)
cv2.putText(img, "Intersection", tuple(map(int, intersection_point)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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

import cv2
import mediapipe as mp
import pandas as pd
import os
import math

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Video file path
video_path = 'sample/sample.mp4'
output_csv = 'output/foot_tip_positions.csv'
output_video = 'output/output_video.mp4'
frame_output_dir = 'output/frames'

# Define foot tip landmarks
foot_tip_ids = [mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]

# Create output directory if it doesn't exist
if not os.path.exists('output'):
    os.makedirs('output')

# Create frames output directory if it doesn't exist
if not os.path.exists(frame_output_dir):
    os.makedirs(frame_output_dir)

# Initialize video capture
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

data = []
frame_count = 0

# Initialize flag
object_detected = False

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    left_foot = None
    right_foot = None
    distance = None 

    if results.pose_landmarks:
        # Set flag to True when object is detected for the first time
        if not object_detected:
            object_detected = True

        for idx in foot_tip_ids:
            landmark = results.pose_landmarks.landmark[idx]
            h, w, c = frame.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            # Store foot tip positions
            if idx == mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value:
                left_foot = (cx, cy)
            elif idx == mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value:
                right_foot = (cx, cy)

            # Calculate distance between left and right foot
            if left_foot and right_foot:
                distance = math.sqrt((right_foot[0] - left_foot[0]) ** 2 + (right_foot[1] - left_foot[1]) ** 2)

            # Append data to list
            if object_detected:
                data.append({
                    'frame': cap.get(cv2.CAP_PROP_POS_FRAMES),
                    'foot_tip_id': idx,
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'distance': distance  # Add distance to the data dictionary
                })

            # Draw landmarks and IDs on frame
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, str(idx), (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Draw distance on frame
    if distance and object_detected:
        cv2.putText(frame, f'Distance: {distance:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Write the frame to the output video
    if object_detected:
        out.write(frame)

        # Save the current frame as an image
        frame_filename = os.path.join(frame_output_dir, f'frame_{frame_count:04d}.png')
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

# Convert to DataFrame and save to CSV
if object_detected:
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

# Release resources
cap.release()
out.release()
pose.close()
cv2.destroyAllWindows()
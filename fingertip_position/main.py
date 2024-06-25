import cv2  # Importing the OpenCV library for image processing
import mediapipe as mp  
import pandas as pd  
import os  

# Initialize Mediapipe hands
mp_hands = mp.solutions.hands 
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)  # Initializing the hand tracking model
mp_drawing = mp.solutions.drawing_utils  # Utility functions for drawing landmarks on images

# Directory containing sample images
image_dir = 'sample_images' 
output_csv = 'fingertip_positions.csv' 
output_dir = 'output_images'  

# Create output directory if it doesn't exist
if not os.path.exists(output_dir): 
    os.makedirs(output_dir)  

# Define fingertip landmarks
fingertip_ids = [4, 8, 12, 16, 20]  # List of landmark IDs corresponding to fingertips
data = [] 

# Process each image
for filename in os.listdir(image_dir):  
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        file_path = os.path.join(image_dir, filename) 
        image = cv2.imread(file_path)  # Reading the image using OpenCV
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converting the image to RGB format
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:  
            for hand_landmarks in results.multi_hand_landmarks:  # Iterating over each detected hand
                for idx in fingertip_ids:  # Iterating over each fingertip landmark ID
                    landmark = hand_landmarks.landmark[idx]  # Extracting the landmark coordinates
                    # Append data to list
                    data.append({
                        'image': filename,
                        'fingertip_id': idx,
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })  # Storing the extracted data in a dictionary and appending it to the list

                    
                    h, w, c = image.shape  # Getting the height, width, and number of channels of the image
                    cx, cy = int(landmark.x * w), int(landmark.y * h)  # Calculating the pixel coordinates of the landmark
                    #Drawing
                    cv2.circle(image, (cx, cy), 5, (0, 255, 0), cv2.FILLED) 
                    coordinate_text = f"{idx}: ({cx}, {cy})"  
                    cv2.putText(image, coordinate_text, (cx - 20, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1) 

            # Save the annotated image
            output_path = os.path.join(output_dir, filename)  
            cv2.imwrite(output_path, image)  

# Convert to DataFrame and save to CSV
df = pd.DataFrame(data) 
df.to_csv(output_csv, index=False) 


hands.close() 

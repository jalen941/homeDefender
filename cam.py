import cv2
from imgbeddings import imgbeddings
import firebaseUtility
import numpy as np
from PIL import Image
import requests

import os
import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Load the alarm sound file
#alarm_sound = pygame.mixer.Sound('alarm.wav')  # Replace 'alarm.wav' with the path to your alarm sound file

# Function to play the alarm sound




# Load the Haar cascade classifier for face detection
alg = "faceData.xml"
haar_cascade = cv2.CascadeClassifier(alg)

# Initialize the webcam
video = cv2.VideoCapture(0)


# Set frame size
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Set frame rate
video.set(cv2.CAP_PROP_FPS, 30)

prev_frame = None


# Load the Firebase utility
cred_path = "homedefender-a7497-firebase-adminsdk-p05pu-d6d28cad6b.json"
db_url = "https://homedefender-a7497-default-rtdb.firebaseio.com/"
firebase_utility = firebaseUtility.FirebaseUtility(cred_path, db_url)

# Function to calculate embeddings for a face image

def send_message(message):
    resp = requests.post('https://textbelt.com/text', {
        'phone': '7706801965',
        'message': message,
        'key': 'textbelt',
    })
    print(resp.json())
def calculate_embeddings(face_img):
    # Convert the numpy array to a PIL image
    pil_image = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    # Calculate embeddings using imgbeddings
    ibed = imgbeddings()
    embedding = ibed.to_embeddings(pil_image)
    return embedding[0].tolist()




def upload_sample_face_embedding(sample_img_path):
    # Read the sample face image
    sample_img = cv2.imread(sample_img_path)
    # Calculate embeddings for the sample face image
    sample_embedding = calculate_embeddings(sample_img)
    # Put the data into the "sample_faces" table in Firebase
    firebase_utility.put_data("sample_faces", {"embedding": sample_embedding})

# Usage: Call this function with the path to the sample face image

upload_sample_face_embedding("myFace.jpg")
upload_sample_face_embedding("WIN_20240402_19_12_24_Pro.jpg")
upload_sample_face_embedding("WIN_20240402_19_12_26_Pro.jpg")
upload_sample_face_embedding("WIN_20240402_19_12_27_Pro.jpg")
upload_sample_face_embedding("WIN_20240402_19_12_28_Pro.jpg")
upload_sample_face_embedding("WIN_20240402_19_12_29_Pro.jpg")
upload_sample_face_embedding("WIN_20240402_19_12_30_Pro.jpg")
upload_sample_face_embedding("WIN_20240402_19_12_31_Pro.jpg")
upload_sample_face_embedding("WIN_20240402_19_12_33_Pro.jpg")
upload_sample_face_embedding("WIN_20240402_19_12_34_Pro.jpg")
upload_sample_face_embedding("WIN_20240402_19_12_35_Pro.jpg")
upload_sample_face_embedding("WIN_20240402_19_12_36_Pro.jpg")
upload_sample_face_embedding("WIN_20240402_19_12_38_Pro.jpg")
upload_sample_face_embedding("WIN_20240402_19_12_39_Pro.jpg")
upload_sample_face_embedding("WIN_20240402_19_12_42_Pro.jpg")
upload_sample_face_embedding("WIN_20240402_19_12_45_Pro.jpg")
upload_sample_face_embedding("WIN_20240402_19_12_51_Pro.jpg")
upload_sample_face_embedding("WIN_20240402_19_12_56_Pro.jpg")
















# Function to check if a face matches the sample face
def is_face_matching(embedding):
    # Load all sample face embeddings from Firebase
    sample_faces = firebase_utility.get_all_data("sample_faces")

    if sample_faces is not None:
        # Compare the detected face embedding with each sample face embedding
        for data in sample_faces.values():
            sample_embedding = data.get("embedding")
            if sample_embedding is not None:
                # Calculate the Euclidean distance between embeddings
                distance = np.linalg.norm(np.array(embedding) - np.array(sample_embedding))
                # Set a threshold for matching
                print(distance)
                if distance < 16:
                    return True

    # If no matching embedding is found, return False
    return False


# Main loop to capture video frames
while True:
    # Read a frame from the webcam
    check, frame = video.read()
    if frame is not None:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            # Compute absolute difference between the current frame and the previous frame
            frame_diff = cv2.absdiff(gray, prev_frame)

            # Threshold the frame difference to identify significant changes
            _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

            # Count the number of non-zero pixels in the thresholded image
            motion_pixels = cv2.countNonZero(thresh)

            # If the number of motion pixels exceeds a threshold, motion is detected
            if motion_pixels > 1000:  # Adjust this threshold as needed
                print("Motion Detected")

            # Store the current frame for comparison in the next iteration
        prev_frame = gray.copy()

        # Display the frame
        cv2.imshow("Motion Detection", frame)

        # Detect faces in the frame
        faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # Loop through the detected faces
        for x, y, w, h in faces:

            box_area = w * h


            if box_area < 5000:  # Small box => person far from the camera
                distance_estimation = "Far (5-10 feet)"
                print(distance_estimation)
            elif box_area < 15000:  # Medium box => person at a decent distance
                distance_estimation = "Decent distance"
                print(distance_estimation)
            else:  # Large box => person close to the camera
                distance_estimation = "Close"
                send_message("someone is on your property")
                print(distance_estimation)



            # Crop the face region
            face_img = frame[y:y+h, x:x+w]
            # Calculate embeddings for the face
            embedding = calculate_embeddings(face_img)
            print(embedding)
            # Check if the face matches the sample face
            if is_face_matching(embedding):
                # Outline the face in green if it matches the sample face
                color = (0, 255, 0)  # Green color
            else:
                # Outline the face in red if it doesn't match the sample face
                send_message("an unknown person is on your property")
                color = (0, 0, 255)  # Red color
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Display the frame
        cv2.imshow("Face Detection", frame)
        # Check for key press (q to quit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close all windows
video.release()
cv2.destroyAllWindows()

import cv2
from imgbeddings import imgbeddings
import firebaseUtility
import numpy as np
from PIL import Image
import requests
import time
import os
import pygame


pygame.mixer.init()



#  play the alarm
def play_alarm_sound():
    pygame.mixer.music.load("alarm.mp3")
    pygame.mixer.music.play()

# Haar cascade classifier for face detection
alg = "faceData.xml"
haar_cascade = cv2.CascadeClassifier(alg)

#cv2 webcam
video = cv2.VideoCapture(0)


# set frame
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# frame rate
video.set(cv2.CAP_PROP_FPS, 30)

prev_frame = None


# initialize firebase
cred_path = "homedefender-a7497-firebase-adminsdk-p05pu-d6d28cad6b.json"
db_url = "https://homedefender-a7497-default-rtdb.firebaseio.com/"
firebase_utility = firebaseUtility.FirebaseUtility(cred_path, db_url)



def send_message(message):
    resp = requests.post('https://textbelt.com/text', {
        'phone': '7706801965',
        'message': message,
        'key': 'textbelt',
    })
    print(resp.json())

def calculate_embeddings(face_img):
    # change the numpy array to a pil image
    pil_image = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    # calc imgbedding
    imgbed = imgbeddings()
    embedding = imgbed.to_embeddings(pil_image)
    return embedding[0].tolist()


def upload_sample_face_embedding(sample_img_path):
    # read the sample face image
    sample_img = cv2.imread(sample_img_path)
    # calc embeddings for the sample face image
    sample_embedding = calculate_embeddings(sample_img)
    # push the data into the "sample_faces" table in Firebase
    firebase_utility.put_data("sample_faces", {"embedding": sample_embedding})

# example usage of uploading face pic
'''

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

'''


# check if a face matches the sample face
def is_face_matching(embedding):
    # get all sample face embeddings from firebase
    sample_faces = firebase_utility.get_all_data("sample_faces")

    if sample_faces is not None:
        # compare frame embedding with each sample face embedding
        for data in sample_faces.values():
            sample_embedding = data.get("embedding")

            if sample_embedding is not None:
                # euclidean distance between embeddings
                dist = np.linalg.norm(np.array(embedding) - np.array(sample_embedding))
                # threshold for matching
                print("distance:" ,dist)
                if dist < 16:
                    return True

    # no matching embedding is found
    return False

total_frames = 0
green_count = 0
red_count = 0
frame_on = False
visited = False
a = 0
b = 0
c = 0
d = 0
temp = (0, 0, 255)
# loop to capture video frames
while True:
    time_sec = time.time()
    # check frame from the webcam
    check, frame = video.read()
    if frame is not None:
        # change the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            # absolute difference between the current frame and the previous frame
            frame_diff = cv2.absdiff(gray, prev_frame)

            # threshold the frame difference
            _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

            # calc the number of non-zero pixels in the thresholded image
            motion_pixels = cv2.countNonZero(thresh)

            # the number of motion pixels > a threshold motion detected
            if motion_pixels > 1000:
                print("Motion Detected")

            # store the current frame for comparison in the next iteration
        prev_frame = gray.copy()

        # detect faces in the frame
        #print ("time: ", time_sec)
        if int(time_sec) % 10 == 0:
            frame_with_detections = frame.copy()

            faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for x, y, w, h in faces:
                total_frames += 1

                box_area = w * h

                if box_area < 5000:  # far from the camera
                    distance_estimation = "far (5-10 feet)"
                    print(distance_estimation)
                elif box_area < 15000:  # decent distance
                    distance_estimation = "decent distance"
                    print(distance_estimation)
                else:  # close to the camera
                    distance_estimation = "close"
                    print(distance_estimation)
                    if (int(time_sec) % 120 == 0):
                       send_message("someone is on your property")

                # crop the face
                face_img = frame[y:y+h, x:x+w]
                color = (0, 0, 255)


                embedding = calculate_embeddings(face_img)
                #print(embedding)

                if is_face_matching(embedding):
                    print("face matching true" )
                    # face in green if it matches the sample face
                    color = (0, 255, 0)  # Green color
                    # make a blank image with white background to display text
                    text_image = np.zeros((100, 300, 3), dtype=np.uint8)
                    text_image.fill(255)

                    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
                    cv2.putText(text_image, "Face Recognized", (10, 50), font, 1, color, 2)

                    cv2.imshow("Face Detection Status", text_image)

                    cv2.waitKey(3000)
                    cv2.destroyWindow("Face Detection Status")

                    green_count += 1

                    temp = (0, 255, 0)
                    green_count += 1
                    frame_on = True
                    visited = True
                    a = x
                    b = y
                    c = w
                    d = h

                else:
                    # outline the face in red if it doesnt match the sample face
                    print("face matching false")
                    #send_message("an unknown person is on your property")

                    color = (0, 0, 255)  # Red color
                    # make a blank image with white background to display text
                    text_image = np.zeros((100, 300, 3), dtype=np.uint8)
                    text_image.fill(255)  # Fill with white color

                    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
                    cv2.putText(text_image, "Face Not Recognized", (10, 50), font, 1, color, 2)

                    # show the text image in a separate window
                    cv2.imshow("Face Detection Status", text_image)
                    play_alarm_sound()

                    cv2.waitKey(3000)
                    cv2.destroyWindow("Face Detection Status")

                    red_count += 1

                    temp = (0, 0, 255)  # Red color
                    red_count += 1
                    #cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    frame_on = False
                    visited = True
                    a = x
                    b = y
                    c = w
                    d = h


        # show the frame
        if int(time_sec) % 3 == 0:
            if frame_on and visited:
                cv2.rectangle(frame, (a, b), (a + c, b + d), temp, 2)
            elif not frame_on and visited:
                cv2.rectangle(frame, (a, b), (a + c, b + d), temp, 2)

        cv2.imshow("Face Detection", frame)

        if total_frames >= 60:
            break
        #  (q to quit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


total_frames= green_count + red_count

# calc proportions
correct = green_count / total_frames
incorrect = red_count / total_frames

print("Total frames:", total_frames)
print("Green count (correctly identified):", green_count)
print("Red count (not recognized):", red_count)
print("Proportion of correct identifications:", correct)
print("Proportion of incorrect identifications:", incorrect)

# close all windows
video.release()
cv2.destroyAllWindows()

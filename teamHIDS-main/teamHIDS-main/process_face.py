from deepface import DeepFace
from retinaface import RetinaFace
import cv2
import time
import mediapipe as mp
import winsound
import numpy as np
import copy
from person_extract import extract_persons

#vid = cv2.VideoCapture(0)
frame_time = time.time()
frame_time_for_pose = time.time()
no_of_frames = 0
face_count = 0
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5,model_complexity=0)
pose_array = []
pose_array_temp = []
criminal_match = {}

vid = cv2.VideoCapture('people-walking.mp4')
#vid = cv2.VideoCapture(0)

while True:
    
    ret, frame = vid.read()
    persons, image_by_yolo = extract_persons(frame)
    cv2.imshow("person detection1", image_by_yolo)        

    print(persons)
    try:
        pose_results = pose.process(frame)
        

        try:
            start_time = time.time()
            facial_data = RetinaFace.detect_faces(frame)
            print(time.time() - start_time)
        except:
            pass

        for face in facial_data:

            x = facial_data[face]['facial_area'][0]
            y = facial_data[face]['facial_area'][1]
            h = facial_data[face]['facial_area'][2] 
            w = facial_data[face]['facial_area'][3] 

            frame = cv2.rectangle(frame,(x,y),(h,w),(255, 0, 0),thickness = 2)
            cropped_image = frame[y:w,x:h] 
            #cv2.imshow("output",cropped_image) 
            cv2.imwrite("D:\\mini project2\\vid to frame\\frame%d.jpg" % face_count, cropped_image)
            face_count+=1
            try:
                criminal_match = DeepFace.verify("frame16.jpg",cropped_image,enforce_detection=False)
            except:
                criminal_match['verified'] = False
            if criminal_match['distance'] <= 0.3 and criminal_match['verified']:
                print("criminal match = ",criminal_match)
                cropped_image = cv2.resize(cropped_image, (300,300), interpolation = cv2.INTER_AREA)

                cv2.imshow("criminal",cropped_image)
                cv2.imwrite("D:\\mini project2\\criminals\\frame%d.jpg" % face_count, cropped_image)
            else:
                print("criminal match = ",criminal_match)

            
        for person in persons:
            #cv2.imshow("person detection", person) 
            print(person.shape[0])  
                 
            if person.shape[0] > 0 and person.shape[1] > 0 and person.shape[2] > 0:
                pose_results = pose.process(person)
            mp_drawing.draw_landmarks(person, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            try:
                cv2.imshow("person detection", person)
            except:
                pass        


        temp_time_pose = time.time() - frame_time_for_pose
        pose_array_temp.append(pose_results.pose_landmarks)

        if temp_time_pose >= 1:
            frame_time_for_pose = 0
            pose_array.append(pose_array_temp)

        
    except:
        print("can't find faces")
        

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    temp_time = time.time() - frame_time
    no_of_frames += 1
    if temp_time >= 1:
        print("frames processed : ",no_of_frames)
        no_of_frames = 0
        frame_time = time.time()
vid.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import face_recognition
import os
import csv
from datetime import datetime

path = 'imagesbasic'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'a+') as f:
        now = datetime.now()
        date_string = now.strftime('%d-%m-%y')
        time_string = now.strftime('%H:%M:%S')

        f.seek(0, os.SEEK_END)  # Move the file pointer to the end of the file
        if f.tell() == 0:  # Check if the file is empty
            f.write('NAME,DATE,TIME\n')  # Write the row headings
        f.write(f'{name},{date_string},{time_string}\n')
        print(f'Attendance marked for {name} at {time_string} on {date_string}')


encodeListKnown = findEncodings(images)
print('Encoding Complete')

display_width = 640
display_height = 480

cap = cv2.VideoCapture(0)
cap.set(3, display_width)  # Set the width
cap.set(4, display_height)  # Set the height

attendance_marked = [False] * len(classNames)  # Flag to track attendance marking for each person

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)

        for i in range(len(matches)):
            if matches[i]:
                name = classNames[i].upper()
                print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                if not attendance_marked[i]:
                    markAttendance(name)
                    attendance_marked[i] = True

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):  # Press 'q' or 'Q' to quit
        break

cap.release()
cv2.destroyAllWindows()

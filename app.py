import cv2
import face_recognition
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
import time

# Initialize Firebase (replace 'your-firebase-credentials.json' with your own Firebase credentials file)
cred = credentials.Certificate('C:/Users/Neeraj/Desktop/doiT/doit-a8d6c-firebase-adminsdk-je7vm-a8f63ddfa8.json')
firebase_admin.initialize_app(cred, {'databaseURL': 'https://doit-a8d6c-default-rtdb.firebaseio.com/'})

# Load the images of the students
students_data = [
    {"name": "Himanshu", "image_path": "C:/Users/Neeraj/Desktop/doiT/Hims.jpg"},
    {"name": "Vanshika", "image_path": "C:/Users/Neeraj/Desktop/doiT/Vans.jpg"},
    {"name": "Gruhit", "image_path": "C:/Users/Neeraj/Desktop/doiT/Gruhit.jpg"},
]

known_face_encodings = []
known_face_names = []

# Encode the face images
for student in students_data:
    student_image = face_recognition.load_image_file(student["image_path"])
    student_face_encoding = face_recognition.face_encodings(student_image)[0]

    known_face_encodings.append(student_face_encoding)
    known_face_names.append(student["name"])


# Initialize the webcam
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

fps = 0
processing_time = 0

while True:
    start_time = time.time()
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the frame to RGB format
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    recognized_students = []

    for face_encoding in face_encodings:
        # Compare the face encoding with the known face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

        name = "Unknown"
        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

        recognized_students.append(name)

        # Update Firebase database if a student is recognized
        if name != "Unknown":
            date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            db.reference('attendance').child(name).set({
                'date_time': date_time,
            })

    for (top, right, bottom, left), name in zip(face_locations, recognized_students):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Calculate FPS and processing time
    end_time = time.time()
    processing_time = end_time - start_time
    fps = 1 / processing_time

    # Display FPS and processing time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, f"Processing Time: {processing_time:.2f} sec", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Check if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
video_capture.release()
cv2.destroyAllWindows()

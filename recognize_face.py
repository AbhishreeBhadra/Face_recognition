import cv2
from keras.models import load_model  # type: ignore
import numpy as np
from PIL import Image  # type: ignore

# Load the model and Haarcascade
model = load_model('facefeatures_new_model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Class labels
class_labels = ['Abhishree', 'Adrika', 'Ahana', 'Aishwairya']  # Ensure this matches the number of model outputs

def face_extractor(img):
    """Extract face from an image using Haar Cascade."""
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    if len(faces) == 0:
        return None
    for (x, y, w, h) in faces:
        return img[y:y+h, x:x+w]

# Start webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert frame to grayscale for Haar Cascade
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_extractor(gray_frame)
    if face is not None:
        face = cv2.resize(face, (224, 224))
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)  # Convert back to RGB for the model
        face = np.expand_dims(face, axis=0) / 255.0

        prediction = model.predict(face)
        class_idx = np.argmax(prediction)
        confidence = prediction[0][class_idx]

        # Handle index out of range and display appropriate name
        if class_idx < len(class_labels) and confidence > 0.5:
            name = class_labels[class_idx]
        else:
            name = "Unknown"

        cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No Face Detected", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

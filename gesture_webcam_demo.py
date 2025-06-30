import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("gesture_cnn_model.h5")

# Define parameters
IMG_SIZE = 100
gesture_labels = {
    0: 'Palm',
    1: 'L',
    2: 'Fist',
    3: 'Fist_moved',
    4: 'Thumb',
    5: 'Index',
    6: 'Ok',
    7: 'Palm_moved',
    8: 'C',
    9: 'Down'
}

# Start webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip frame to mirror user
    frame = cv2.flip(frame, 1)

    # Draw region of interest
    x1, y1, x2, y2 = 100, 100, 300, 300
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    roi = frame[y1:y2, x1:x2]

    # Preprocess ROI
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype("float32") / 255.0
    input_tensor = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # Predict
    preds = model.predict(input_tensor)
    class_id = np.argmax(preds)
    class_name = gesture_labels[class_id]

    # Display prediction
    cv2.putText(frame, f"Gesture: {class_name}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

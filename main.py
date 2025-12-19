import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained AI model (replace with your model path)
model = load_model("waste_classifier_model.h5")

# Waste class labels - update based on your model
labels = ["Plastic", "Paper", "Metal", "Glass", "Organic", "General"]

# Open webcam
cam = cv2.VideoCapture(0)

print("📌 Smart Waste Sorting System Activated")
print("Press 'q' to quit")

while True:
    ret, frame = cam.read()
    
    if not ret:
        print("Camera not detected!")
        break
    
    # Preprocess image for model
    img = cv2.resize(frame, (150, 150))  # model input size
    img = img / 255.0
    img = np.reshape(img, (1, 150, 150, 3))

    # Predict
    prediction = model.predict(img)
    category = labels[np.argmax(prediction)]

    # Display output on screen
    cv2.putText(frame, f"Detected: {category}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Smart Waste Sorting System", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

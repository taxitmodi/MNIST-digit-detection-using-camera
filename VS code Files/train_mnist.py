# import cv2
# import numpy as np
# import tensorflow as tf
# from collections import deque

# # Load trained MNIST model
# model = tf.keras.models.load_model("mnist_custom_model.keras")

# # Start webcam
# cap = cv2.VideoCapture(0)

# # Fixed ROI box position
# BOX_X, BOX_Y, BOX_SIZE = 200, 100, 200  # (X, Y, Width=Height)

# # Store last 10 predictions for smoothing
# prediction_buffer = deque(maxlen=10)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Extract ROI
#     roi = gray[BOX_Y:BOX_Y + BOX_SIZE, BOX_X:BOX_X + BOX_SIZE]

#     # Apply preprocessing
#     roi = cv2.GaussianBlur(roi, (7, 7), 0)  # Larger kernel to smooth more
#     roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  # Adaptive thresholding

#     # Apply morphological operations to clean the digit
#     kernel = np.ones((3, 3), np.uint8)
#     roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)  # Removes small noise

#     # Resize and normalize
#     roi = cv2.resize(roi, (28, 28))
#     roi = roi / 255.0  # Normalize pixel values
#     roi = roi.reshape(1, 28, 28, 1)  # Reshape for CNN model

#     # Predict digit
#     prediction = model.predict(roi)
#     digit = np.argmax(prediction)
#     confidence = np.max(prediction) * 100  # Convert to percentage

#     # Store prediction in buffer and take the most frequent value
#     prediction_buffer.append(digit)
#     stable_digit = max(set(prediction_buffer), key=prediction_buffer.count)

#     # Draw ROI box
#     cv2.rectangle(frame, (BOX_X, BOX_Y), (BOX_X + BOX_SIZE, BOX_Y + BOX_SIZE), (0, 255, 0), 2)

#     # Display stabilized prediction
#     text = f"Digit: {stable_digit} ({confidence:.2f}%)"
#     cv2.putText(frame, text, (BOX_X - 10, BOX_Y - 10), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     # Show output
#     cv2.imshow("Digit Recognition", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()








# import cv2
# import numpy as np
# import tensorflow as tf
# from collections import deque

# # Load trained MNIST model
# model = tf.keras.models.load_model("mnist_custom_model.keras")

# # Start webcam
# cap = cv2.VideoCapture(0)

# # Fixed ROI box position
# BOX_X, BOX_Y, BOX_SIZE = 200, 100, 200  # (X, Y, Width=Height)

# # Store last 10 predictions for smoothing
# prediction_buffer = deque(maxlen=10)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Extract ROI
#     roi = gray[BOX_Y:BOX_Y + BOX_SIZE, BOX_X:BOX_X + BOX_SIZE]

#     # Apply preprocessing
#     roi = cv2.GaussianBlur(roi, (7, 7), 0)  # Larger kernel to smooth more
#     roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  # Adaptive thresholding

#     # Apply morphological operations to clean the digit
#     kernel = np.ones((3, 3), np.uint8)
#     roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)  # Removes small noise

#     # **Digit presence check** (Count white pixels)
#     non_zero_pixels = cv2.countNonZero(roi)
#     digit_present = non_zero_pixels > 100  # Adjust threshold if needed

#     # Draw ROI box
#     box_color = (0, 255, 0) if digit_present else (0, 0, 255)  # Green if digit detected, Red otherwise
#     cv2.rectangle(frame, (BOX_X, BOX_Y), (BOX_X + BOX_SIZE, BOX_Y + BOX_SIZE), box_color, 2)

#     if digit_present:
#         # Resize and normalize
#         roi = cv2.resize(roi, (28, 28))
#         roi = roi / 255.0  # Normalize pixel values
#         roi = roi.reshape(1, 28, 28, 1)  # Reshape for CNN model

#         # Predict digit
#         prediction = model.predict(roi)
#         digit = np.argmax(prediction)
#         confidence = np.max(prediction) * 100  # Convert to percentage

#         # Store prediction in buffer and take the most frequent value
#         prediction_buffer.append(digit)
#         stable_digit = max(set(prediction_buffer), key=prediction_buffer.count)

#         # Display stabilized prediction
#         text = f"Digit: {stable_digit} ({confidence:.2f}%)"
#         cv2.putText(frame, text, (BOX_X - 10, BOX_Y - 10), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     else:
#         # If no digit is detected, do not display anything
#         prediction_buffer.clear()  # Reset buffer to avoid incorrect predictions

#     # Show output
#     cv2.imshow("Digit Recognition", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# Load trained MNIST model
model = tf.keras.models.load_model("mnist_custom_model.keras")

# Start webcam
cap = cv2.VideoCapture(0)

# Fixed ROI box position
BOX_X, BOX_Y, BOX_SIZE = 200, 100, 200  # (X, Y, Width=Height)

# Store last 10 predictions for smoothing
prediction_buffer = deque(maxlen=10)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Extract ROI
    roi = gray[BOX_Y:BOX_Y + BOX_SIZE, BOX_X:BOX_X + BOX_SIZE]

    # Apply preprocessing
    roi = cv2.GaussianBlur(roi, (7, 7), 0)
    roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply morphological operations
    kernel = np.ones((3, 3), np.uint8)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)

    # **Detect contours (shapes) in the ROI**
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if a valid digit-like contour is present
    digit_present = False
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(cnt)

        # A valid digit should have a reasonable size and aspect ratio
        if 30 < area < 15000 and 0.15 < aspect_ratio < 1.5:  
            digit_present = True
            break

    # Draw ROI box (Red = No digit, Green = Digit detected)
    box_color = (0, 255, 0) if digit_present else (0, 0, 255)
    cv2.rectangle(frame, (BOX_X, BOX_Y), (BOX_X + BOX_SIZE, BOX_Y + BOX_SIZE), box_color, 2)

    if digit_present:
        # Resize and normalize
        roi = cv2.resize(roi, (28, 28))
        roi = roi / 255.0
        roi = roi.reshape(1, 28, 28, 1)

        # Predict digit
        prediction = model.predict(roi)
        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # Store prediction in buffer and stabilize output
        prediction_buffer.append(digit)
        stable_digit = max(set(prediction_buffer), key=prediction_buffer.count)
        
        print(f"Predicted Digit: {stable_digit}, Confidence: {confidence:.2f}%")

        # Display stabilized prediction
        text = f"Digit: {stable_digit} ({confidence:.2f}%)"
        cv2.putText(frame, text, (BOX_X - 10, BOX_Y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        # Clear prediction buffer when no digit is present
        prediction_buffer.clear()
        
        # Print message when no digit is detected
        print("No digit present")

    # Show output
    cv2.imshow("Digit Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

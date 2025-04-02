import cv2
import numpy as np
import tensorflow as tf

# Load trained MNIST model
model = tf.keras.models.load_model("mnist_custom_model.keras")

# Create a blank whiteboard
canvas = np.ones((400, 400), dtype="uint8") * 255
drawing = False  # Flag to track drawing state

# Mouse callback function
def draw(event, x, y, flags, param):
    global drawing

    if event == cv2.EVENT_LBUTTONDOWN:  # When left mouse button is pressed
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:  # When moving the mouse
        if drawing:
            cv2.circle(canvas, (x, y), 10, (0, 0, 0), -1)  # Draw black circles
    elif event == cv2.EVENT_LBUTTONUP:  # When mouse button is released
        drawing = False

# Set up OpenCV window
cv2.namedWindow("Draw a Digit")
cv2.setMouseCallback("Draw a Digit", draw)

while True:
    # Show the drawing canvas
    cv2.imshow("Draw a Digit", canvas)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):  # Clear the canvas
        canvas[:] = 255
    elif key == ord("p"):  # Predict digit
        # Preprocess image for model
        img = cv2.resize(canvas, (28, 28))  # Resize to match MNIST model input
        img = cv2.bitwise_not(img)  # Invert colors (white -> black, black -> white)
        img = img / 255.0  # Normalize
        img = img.reshape(1, 28, 28, 1)  # Reshape for CNN

        # Predict digit
        prediction = model.predict(img)
        digit = np.argmax(prediction)

        # Display prediction
        print(f"Predicted Digit: {digit}")
        cv2.putText(canvas, f"Digit: {digit}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif key == ord("q"):  # Quit the program
        break

cv2.destroyAllWindows()

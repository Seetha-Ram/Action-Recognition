import streamlit as st
import cv2
import numpy as np
from io import BytesIO

# Define the actions globally
actions = ["Sitting", "Running", "Walking"]

# Function to recognize actions from video frames
def recognize_action(frame):
    # Simulate action recognition with random scores
    scores = np.random.rand(len(actions))
    recognized_action = actions[np.argmax(scores)]

    # Return the recognized action and corresponding score
    return recognized_action, scores

def main():
    st.title("Action Recognition Streamlit App")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

    if uploaded_file is not None:
        # Display the uploaded video
        st.video(uploaded_file)

        # Convert the file content to bytes
        file_bytes = uploaded_file.read()

        # Convert the bytes to a numpy array
        nparr = np.frombuffer(file_bytes, np.uint8)

        # Decode the numpy array into an OpenCV image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process each frame in the video
        while True:
            # Perform action recognition on the frame
            recognized_action, scores = recognize_action(frame)

            # Display the recognized action with scores
            st.write(f"Recognized Action: {recognized_action}")
            st.write("Action Scores:")
            for action, score in zip(actions, scores):
                st.write(f"{action}: {score:.2f}")

            # Break the loop if no more frames
            if cv2.waitKey(30) & 0xFF == 27:
                break

        # Release resources
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

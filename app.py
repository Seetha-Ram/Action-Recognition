import streamlit as st
import cv2
import numpy as np
from io import BytesIO

# Define the actions globally
actions = ["Sitting", "Running", "Walking", "Jumping", "Climbing", "Taking a gun", "Shooting", "Falling", "Falling from bed"]

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

        # Create a dynamic table to display action scores
        scores_table = st.table([[action, 0.0] for action in actions])

        # Process each frame in the video
        while True:
            # Perform action recognition on the frame
            recognized_action, scores = recognize_action(frame)

            # Update the action scores in the dynamic table
            new_values = [[action, score] for action, score in zip(actions, scores)]
            scores_table.table(new_values)

            # Display the recognized action
            st.write(f"Recognized Action: {recognized_action}")

            # Break the loop if no more frames
            if cv2.waitKey(30) & 0xFF == 27:
                break

        # Release resources
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

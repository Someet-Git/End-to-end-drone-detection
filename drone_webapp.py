import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import numpy as np

# Load YOLO model
model = YOLO('my_model.pt')

# Streamlit app title
st.title("YOLOv8 Image and Video Prediction Web App")

# Option to select image or video
option = st.selectbox('What would you like to upload?', ('Image', 'Video'))

# Handle Image Upload
if option == 'Image':
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the uploaded file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        
        # Display uploaded image
        st.image(img, caption='Uploaded Image', use_column_width=True)
        st.write("Running prediction...")

        # Predict using YOLO model
        results = model.predict(source=img, save=False)

        # Plot the results on the image
        predicted_img = results[0].plot()

        # Convert the predicted image (from BGR to RGB) for Streamlit display
        predicted_img = cv2.cvtColor(predicted_img, cv2.COLOR_BGR2RGB)

        # Display the prediction results
        st.image(predicted_img, caption='Prediction', use_column_width=True)

        # Option to save the image locally
        if st.button('Save Prediction'):
            prediction_path = os.path.join("predicted_output_image.jpg")
            cv2.imwrite(prediction_path, cv2.cvtColor(predicted_img, cv2.COLOR_RGB2BGR))
            st.write(f"Prediction saved at {prediction_path}")

# Handle Video Upload
elif option == 'Video':
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        # Save the uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        # Read the video using OpenCV
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Set up output video path
        output_video_path = 'output_video.avi'
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

        st.write("Running video prediction...")

        # Process each frame of the video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO model on the frame
            results = model.predict(source=frame, save=False)

            # Get the frame with predictions plotted
            predicted_frame = results[0].plot()

            # Write the predicted frame to the output video
            out.write(predicted_frame)

        cap.release()
        out.release()

        st.write("Video processing complete!")

        # Display download button for the processed video
        with open(output_video_path, "rb") as f:
            st.download_button("Download Processed Video", f, file_name="output_video.avi")

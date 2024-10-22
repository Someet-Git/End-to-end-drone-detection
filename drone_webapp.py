import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from collections import defaultdict

# Page configuration
st.set_page_config(
    page_title="YOLOv8 Object Detection",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #2E4053;
        font-size: 3rem !important;
    }
    .section-header {
        color: #34495E;
        font-size: 1.8rem;
        padding-top: 2rem;
    }
    .info-box {
        background-color: #F8F9F9;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Main title with emoji
st.title("🔍 YOLOv8 Object Detection Hub")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Home", "Object Detection", "About YOLO"])

def plot_confidence_distribution(confidence_scores):
    fig = px.histogram(
        confidence_scores,
        nbins=20,
        title="Detection Confidence Distribution",
        labels={'value': 'Confidence Score', 'count': 'Number of Detections'},
        color_discrete_sequence=['#3498db']
    )
    fig.update_layout(showlegend=False)
    return fig

def plot_class_distribution(detected_classes, class_names):
    classes, counts = np.unique(detected_classes, return_counts=True)
    class_labels = [class_names.get(int(cls), f"Class {int(cls)}") for cls in classes]
    
    fig = px.bar(
        x=class_labels,
        y=counts,
        title="Detected Objects Distribution",
        labels={'x': 'Class', 'y': 'Count'},
        color_discrete_sequence=['#2ecc71']
    )
    return fig

def create_detection_heatmap(boxes, image_shape):
    heatmap = np.zeros(image_shape[:2])
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        heatmap[y1:y2, x1:x2] += 1
    
    fig = px.imshow(
        heatmap,
        title="Detection Density Heatmap",
        color_continuous_scale="Viridis"
    )
    return fig

if page == "Home":
    st.markdown("## Welcome to Object Detection with YOLOv8! 🚀")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### What is YOLO?
        YOLO (You Only Look Once) is a state-of-the-art object detection system that:
        - Processes images in real-time
        - Identifies multiple objects in a single frame
        - Provides accurate bounding boxes and classifications
        """)
    
    with col2:
        # Using a more technical architectural diagram
        st.image("https://yolov8.org/wp-content/uploads/2024/01/What-is-YOLOv8-1.webp",
                 caption="YOLO Architecture")
    
    # Interactive Architecture Explanation
    st.markdown("### 🔄 YOLO Architecture Components")
    
    tabs = st.tabs(["Backbone", "Neck", "Head"])
    
    with tabs[0]:
        st.markdown("""
        ### Backbone Network (Feature Extraction)
        - CSPDarknet architecture (modified)
        - Extracts hierarchical features
        - Multiple scale processing
        """)
        
    with tabs[1]:
        st.markdown("""
        ### Neck (Feature Fusion)
        - Path Aggregation Network (PAN)
        - Feature Pyramid Network (FPN)
        - Multi-scale feature combination
        """)
        
    with tabs[2]:
        st.markdown("""
        ### Detection Head
        - Dense prediction
        - Multiple anchors
        - Class prediction
        - Bounding box regression
        """)

elif page == "Object Detection":
    st.markdown("## 🎯 Object Detection Studio")
    
    # Model loading with status
    with st.spinner("Loading YOLO model..."):
        model = YOLO('my_model.pt')
    st.success("Model loaded successfully!")
    
    # Option to select input type
    option = st.selectbox(
        'Select Input Type',
        ('Image', 'Video'),
        help="Choose whether to upload an image or video for object detection"
    )
    
    if option == 'Image':
        st.markdown("### 📸 Image Detection")
        uploaded_file = st.file_uploader(
            "Upload an image...",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            # Create columns for before/after comparison
            col1, col2 = st.columns(2)
            
            # Convert and display uploaded image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            with col1:
                st.markdown("#### Original Image")
                st.image(img, use_column_width=True)
            
            # Progress bar for prediction
            with st.spinner("🔍 Detecting objects..."):
                results = model.predict(source=img, save=False)
                predicted_img = results[0].plot()
                predicted_img = cv2.cvtColor(predicted_img, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.markdown("#### Detection Result")
                st.image(predicted_img, use_column_width=True)
            
            # Detailed Metrics Section
            st.markdown("### 📊 Detection Analytics")
            
            # Get detection data
            boxes = results[0].boxes
            detected_classes = boxes.cls.cpu().numpy()
            confidence_scores = boxes.conf.cpu().numpy()
            bounding_boxes = boxes.xyxy.cpu().numpy()
            
            # Create metrics columns
            metric_cols = st.columns(3)
            
            with metric_cols[0]:
                st.markdown("#### 📈 Overall Statistics")
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #4CAF50;">Total Detections: {len(detected_classes)}</h4>
                    <h4 style="color: #4CAF50;">Average Confidence: {np.mean(confidence_scores)*100:.2f}%
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence Distribution
            with metric_cols[1]:
                st.markdown("#### Confidence Distribution")
                conf_fig = plot_confidence_distribution(confidence_scores)
                st.plotly_chart(conf_fig, use_container_width=True)
            
            # Class Distribution
            with metric_cols[2]:
                st.markdown("#### Class Distribution")
                class_fig = plot_class_distribution(detected_classes, model.names)
                st.plotly_chart(class_fig, use_container_width=True)
            
            # Detection Heatmap
            st.markdown("#### Detection Density Heatmap")
            heatmap_fig = create_detection_heatmap(bounding_boxes, img.shape)
            st.plotly_chart(heatmap_fig, use_container_width=True)
            
            # Detailed Analysis Expander
            with st.expander("🔍 Detailed Analysis"):
                # Per-class metrics
                st.markdown("#### Per-Class Metrics")
                for cls in np.unique(detected_classes):
                    class_mask = detected_classes == cls
                    class_confidences = confidence_scores[class_mask]
                    
                    st.markdown(f"""
                    ##### Class {model.names.get(int(cls), f'Class {int(cls)}')}
                    - Count: {np.sum(class_mask)}
                    - Average Confidence: {np.mean(class_confidences)*100:.2f}%
                    - Min Confidence: {np.min(class_confidences)*100:.2f}%
                    - Max Confidence: {np.max(class_confidences)*100:.2f}%
                    """)
            
            # Save option with better UI
            if st.button('💾 Save Detection Result'):
                prediction_path = os.path.join("predicted_output_image.jpg")
                cv2.imwrite(prediction_path, cv2.cvtColor(predicted_img, cv2.COLOR_RGB2BGR))
                st.success(f"✅ Detection result saved as 'predicted_output_image.jpg'")
    
    else:  # Video option
        st.markdown("### 🎥 Video Detection")
        uploaded_video = st.file_uploader(
            "Upload a video...",
            type=["mp4", "avi", "mov"],
            help="Supported formats: MP4, AVI, MOV"
        )
        
        if uploaded_video is not None:
            # Save uploaded video
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            video_path = tfile.name
            
            # Video processing with progress bar
            with st.spinner("🎬 Processing video..."):
                cap = cv2.VideoCapture(video_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                output_video_path = 'output_video.avi'
                out = cv2.VideoWriter(output_video_path, 
                                    cv2.VideoWriter_fourcc(*'XVID'),
                                    fps, (width, height))
                
                # Initialize metrics tracking
                frame_metrics = defaultdict(list)
                
                # Progress bar
                progress_bar = st.progress(0)
                frame_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    results = model.predict(source=frame, save=False)
                    predicted_frame = results[0].plot()
                    out.write(predicted_frame)
                    
                    # Collect metrics
                    detected_classes = results[0].boxes.cls.cpu().numpy()
                    confidence_scores = results[0].boxes.conf.cpu().numpy()
                    
                    frame_metrics['detections'].append(len(detected_classes))
                    frame_metrics['confidence'].append(np.mean(confidence_scores) if len(confidence_scores) > 0 else 0)
                    
                    # Update progress
                    frame_count += 1
                    progress_bar.progress(frame_count / total_frames)
                
                cap.release()
                out.release()
            
            st.success("✅ Video processing complete!")
            
            # Display video metrics
            st.markdown("### 📊 Video Analysis Metrics")
            
            metric_cols = st.columns(2)
            
            with metric_cols[0]:
                # Plot detections over time
                fig_detections = px.line(
                    x=list(range(len(frame_metrics['detections']))),
                    y=frame_metrics['detections'],
                    title="Detections per Frame",
                    labels={'x': 'Frame', 'y': 'Number of Detections'}
                )
                st.plotly_chart(fig_detections)
            
            with metric_cols[1]:
                # Plot confidence over time
                fig_confidence = px.line(
                    x=list(range(len(frame_metrics['confidence']))),
                    y=frame_metrics['confidence'],
                    title="Average Confidence per Frame",
                    labels={'x': 'Frame', 'y': 'Confidence'}
                )
                st.plotly_chart(fig_confidence)
            
            # Download button with better styling
            with open(output_video_path, "rb") as f:
                st.download_button(
                    label="📥 Download Processed Video",
                    data=f,
                    file_name="detected_output.avi",
                    mime="video/x-msvideo"
                )

else:  # About YOLO page
    st.markdown("## 📚 Understanding YOLO")
    
    # YOLO explanation with detailed architecture
    st.markdown("""
    ### YOLO Architecture Deep Dive
    YOLO's architecture consists of three main components:
    """)
    st.image("yolo_flow.png",
             caption="YOLO Flow Diagram")

    arch_cols = st.columns(3)
    
    with arch_cols[0]:
        st.markdown("""
        #### 🔄 Backbone
        - CSPDarknet
        - Residual connections
        - Feature pyramids
        - Spatial pyramid pooling
        """)
    
    with arch_cols[1]:
        st.markdown("""
        #### 🔀 Neck
        - FPN (Feature Pyramid Network)
        - PAN (Path Aggregation Network)
        - Multi-scale feature fusion
        - Information flow optimization
        """)
    
    with arch_cols[2]:
        st.markdown("""
        #### 🎯 Head
        - Dense prediction layers
        - Multi-scale detection
        - Classification branch
        - Regression branch
        """)
    
    # Training process explanation
    st.markdown("""
    ### 🎯 Training Process
    1. **Data Preparation**
       - Dataset curation and cleaning
       - Annotation in YOLO format
       - Augmentation strategies
       - Train/val/test split
    
    2. **Fine-tuning**
       - Transfer learning setup
       - Learning rate scheduling
       - Batch size optimization
       - Loss function tuning
    
    3. **Validation & Optimization**
       - mAP monitoring
       - IoU threshold adjustment
       - NMS parameter tuning
       - Model ensemble strategies
    """)
    
    # Performance Metrics
    st.markdown("""
    ### 📊 Key Performance Metrics
    
    #### Detection Quality
    - **mAP (mean Average Precision)**
      - Overall detection accuracy
      - Multiple IoU thresholds
      - Per-class evaluation
    
    #### Localization Accuracy
    - **IoU (Intersection over Union)**
      - Bounding box precision
      - Overlap threshold
      - Spatial accuracy
    
    #### Speed Metrics
    - **FPS (Frames Per Second)**
      - Inference speed
      - Hardware utilization
      - Batch processing efficiency
      - Real-time capability assessment
    
    #### Precision & Recall
    - **Precision**: True positives / (True positives + False positives)
    - **Recall**: True positives / (True positives + False negatives)
    - **F1 Score**: Harmonic mean of precision and recall
    """)

    # Add interactive metric visualization
    st.markdown("### 📈 Interactive Metric Visualization")
    
    # Sample data for visualization
    sample_data = {
        'Metric': ['mAP@0.5', 'mAP@0.75', 'mAP@0.5:0.95'],
        'Score': [0.89, 0.76, 0.68]
    }
    
    # Create metric visualization
    fig = px.bar(
        sample_data,
        x='Metric',
        y='Score',
        title='YOLO Performance Metrics',
        color='Score',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Add Speed-Accuracy Trade-off Section
    st.markdown("""
    ### ⚖️ Speed-Accuracy Trade-off
    
    Understanding the balance between detection speed and accuracy is crucial for real-world applications:
    """)

    trade_off_cols = st.columns(2)
    
    with trade_off_cols[0]:
        st.markdown("""
        #### 🚀 Speed Optimization
        - **Model Pruning**
            - Channel pruning
            - Layer reduction
            - Weight quantization
        - **Hardware Acceleration**
            - GPU optimization
            - TensorRT integration
            - Batch processing
        """)
    
    with trade_off_cols[1]:
        st.markdown("""
        #### 🎯 Accuracy Optimization
        - **Model Architecture**
            - Deeper backbone
            - Feature fusion
            - Multi-scale detection
        - **Training Strategies**
            - Data augmentation
            - Loss function tuning
            - Learning rate scheduling
        """)

    # Add Real-world Applications Section
    st.markdown("""
    ### 🌍 Real-world Applications
    
    YOLO's versatility makes it suitable for various applications:
    """)

    app_cols = st.columns(3)
    
    with app_cols[0]:
        st.markdown("""
        #### 🚗 Transportation
        - Vehicle detection
        - Traffic monitoring
        - Parking management
        - License plate recognition
        """)
    
    with app_cols[1]:
        st.markdown("""
        #### 🏭 Industrial
        - Quality control
        - Defect detection
        - Assembly line monitoring
        - Safety compliance
        """)
    
    with app_cols[2]:
        st.markdown("""
        #### 🏥 Healthcare
        - Medical imaging
        - Patient monitoring
        - Equipment tracking
        - Safety protocols
        """)

    # Add Model Comparison Section
    st.markdown("### 🔄 Model Evolution")
    
    evolution_data = {
        'Version': ['YOLOv3', 'YOLOv4', 'YOLOv5', 'YOLOv8'],
        'mAP': [0.72, 0.78, 0.84, 0.89],
        'FPS': [45, 54, 65, 85],
        'Year': [2018, 2020, 2021, 2023]
    }
    
    df = pd.DataFrame(evolution_data)
    
    # Create interactive evolution chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Version'],
        y=df['mAP'],
        name='mAP',
        mode='lines+markers',
        line=dict(color='#2ecc71', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Version'],
        y=np.array(df['FPS'])/100,  # Normalized for visualization
        name='FPS (normalized)',
        mode='lines+markers',
        line=dict(color='#3498db', width=3)
    ))
    
    fig.update_layout(
        title='YOLO Evolution: Performance Improvements',
        xaxis_title='Version',
        yaxis_title='Score',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Add Tips and Best Practices
    st.markdown("""
    ### 💡 Optimization Tips
    
    #### Model Selection
    - Choose backbone based on your computational resources
    - Consider input resolution vs. detection requirements
    - Evaluate batch size impact on throughput
    
    #### Training Optimization
    - Use appropriate augmentation for your use case
    - Implement proper learning rate scheduling
    - Monitor validation metrics regularly
    
    #### Deployment Considerations
    - Optimize model for target hardware
    - Implement proper pre/post-processing
    - Consider model quantization when applicable
    """)

    # Add Additional Resources
    st.markdown("""
    ### 📚 Additional Resources
    
    - **Documentation**: Official YOLOv8 documentation and guides
    - **Research Papers**: Original YOLO papers and improvements
    - **Community**: Forums and discussion groups
    - **Tutorials**: Step-by-step implementation guides
    """)

# Footer (updated)
st.markdown("""
---
### 📊 Model Information
- Architecture: YOLOv8
- Backend: PyTorch
- Metrics Tracking: Streamlit
- Visualization: Plotly

Made with ❤️ using Streamlit and YOLOv8
""")

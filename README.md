![](https://github.com/Someet-Git/yolo_drone_detection/blob/Main/drone.gif)
# Drone Object Detection 
------------------------------------
  - Let's understand how we collected the dataset and trained the CNN->(Convulation Neural Network) model on that dataset and predict the drones or small drones present in the images and videos.
  - We can also use the saved model after the training to predict the drones in any any type of input either live video streams from cameras, or recorded videos.

### Link to the notebook
https://www.kaggle.com/code/someet2405/yolo-drone-vid/notebook
### Link to the Dataset
https://drive.google.com/file/d/1Aqu0pXm_fk4zdxwLeHSFI_LNFHBlulkS/view?usp=drive_link
-----------------------------------------------
## Created a simple webapp which you can run loccaly to predict on new images or videos
**a simple web application built using Streamlit that allows you to upload an image or video, use a YOLOv8 model to run object detection, and download the processed result. The application supports both image and video inputs.**
## Features

- **Image Prediction**: Upload an image (JPG, JPEG, PNG) and get predictions from the YOLOv8 model.
- **Video Prediction**: Upload a video (MP4, AVI, MOV) and get predictions frame-by-frame from the YOLOv8 model.
- **Download Processed Files**: After prediction, you can download the processed image or video.

## Prerequisites
1. **Clone the repository:**

```
git clone https://github.com/Someet-Git/End-to-end-drone-detection
cd yolov8-streamlit-app
```
2. **Before you start, ensure you have the following installed on your system:**

- **Python 3.8 or above**
- **pip** (Python package installer)
- **Virtual Environment** (optional, but recommended)
```
python -m venv venv
source venv/bin/activate      # On macOS/Linux
.\venv\Scripts\activate       # On Windows
```
3. **Install Required Python Packages**
```
pip install -r requirements.txt
```
4. **Run the drone_webapp.py:**
```
streamlit run drone_webapp.py
```
Once the app is running, it will provide a local URL in the terminal (usually something like http://localhost:8501).
Open this link in your web browser to use the application.

Below is the steps taken for Model Traning
---------------------------------------------
## Step 1: Data Collection and Preprocessing
---------------------------------------------
- Collect a dataset containing images or video frames with small drones as the target objects. Annotating these datasets with bounding boxes indicating the location of drones.
Then preprocessing the dataset by resizing images to a fixed size, normalizing pixel values, and augmenting data to increase dataset diversity (e.g., rotation, flipping, scaling).

- So the dataset that I collected was from the Roboflow open source dataset uploaded by some user which contains around 13800 images of Training and 3000 images Validation with the information of there respective boundary boxes. 
- I also collected a separate test data which contains around 163 images to see the performance of the fine tuned model.

-----------------------------------------
## Step 2: Network Architecture
- Now Design a CNN architecture suitable for small object detection and tracking. We can base our design on popular architectures like YOLO (You Only Look Once), SSD (Single Shot MultiBox Detector), or Faster R-CNN (Region-based Convolutional Neural Networks).
- After that we have to ensure that the architecture is lightweight and optimized for real-time performance on devices like drones. This may involve reducing the number of layers, using smaller filter sizes, or employing depthwise separable convolutions.

- I used YOLO model which is based on CNN and the version that I used is v8.0.196. 

- This is the YOLOV8 architecture, you can see that this is an Anchor free model which means it directly predicts the centre of the image instead of the offsets called anchor box.

![image](https://github.com/Someet-Git/yolo_drone_detection/assets/32305867/5509539b-a133-40ce-b95e-c78c9d0122e9)

-----------------------------------------------------

And the model that I was traing consist of 295 layers and around 25856899 parameters , and working at 79.1 GFLOPS.

-----------------------------------------
![image](https://github.com/Someet-Git/yolo_drone_detection/assets/32305867/fdb6e27e-6738-43c6-aba7-8c2172d3bfc6)
---------------------------------------------------
- I trained the model for 10 epochs(you can train for more number of epochs like 50 or 100 with a larger Batch size, I used a image batch of 512 images, we can also use 800 or more if you have the larger and powerful GPUs.
- Optimization is **Adamw optimizer and learning rate = 0.002**. 
- After the fine tuning of the model the model is saved(best.pt) and we can use that to predict on images and video streams.

------------------------------------------

## Step 3: Evaluation
- Evaluate the trained model on a separate test set to assess its performance in terms of detection accuracy, tracking stability, and real-time inference speed.
- Then I computed metrics like precision, recall, F1-score, and Confusion matric to see the how the model predicted on the validation and test images. 
- Then I analyze the results and iteratively refine the model architecture and training process based on the observed performance.
- access the loss function in the `working/runs/detect/train/result.png` --> see it is for 10 epochs only and the we can increase the epochs to reduce the errors.
- access the confusion matrix in `working/runs/detect/train/confusion_matrix.png`
- 
------------------------------------------
## Step 4: Prediction on Image and Video data
- The predictions from Model can be found on the `Predict4` folder
![image](https://github.com/Someet-Git/yolo_drone_detection/blob/Main/predict4/00475_jpg.rf.b6113f88c935a14e766529ee47ba1ecb.jpg?raw=true)
- After that I uploaded 2 video from the youtube to see how the model performs.
- You can access the output from the model it is upladed as `drone detection on video data.mp4`
  
**How it is predicting in the videos basically it is converting the videos into frames and for each frame it is finding Drones in them**

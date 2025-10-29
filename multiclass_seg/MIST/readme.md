create the readme file the context is 



Real-Time Multi-Class Polyp Segmentation

This project provides a complete pipeline for training a deep learning model for multi-class semantic segmentation of medical polyps and deploying it for real-time inference on a live video stream. The core model architecture used is MIST_CAM, and the pipeline is optimized for performance and ease of use.

Key Features

End-to-End Pipeline: Covers everything from data preparation and training to model export and real-time deployment.

Multi-Class Segmentation: Capable of distinguishing between different classes of polyps (e.g., Adenoma, Hyperplastic) by mapping specific colors in mask images to class labels.

Robust Data Preparation: The data processing script automatically handles images without masks, treating them as background-only samples to improve model robustness.

High-Performance Inference: Uses ONNX Runtime with GPU (CUDA) acceleration to achieve real-time latency (typically 40-60 ms end-to-end).

Portable Configuration: All file paths are managed through a central .env file, allowing the project to run on different machines (like a local Windows PC or a Linux AWS server) without changing the code.

Dataset Verification: Includes a utility script to analyze the processed dataset, providing insights into class balance at both the pixel and image level.

Project Structure

To use this project, you should organize your files and folders as follows. The dataset, processed_data, and .env files are ones you will create.

code
Code
download
content_copy
expand_less
/your_project_root/
|
├── .env                  <-- You will CREATE this configuration file
|
├── dataset/                <-- You will CREATE this for your raw data
│   ├── images/
│   │   ├── image1.jpg
│   │   └── image2.png
│   └── masks/
│       ├── image1.png
│       └── image2.png
|
├── processed_data/         <-- The scripts will CREATE these folders
│   ├── npz_files/
│   └── list_files/
|
├── models/                 <-- The training script will CREATE this folder
│   └── Synapse/
|
├── src/                    <-- All the Python scripts are located here
│   ├── data_ready3.py
│   ├── check2.py
│   ├── Synapse_train.py
│   ├── onnx_export.py
│   └── onnx_test.py
|
└── README.md
Setup and Installation

Clone the Repository:

code
Bash
download
content_copy
expand_less
git clone https://github.com/your-username/your-repository.git
cd your-repository

Create a Python Virtual Environment (Recommended):

code
Bash
download
content_copy
expand_less
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

Install Dependencies:
Install all required libraries using pip. It is recommended to use a version of onnxruntime-gpu that matches your system's CUDA version.

code
Bash
download
content_copy
expand_less
pip install opencv-python numpy scikit-learn python-dotenv torch torchvision onnx onnxruntime-gpu
Step-by-Step Workflow

Follow these steps in order to go from a raw dataset to a running real-time application.

Step 1: Prepare Your Dataset

Before running any scripts, you must organize your image data.

Create a dataset folder in the project root.

Inside dataset, create two subfolders: images and masks.

Place all your raw images (e.g., .jpg, .png) into the dataset/images/ folder.

Place their corresponding mask images into the dataset/masks/ folder.

Important Rules:

Each image and its mask must share the same base filename (e.g., frame_123.jpg and frame_123.png).

The masks must use specific colors for each class:

Red (RGB: 255, 0, 0) for Class 1 (e.g., Adenoma).

Green (RGB: 0, 255, 0) for Class 2 (e.g., Hyperplastic).

Black (RGB: 0, 0, 0) for the Background.

Step 2: Configure Your Environment

You must tell the scripts where to find your data and where to save the results.

In the project's root directory, create a new file named .env.

Copy and paste the following content into the .env file, updating the paths to match the locations on your computer. Use absolute paths for simplicity.

code
Env
download
content_copy
expand_less
# --- .env Configuration File ---

# 1. Paths for the raw dataset (from Step 1)
data_2_0_image="C:/path/to/your_project_root/dataset/images"
data_2_0_mask="C:/path/to/your_project_root/dataset/masks"

# 2. Paths for processed data (the scripts will create and use these)
data_2_0_npz="C:/path/to/your_project_root/processed_data/npz_files"
data_2_0_synapse="C:/path/to/your_project_root/processed_data/list_files"
Step 3: Run the Data Preparation Script

This script will read your raw data, split it into train/validation/test sets, convert color masks to class labels, and save it in the efficient .npz format.

code
Bash
download
content_copy
expand_less
python src/data_ready3.py

After running, your processed_data folder will be populated with .npz files and .txt list files.

Step 4: (Optional) Verify Your Processed Dataset

To ensure your data was processed correctly and to check for class imbalance, you can run the analysis script. This is highly recommended to understand your dataset's characteristics before training.

code
Bash
download
content_copy
expand_less
python src/check2.py

This will print a detailed report to the console showing image counts and pixel-level distribution for each class in each data split.

Step 5: Train the Model

Now you are ready to train the segmentation model. This script will load the processed .npz files and start the training process.

code
Bash
download
content_copy
expand_less
python src/Synapse_train.py

Trained model checkpoints (.pth files) will be saved in the models/Synapse/ directory.

The save folder will have a descriptive name that includes the training parameters and a timestamp (e.g., run_2025-10-14_18...).

You can adjust hyperparameters like learning rate, batch size, and epochs directly inside the Synapse_train.py script.

Step 6: Export the Model to ONNX

To use the model for high-speed inference, you must convert the trained PyTorch (.pth) model to the standard ONNX (.onnx) format.

IMPORTANT: Open the src/onnx_export.py script in an editor.

Find the PYTORCH_MODEL_PATH variable and update it to the exact path of the trained model you want to export (e.g., .../epoch_49.pth from the previous step).

Run the script from your terminal:

code
Bash
download
content_copy
expand_less
python src/onnx_export.py

This will create a mist_cam_polyp_rgb.onnx file in the same folder where your .pth model is located.

Step 7: Test the Model with Real-Time Inference

Finally, you can run the live inference script to see your model in action on a video file or webcam feed. The onnx_test.py script is optimized for performance.

IMPORTANT: Open the src/onnx_test.py script in an editor.

Update the PYTORCH_MODEL_FOLDER variable to point to the directory containing your new .onnx file.

Update the VIDEO_SOURCE variable:

To use a webcam, set it to a number (e.g., 0 for the first camera, 1 for the second).

To use a video file, set it to the full path of the file (e.g., "C:/Users/YourUser/Videos/test_video.mp4").

Run the script:

code
Bash
download
content_copy
expand_less
python src/onnx_test.py

A window will appear showing the video feed with bounding boxes and class labels drawn around any detected polyps in real-time. Press 'q' to quit.
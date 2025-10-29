🧠 Real-Time Multi-Class Polyp Segmentation

This project provides a complete pipeline for training and deploying a deep learning model for real-time multi-class semantic segmentation of medical polyps.
It leverages the MIST_CAM architecture and is optimized for accuracy, performance, and ease of use.

🚀 Key Features

✅ End-to-End Pipeline — From dataset preparation and model training to ONNX export and real-time inference.
✅ Multi-Class Segmentation — Detects and classifies polyps (e.g., Adenoma, Hyperplastic) using color-coded masks.
✅ Robust Data Preparation — Automatically handles missing masks by labeling them as background-only samples.
✅ High-Performance Inference — Real-time inference via ONNX Runtime + CUDA, achieving ~40–60 ms latency.
✅ Portable Configuration — Centralized configuration via .env file for easy portability across Windows/Linux systems.
✅ Dataset Verification Utility — Analyze dataset composition and pixel-level class balance before training.

📁 Project Structure

Your project should be organized as follows:

/your_project_root/
│
├── .env                     <-- You will CREATE this file
│
├── dataset/                 <-- You will CREATE this folder
│   ├── images/
│   │   ├── image1.jpg
│   │   └── image2.png
│   └── masks/
│       ├── image1.png
│       └── image2.png
│
├── processed_data/          <-- Created automatically by scripts
│   ├── npz_files/
│   └── list_files/
│
├── models/                  <-- Created during training
│   └── Synapse/
│
├── src/                     <-- All source Python scripts
│   ├── data_ready3.py
│   ├── check2.py
│   ├── Synapse_train.py
│   ├── onnx_export.py
│   └── onnx_test.py
│
└── README.md

⚙️ Setup and Installation
1️⃣ Clone the Repository
git clone https://github.com/your-username/your-repository.git
cd your-repository

2️⃣ Create a Python Virtual Environment
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

3️⃣ Install Dependencies

Ensure you install the correct version of onnxruntime-gpu that matches your CUDA setup.

pip install opencv-python numpy scikit-learn python-dotenv torch torchvision onnx onnxruntime-gpu

🧩 Step-by-Step Workflow
Step 1: Prepare Your Dataset

Create a dataset folder with subfolders images/ and masks/.

Ensure that:

Each image and mask share the same base name (e.g., frame_123.jpg ↔ frame_123.png).

The mask colors follow this convention:

Red (255, 0, 0) → Class 1 (e.g., Adenoma)

Green (0, 255, 0) → Class 2 (e.g., Hyperplastic)

Black (0, 0, 0) → Background

Step 2: Configure Your Environment

Create a .env file in your project root with the following contents (update the paths accordingly):

# --- .env Configuration File ---

# 1. Raw dataset paths
data_2_0_image="C:/path/to/your_project_root/dataset/images"
data_2_0_mask="C:/path/to/your_project_root/dataset/masks"

# 2. Processed data paths
data_2_0_npz="C:/path/to/your_project_root/processed_data/npz_files"
data_2_0_synapse="C:/path/to/your_project_root/processed_data/list_files"

Step 3: Run the Data Preparation Script

Converts color masks to numeric labels and prepares .npz datasets:

python src/data_ready3.py


The processed files will be saved inside processed_data/npz_files and processed_data/list_files.

Step 4: Verify the Processed Dataset (Optional)

Analyze dataset statistics and check for class imbalance:

python src/check2.py


This generates a report summarizing image counts and pixel-level class distributions.

Step 5: Train the Model

Train the MIST_CAM model using the processed data:

python src/Synapse_train.py


Outputs:

Model checkpoints (.pth) saved in models/Synapse/

Folder names include timestamps and hyperparameter info (e.g., run_2025-10-14_18...)

Adjust hyperparameters (batch size, learning rate, epochs) directly in Synapse_train.py.

Step 6: Export the Model to ONNX

Convert the trained PyTorch model to ONNX format for fast deployment.

Open src/onnx_export.py

Update:

PYTORCH_MODEL_PATH = "path/to/your_model.pth"


Run:

python src/onnx_export.py


This generates a file named mist_cam_polyp_rgb.onnx in your model directory.

Step 7: Run Real-Time Inference

Visualize real-time segmentation on webcam or video.

Open src/onnx_test.py

Update:

PYTORCH_MODEL_FOLDER = "path/to/your/onnx/model"
VIDEO_SOURCE = 0  # or "path/to/video.mp4"


Run:

python src/onnx_test.py


A window will appear showing live video with segmented polyps highlighted.
Press ‘q’ to exit.

🧠 Notes & Best Practices

Ensure CUDA and onnxruntime-gpu are properly configured for GPU inference.

Keep your .env paths consistent across scripts.

Use descriptive filenames for models and logs to track experiments.

🧾 License

This project is released under the MIT License.
You are free to use, modify, and distribute it with proper attribution.
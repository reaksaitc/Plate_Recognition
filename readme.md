# 🚗 Vehicle & License Plate Recognition System

This project provides a complete vehicle and license plate recognition pipeline with a **Streamlit web interface**.

The pipeline uses multiple trained models to process an uploaded image and produce vehicle, plate, and recognition results.

---

## 📌 Main Features

The application can:

- Upload an image for testing
- Detect vehicles
- Classify vehicle information
- Detect license plates
- Process license plate regions
- Recognize license plate text
- Display detection and recognition results
- Run the complete pipeline through a Streamlit interface

---

## 📁 Project Structure

```text
model/
│
├── app.py
│   └── Streamlit web application
│
├── pipeline_without_dir.py
│   └── Main inference pipeline
│
├── pc_class_if.json
│   └── Plate class configuration
│
├── pc_classifier_best.pt
│   └── Trained plate classification model
│
├── plate.pt
│   └── License plate detection model
│
├── pn_bbx.pt
│   └── Plate number / character detection model
│
├── pn_crnn_best.pt
│   └── CRNN plate recognition model
│
├── pn_ocr.json
│   └── OCR configuration / character mapping
│
├── vc1.pt
│   └── Vehicle classification/detection model
│
├── vc2.pt
│   └── Vehicle classification/detection model
│
├── requirements.txt
│   └── Required Python libraries
│
├── python_version.txt
│   └── Recommended Python version
│
└── README.md
    └── Project setup instructions
```

> `.venv/` and `__pycache__/` are not included because each user should create their own virtual environment.

---

# ⚙️ Installation

## 1. Install Python

First, make sure Python is installed.

Check your Python version:

```bash
python --version
```

The recommended Python version for this project can also be found in:

```text
python_version.txt
```

It is recommended to use the same Python version used during development.

---

## 2. Open the Project Folder

After downloading or extracting the ZIP file, open a terminal inside the project folder.

Example:

```bash
cd model
```

---

## 3. Create a Virtual Environment

Create a new Python virtual environment:

```bash
python -m venv .venv
```

This keeps the project dependencies separate from other Python projects.

---

## 4. Activate the Virtual Environment

### Git Bash

```bash
source .venv/Scripts/activate
```

### Windows Command Prompt

```cmd
.venv\Scripts\activate
```

### PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

After activation, the terminal should show something similar to:

```text
(.venv)
```

---

## 5. Upgrade pip

Run:

```bash
python -m pip install --upgrade pip
```

---

## 6. Install Required Libraries

Install all required dependencies using:

```bash
pip install -r requirements.txt
```

This may take several minutes because packages such as:

- PyTorch
- Ultralytics
- OpenCV
- NumPy
- Streamlit

may need to be downloaded.

---

# ▶️ Run the Application

Make sure the virtual environment is activated.

Then run:

```bash
streamlit run app.py
```

Streamlit should display something similar to:

```text
Local URL: http://localhost:8501
```

Open the URL in your browser.

Usually:

```text
http://localhost:8501
```

---

# 🖼️ Using the Application

After the application starts:

1. Open the Streamlit page.
2. Upload an image.
3. Wait for the models to process the image.
4. The application will run the complete recognition pipeline.
5. Detection and recognition results will be displayed on the page.

The models are loaded from the `.pt` files included in the project.

---

# 🧠 Models Used

The project contains several trained models because the recognition system is divided into multiple stages.

```text
Input Image
     │
     ▼
Vehicle Processing
     │
     ▼
Vehicle / Plate Classification
     │
     ▼
License Plate Detection
     │
     ▼
Plate Region Processing
     │
     ▼
OCR / CRNN Recognition
     │
     ▼
Final Result
```

Model files include:

```text
vc1.pt
vc2.pt
plate.pt
pc_classifier_best.pt
pn_bbx.pt
pn_crnn_best.pt
```

⚠️ Do not delete or rename these model files unless the corresponding paths in the Python code are also updated.

---

# 💻 GPU Support

The project can use an NVIDIA GPU if PyTorch detects CUDA correctly.

Check whether PyTorch can see your GPU:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Example successful result:

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
```

If it shows:

```text
CUDA available: False
```

the application may still run using the CPU, but inference will usually be slower.

---

# ⚠️ PyTorch / CUDA Note

Different computers may have different:

- NVIDIA GPUs
- NVIDIA drivers
- CUDA configurations

Therefore, if installation fails specifically for `torch`, `torchvision`, or CUDA-related packages, install the appropriate PyTorch version for that computer and then install the remaining requirements.

You can check the current PyTorch installation with:

```bash
python -c "import torch; print(torch.__version__)"
```

And CUDA support with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

# 🔧 Common Problems

## `streamlit` is not recognized

Make sure the virtual environment is activated:

```bash
source .venv/Scripts/activate
```

Then install the requirements again:

```bash
pip install -r requirements.txt
```

---

## `ModuleNotFoundError`

Example:

```text
ModuleNotFoundError: No module named 'cv2'
```

Run:

```bash
pip install -r requirements.txt
```

Or install the missing package individually if necessary.

For example:

```bash
pip install opencv-python
```

---

## Model file not found

Example:

```text
FileNotFoundError: plate.pt
```

Make sure all model files remain inside the project folder.

For example:

```text
model/
├── app.py
├── plate.pt
├── vc1.pt
├── vc2.pt
└── ...
```

Do not move the `.pt` files unless you also update their paths in the source code.

---

## Streamlit port already in use

Run the application on another port:

```bash
streamlit run app.py --server.port 8502
```

Then open:

```text
http://localhost:8502
```

---

# 📦 Sharing the Project

When sharing this project with another team member, include:

```text
app.py
pipeline_without_dir.py
*.pt
*.json
requirements.txt
python_version.txt
README.md
```

Do **not** include:

```text
.venv/
__pycache__/
```

Each team member should create their own `.venv`.

---

# 🚀 Quick Setup

For Windows + Git Bash, the complete setup is:

```bash
cd model

python -m venv .venv

source .venv/Scripts/activate

python -m pip install --upgrade pip

pip install -r requirements.txt

streamlit run app.py
```

After installation, future runs only require:

```bash
cd model

source .venv/Scripts/activate

streamlit run app.py
```

---

# 👥 Team Setup Summary

When another team member receives this project:

```text
Download / Extract ZIP
        ↓
Open project folder
        ↓
Create .venv
        ↓
Activate .venv
        ↓
Install requirements.txt
        ↓
Run app.py with Streamlit
        ↓
✅ Application Ready
```

---

## Important

Please keep all model files and configuration files in their original locations because the inference pipeline depends on them.

If the program works correctly on the original development computer but fails on another computer, first check:

1. Python version
2. Virtual environment activation
3. Installed dependencies
4. PyTorch version
5. CUDA/GPU support
6. Model file paths
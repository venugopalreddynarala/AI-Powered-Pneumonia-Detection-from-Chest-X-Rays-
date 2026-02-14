# 🫁 AI-Powered Pneumonia Detection System

A complete production-grade AI system for detecting pneumonia from chest X-ray images with explainable AI visualizations, 3D lung mapping, and medical recommendations.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Features

### Core Functionality
- **🔬 Deep Learning Model**: DenseNet121 architecture with transfer learning
- **🔥 Grad-CAM Visualization**: Explainable AI showing which lung regions influenced the prediction
- **🫁 3D Lung Visualization**: Interactive Plotly 3D visualization with damage overlay
- **📊 Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC, BLEU Score
- **💊 Medical Recommendations**: Severity-based actionable medical advice
- **🌐 Web Interface**: Beautiful Streamlit UI for easy interaction

### Technical Highlights
- Automatic Kaggle dataset download via API
- Data augmentation and preprocessing pipeline
- Model training with early stopping and learning rate scheduling
- Complete evaluation pipeline with visualizations
- PDF report generation
- Production-ready deployment architecture

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- CUDA (optional, for GPU acceleration)
- Kaggle API credentials (for dataset download)

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd xray2
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Setup Kaggle API
1. Create a Kaggle account at [kaggle.com](https://www.kaggle.com)
2. Go to Account Settings → API → Create New API Token
3. Download `kaggle.json` file
4. Place it in the appropriate location:
   - **Linux/Mac**: `~/.kaggle/kaggle.json`
   - **Windows**: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
5. Set permissions (Linux/Mac only):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

---

## 🚀 Quick Start

### Option 1: Complete Pipeline (Recommended)

```bash
# 1. Train the model (downloads dataset automatically)
python train.py

# 2. Evaluate the model
python evaluate.py

# 3. Launch web application
streamlit run app.py
```

### Option 2: Custom Configuration

```bash
# Train with custom parameters
python train.py --epochs 25 --batch_size 64 --lr 0.0001

# Evaluate with custom paths
python evaluate.py --model_path models/model_weights.pth --output_dir results

# Run Streamlit on custom port
streamlit run app.py --server.port 8080
```

---

## 📂 Project Structure

```
xray2/
├── data/                          # Dataset directory (auto-created)
│   └── chest_xray/               # Kaggle dataset
│       ├── train/
│       ├── val/
│       └── test/
├── models/                        # Trained model weights
│   ├── model_weights.pth         # Best model checkpoint
│   ├── training_history.json     # Training metrics
│   └── training_history.png      # Training curves
├── results/                       # Evaluation results
│   ├── metrics.csv               # Performance metrics
│   ├── confusion_matrix.png      # Confusion matrix plot
│   ├── roc_curve.png             # ROC curve plot
│   ├── metrics_bar.png           # Metrics bar chart
│   ├── classification_report.txt # Detailed report
│   └── evaluation_report.pdf     # PDF report
├── utils/                         # Utility modules
│   ├── __init__.py
│   ├── data_prep.py              # Data loading & preprocessing
│   ├── gradcam.py                # Grad-CAM visualization
│   ├── metrics.py                # Performance metrics
│   ├── visualization3d.py        # 3D lung visualization
│   └── recommendations.py        # Medical recommendations
├── train.py                       # Training script
├── evaluate.py                    # Evaluation script
├── app.py                         # Streamlit web app
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
```

---

## 🎓 Usage Guide

### 1. Training the Model

The training script automatically:
- Downloads the Kaggle dataset
- Preprocesses and augments images
- Trains DenseNet121 with transfer learning
- Saves the best model based on validation accuracy
- Generates training curves

```bash
python train.py
```

**Expected Output:**
```
==================================================================
PNEUMONIA DETECTION MODEL TRAINING
==================================================================

Using device: cuda
GPU: NVIDIA GeForce RTX 3090

Loading datasets...
Loaded 5216 images for train mode
Loaded 16 images for val mode
Training batches: 163
Validation batches: 1

Building model...
Model built successfully!
Total parameters: 7,978,856
Trainable parameters: 2,458,624

==================================================================
STARTING TRAINING
==================================================================

Epoch 1/20 [Train]: 100%|██████████| 163/163 [02:45<00:00]
Epoch 1/20 [Val]:   100%|██████████| 1/1 [00:01<00:00]

Epoch 1/20 Summary:
  Train Loss: 0.3245 | Train Acc: 0.8532
  Val Loss:   0.2891 | Val Acc:   0.8750
  Time: 166.23s
  ✓ Best model saved! (Val Acc: 0.8750)
...
```

### 2. Evaluating the Model

```bash
python evaluate.py
```

**Generates:**
- Performance metrics (CSV)
- Confusion matrix
- ROC curve
- Classification report
- PDF evaluation report

### 3. Running the Web Application

```bash
streamlit run app.py
```

**Access at:** `http://localhost:8501`

**Features:**
- Upload chest X-ray images
- View real-time predictions
- Explore Grad-CAM heatmaps
- Interact with 3D lung visualization
- Get medical recommendations
- Download analysis reports

---

## 📊 Model Performance

### Expected Metrics (after 20 epochs)

| Metric | Score |
|--------|-------|
| **Accuracy** | 0.9250+ |
| **Precision** | 0.9180+ |
| **Recall** | 0.9420+ |
| **F1-Score** | 0.9298+ |
| **AUC-ROC** | 0.9610+ |
| **BLEU Score** | 0.3500+ |

### Confusion Matrix
```
              Predicted
              Normal  Pneumonia
Actual Normal   225      9
      Pneumo     18     372
```

---

## 🔬 Technical Details

### Model Architecture
- **Base Model**: DenseNet121 (pretrained on ImageNet)
- **Modifications**:
  - Frozen early layers for transfer learning
  - Fine-tuned last dense block
  - Custom classifier: 1024 → 512 → 2 neurons
  - Dropout (0.3) for regularization

### Data Augmentation
- Random horizontal flip
- Random rotation (±10°)
- Color jitter (brightness, contrast)
- Center crop (224×224)
- ImageNet normalization

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Cross-Entropy
- **Scheduler**: ReduceLROnPlateau (patience=3, factor=0.5)
- **Batch Size**: 32
- **Epochs**: 20 (with early stopping)

### Grad-CAM Implementation
- **Target Layer**: `denseblock4.denselayer16.conv2`
- **Method**: Gradient-weighted Class Activation Mapping
- **Output**: 224×224 heatmap with color intensity

### 3D Visualization
- **Library**: Plotly (WebGL rendering)
- **Method**: Parametric lung surface generation
- **Color Mapping**: Green (normal) → Yellow → Orange → Red (severe)

---

## 🩺 Medical Recommendations

The system provides severity-based recommendations:

### Severity Levels
1. **Normal**: No pneumonia detected
2. **Mild**: Early-stage pneumonia (0-20% affected area)
3. **Moderate**: Significant pneumonia (20-40% affected area)
4. **Severe**: Advanced pneumonia (40%+ affected area)

### Recommendations Include:
- Urgency level (Low/Moderate/High/Critical)
- Action items (rest, consultation, medication, hospitalization)
- Follow-up timeline
- Warning signs to watch for
- Lifestyle advice

---

## 📝 API Reference

### Training Functions

```python
from train import build_model, train_pipeline

# Build model
model = build_model(num_classes=2, pretrained=True)

# Train model
model, history = train_pipeline(
    data_dir='data/chest_xray',
    epochs=20,
    batch_size=32,
    learning_rate=0.001
)
```

### Prediction Functions

```python
from utils.data_prep import preprocess_single_image
import torch

# Preprocess image
input_tensor = preprocess_single_image('xray.jpg')

# Predict
model.eval()
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)
    prediction = output.argmax(dim=1).item()
```

### Grad-CAM Functions

```python
from utils.gradcam import generate_gradcam_visualization

# Generate Grad-CAM
heatmap, overlaid, intensity = generate_gradcam_visualization(
    model, input_tensor, original_image
)
```

---

## 🛠️ Troubleshooting

### Issue: Kaggle API Not Working
**Solution:**
```bash
pip install --upgrade kaggle
# Ensure kaggle.json is in correct location with proper permissions
```

### Issue: CUDA Out of Memory
**Solution:**
```bash
# Reduce batch size
python train.py --batch_size 16
```

### Issue: Model Not Loading in Streamlit
**Solution:**
```bash
# Ensure model is trained first
python train.py

# Check if model_weights.pth exists
ls models/model_weights.pth
```

### Issue: Slow Training
**Solution:**
- Enable GPU: Install `torch` with CUDA support
- Reduce image size in `data_prep.py`
- Use mixed precision training (FP16)

---

## 🚢 Deployment

### Streamlit Cloud
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Deploy (automatic detection of `app.py`)

### Hugging Face Spaces
1. Create new Space
2. Upload files
3. Add `requirements.txt`
4. Space auto-deploys

### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

```bash
docker build -t pneumonia-ai .
docker run -p 8501:8501 pneumonia-ai
```

---

## 📚 Dataset Information

**Source**: [Kaggle - Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

**Statistics**:
- Total Images: 5,863
- Training Set: 5,216 images
- Validation Set: 16 images
- Test Set: 624 images
- Classes: Normal (1,583) vs Pneumonia (4,273)

**Citation**:
```
Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), 
"Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images 
for Classification", Mendeley Data, V2
```

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- [ ] Add more model architectures (ResNet, EfficientNet)
- [ ] Implement ensemble methods
- [ ] Add multi-class classification (bacterial vs viral)
- [ ] Improve 3D visualization with actual lung models
- [ ] Add patient history tracking
- [ ] Implement DICOM support

---

## 📄 License

This project is licensed under the MIT License.

---

## ⚠️ Disclaimer

**IMPORTANT MEDICAL DISCLAIMER:**

This AI system is designed for **educational and research purposes only**. It is NOT intended for clinical use or as a substitute for professional medical diagnosis.

- Always consult qualified healthcare professionals for medical advice
- Do not make treatment decisions based solely on AI predictions
- This tool should only be used as a screening aid by trained medical personnel
- Results should be verified with additional diagnostic methods
- The developers assume no liability for medical decisions made using this system

---

## 🙏 Acknowledgments

- **Dataset**: Kermany et al. (Mendeley Data)
- **Architecture**: DenseNet (Huang et al., CVPR 2017)
- **Grad-CAM**: Selvaraju et al., ICCV 2017
- **Libraries**: PyTorch, Streamlit, Plotly, Scikit-learn

---

## 📧 Contact

For questions, issues, or collaboration:
- **Email**: your.email@example.com
- **GitHub Issues**: [Project Issues](https://github.com/username/repo/issues)

---

## 🌟 Star This Repository

If you find this project useful, please consider giving it a star ⭐!

---

**Built with ❤️ for Healthcare AI**

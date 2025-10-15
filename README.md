
# 🧠 Brain Tumor Detection System

An AI-powered web application that detects brain tumors from MRI scans with 91.2% accuracy using YOLOv8 deep learning.

## 🚀 Live Demo
[**Try the Live App on Hugging Face**](https://huggingface.co/spaces/anis0Talbi/brain_tumor)

![Brain Tumor Detection](https://img.shields.io/badge/Accuracy-91.2%25-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-8.0%2B-green)

## 📊 Project Highlights
- **91.2% mAP50** accuracy on test data
- **Real-time tumor detection** from MRI images
- **Multiple tumor types**: Glioma, Meningioma, Pituitary
- **Professional web interface** built with Gradio

## 🛠️ Technical Stack
- **Model**: YOLOv8 (Ultralytics)
- **Framework**: PyTorch
- **Interface**: Gradio
- **Dataset**: RoboFlow Brain Tumor MRI (3,903 images)

## 🎯 Performance Metrics
| Tumor Type | Detection Accuracy |
|------------|-------------------|
| **Overall** | 91.2% |
| Meningioma | 97.3% |
| Pituitary | 94.9% |
| Glioma | 81.5% |

## 💻 Usage
1. Upload a brain MRI scan (JPG/PNG)
2. Get instant tumor detection results
3. View bounding boxes and confidence scores
4. Receive detailed analysis report

## 📁 Project Structure
brain-tumor-detection/
├── app.py # Gradio web interface
├── requirements.txt # Python dependencies
├── brain_tumor_model.pt # Trained model weights
└── README.md # Project documentation 

## 🔬 Medical Context
This tool is designed for:
- **Educational purposes** in medical imaging
- **Research applications** in oncology
- **Radiology assistance** for preliminary screening

*Note: For educational purposes. Always consult healthcare professionals for medical diagnosis.*

## 📄 License
MIT License - Feel free to use this project for learning and research purposes.

---
**Built with ❤️ using PyTorch and YOLOv8**

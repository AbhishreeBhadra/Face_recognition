# Face_recognition

Real-Time Face Recognition System

Description  
A real-time face recognition system built using Python, OpenCV, TensorFlow, and Keras. The project leverages a fine-tuned VGG16 Convolutional Neural Network (CNN) for multi-class classification, achieving 99% accuracy. It includes efficient image preprocessing, real-time face detection, and live inference for seamless performance.

Features  
- Real-time face detection and recognition using webcam.  
- Multi-class classification with high accuracy.  
- Fine-tuned VGG16 CNN model for robust predictions.  
- Scalable and modular codebase for future enhancements.

Installation  
1. Clone the repository:  
   ```
   git clone https://github.com/yourusername/face-recognition.git  
   cd face-recognition  
   ```

2. Install the required Python libraries:  
   ```
   pip install -r requirements.txt  
   ```

3. Download the Haarcascade XML file:  
   Download from https://github.com/opencv/opencv/tree/master/data/haarcascades.

Usage  
1. Dataset Preparation:  
   Place images in Datasets/Train/ with subfolders for each class (e.g., Abhishree, Adrika, Ahana, Aishwairya).

2. Train the Model:  
   Run the training script to fine-tune the VGG16 model:  
   ```
   python train_model.py  
   ```

3. Real-Time Recognition:  
   Use the trained model for face recognition via webcam:  
   ```
   python recognize_face.py  
   ```

Project Structure  
```
face-recognition/
│
├── Datasets/
│   ├── Train/        Training images organized by class
│   ├── Test/         Placeholder for test data (optional)
├── train_model.py     Script for training the model
├── recognize_face.py  Real-time face recognition script
├── requirements.txt   Python dependencies
└── README.md          Project documentation
```

Acknowledgments  
- OpenCV for face detection using Haar cascades.  
- TensorFlow/Keras for model fine-tuning and predictions.  
- VGG16 pretrained CNN architecture for transfer learning.


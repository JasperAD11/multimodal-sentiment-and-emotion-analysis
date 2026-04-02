# Multimodal Sentiment and Emotion Analysis: Neural Networks and Large Language Models

This repository contains a class project for Advanced Topics in Predictive Analytics at Católica Lisbon. It examines sentiment and emotion analysis across text and speech data using neural networks and compares selected results with large language models (LLMs) implemented in Python and TensorFlow.

---

## Overview

This project investigates the effectiveness of traditional neural networks versus large language models in sentiment and emotion classification tasks. It encompasses:

- **Binary Sentiment Classification:** Utilizing the IMDB dataset to classify text as positive or negative.
- **Multi-label Emotion Detection:** Employing the GoEmotions dataset to identify multiple emotions present in text.
- **Speech Emotion Recognition:** Analyzing audio inputs to detect emotions using models such as Whisper.
- **Comparison with LLMs:** Evaluating the performance of models such as GPT-3.5 and GPT-4 against traditional neural networks.

---

## Project Structure

```text
multimodal-sentiment-and-emotion-analysis/
├── binary_sentiment_model.h5                 # Pretrained binary sentiment model
├── multilabel_emotion_model.h5              # Pretrained multi-label emotion model
├── text_vectorizer_vocab.txt                # Vocabulary for text vectorization
├── merged_dataset.csv                       # Merged dataset used for training
├── emotion_labels.csv                       # Labels for the datasets
├── model_architecture.py                    # Model architecture definitions
├── demo_inference.ipynb                     # Demo notebook for model inference
├── data_preprocessing.ipynb                 # Initial data exploration and preprocessing
├── training_evaluation.ipynb                # Final training and evaluation notebook
├── text_analysis_additional.ipynb           # Additional analyses and visualizations
├── llm_comparison.ipynb                     # Comparative study with LLMs
├── speech_emotion_recognition_whisper.ipynb # Speech emotion recognition using Whisper
└── README.md                                # Project documentation

---

**Setup section adjustment**

Change the clone line to:

```bash
git clone https://github.com/YOUR-USERNAME/multimodal-sentiment-and-emotion-analysis.git
cd multimodal-sentiment-and-emotion-analysis

2. **Create a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download Pretrained Models**

   Due to GitHub's file size limitations, large model files are not included. You can download them using the provided `download_models.py` script or manually from the provided links.

   ```bash
   python download_models.py
   ```

---

## 🚀 Running the Demo

After setting up the environment and downloading the models:

1. **Launch the Demo Notebook**

   ```bash
   jupyter notebook demo.ipynb
   ```

2. **Interact with the Models**

   - Input custom text to see sentiment and emotion predictions.
   - Analyze audio files for emotion detection using the Whisper model.

---

## 📊 Results & Findings

- **Binary Sentiment Model**: Achieved an accuracy of 92% on the IMDB test set.
- **Multi-label Emotion Model**: Demonstrated a macro F1-score of 0.76 on the GoEmotions dataset.
- **LLM Comparison**: GPT-4 outperformed traditional models in zero-shot settings but required significantly more computational resources.
- **Speech Emotion Recognition**: The Whisper-based model accurately identified emotions in 85% of the test audio samples.

---

## 🤖 Model Architectures

### Binary Sentiment Model

- **Input**: Text sequences
- **Layers**:
  - TextVectorization
  - Embedding
  - GlobalAveragePooling1D
  - Dense (ReLU)
  - Dense (Sigmoid)

### Multi-label Emotion Model

- **Input**: Text sequences
- **Layers**:
  - TextVectorization
  - Embedding
  - Bidirectional LSTM
  - Dense (ReLU)
  - Dense (Sigmoid for multi-label output)

---

## 📈 Evaluation Metrics

- **Accuracy**: For binary classification tasks.
- **Precision, Recall, F1-Score**: For multi-label emotion detection.
- **Confusion Matrix**: To visualize model performance.
- **ROC-AUC**: For assessing classification thresholds.

---

## 🧠 Future Work

- **Model Optimization**: Implementing attention mechanisms to improve performance.
- **Dataset Expansion**: Incorporating more diverse datasets for better generalization.
- **Real-time Deployment**: Developing a web application for live sentiment and emotion analysis.
- **Multimodal Analysis**: Combining text and audio inputs for enhanced emotion detection.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contributors

- António Frutuoso Frade
- Jasper Sänger
- João Filipe Alho Afonso
- Joaquim Firmino da Cunha Reis
  

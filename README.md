# 🫁 Chest X-Ray Pneumonia Classifier

A deep learning project that classifies chest X-ray images as **Normal** or **Pneumonia**. It compares a custom CNN built from scratch against transfer learning with **DenseNet121**, and deploys the better model as an interactive **Streamlit** web app.

> ⚠️ **Disclaimer:** This is an educational project. It is **not** a medical device and must not be used for diagnosis. Always consult a qualified medical professional.

---

## 📌 Project Overview

| | |
| :--- | :--- |
| **Task** | Binary image classification (Normal = 0, Pneumonia = 1) |
| **Framework** | TensorFlow / Keras |
| **Input size** | 256 × 256 |
| **Dataset** | [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia): 5,226 training, 16 validation and 624 test images |
| **Best model** | DenseNet121 (transfer learning) |
| **Deployment** | Streamlit web app |

---

## 📂 Repository Structure

```
├── Results/                          # Screenshots of results and the app
├── app.py                            # Streamlit web app
├── model_transfer.keras              # Trained DenseNet121 model used by the app
├── X_ray_Image_Classification.ipynb  # Data pipeline, model training and evaluation
└── README.md
```

---

## 🧠 Methodology

### 1. Data Pipeline
- Loaded the `train`, `val` and `test` folders with `image_dataset_from_directory` (labels inferred from folder names, batch size 32, images resized to 256 × 256).
- Normalized pixel values from `[0, 255]` to `[0, 1]`.
- Used **1-channel grayscale** input for the custom CNN, and converted images to **3-channel RGB** for DenseNet121, which expects the 3 channels of ImageNet.

### 2. Models Trained

**Model 1: Custom CNN**
- Light augmentation (random contrast and brightness) → 3 × (Conv2D + MaxPooling) with 64, 128 and 256 filters → Flatten → Dense(64) → Dropout(0.5) → Dense(1, sigmoid).
- About 51.4 million parameters, trained with Adam and binary cross-entropy.
- Early Stopping on validation loss (patience = 5, restores the best weights). Training stopped at epoch 23, and the weights of epoch 18 were kept.

**Model 2: DenseNet121 Transfer Learning** ⭐
- DenseNet121 pre-trained on ImageNet, used as a **frozen** feature extractor.
- Light augmentation → DenseNet121 → Flatten → Dense(128) → Dense(64) → Dropout(0.5) → Dense(1, sigmoid).
- Same optimizer, loss and Early Stopping settings. Training stopped at epoch 5, and the weights of epoch 1 were kept.

Both models were tracked with accuracy, precision, recall and AUC, not accuracy alone. In a screening task, **recall** (catching real pneumonia cases) matters as much as overall accuracy.

---

## 📊 Results

Evaluated on the **624-image test set**:

| Model | Accuracy | Precision | Recall | AUC |
| :--- | :---: | :---: | :---: | :---: |
| Custom CNN | 72.12% | 70.00% | 96.92% | 0.750 |
| **DenseNet121 Transfer Learning** | **81.09%** | **76.77%** | **100.00%** | **0.835** |

### Key Findings
- Transfer learning improved every metric compared with the custom CNN, with about 9 points more accuracy and a higher AUC.
- The DenseNet121 model reached **100% recall** on the test set, so it missed no pneumonia cases.
- Its precision of about 77% shows the trade-off: the model produces **false alarms** on healthy X-rays. Both models lean towards predicting "Pneumonia".
- Both models scored near 100% on the validation set but much lower on the test set (see the limitations below).



## 🚀 Run the App Locally

**1. Clone the repository**
```bash
git clone https://github.com/Donna152/<your-repo-name>.git
cd <your-repo-name>
```

**2. Install the dependencies**
```bash
pip install streamlit tensorflow numpy pandas
```

**3. Start the app** (run it from the repository root so `model_transfer.keras` is found)
```bash
streamlit run app.py
```

**4. Use it**
1. Upload a chest X-ray image (`.jpg`, `.jpeg` or `.png`).
2. Click **PREDICT**.
3. The app shows the predicted class, the confidence percentage and the raw model score.

### How the app makes a prediction
The uploaded image is resized to 256 × 256 RGB, preprocessed and passed to the model. The model outputs a single sigmoid score: a score below 0.5 means **Normal**, and 0.5 or above means **Pneumonia**.

---

## 🔁 Reproduce the Training

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) so that you have `train/`, `val/` and `test/` folders, each with `NORMAL/` and `PNEUMONIA/` subfolders.
2. Open `X_ray_Image_Classification.ipynb` in Jupyter, Google Colab or Kaggle.
3. Update the dataset paths in the notebook (it was run on Kaggle).
4. Run all cells to train both models and evaluate them on the test set.

---

## 🛠️ Tech Stack
Python · TensorFlow / Keras · DenseNet121 · NumPy · Matplotlib · OpenCV · Streamlit

---

## 👤 Author
**Donna Ahmed Mohamed**
[LinkedIn](https://www.linkedin.com/in/donna-ahmed-/) · [GitHub](https://github.com/Donna152)

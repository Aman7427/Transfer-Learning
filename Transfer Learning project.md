# Skin Lesion Segmentation using CNN with MobileNet Encoder

## 🧪 Course
**Deep Learning (CSL7590)**  
Indian Institute of Technology Jodhpur  
**Instructor:** Dr. Deepak Mishra  
**Submitted by:** Aman Kanshotia (M21MA201)  
📅 **Date:** March 19, 2024

---

## 📄 Project Summary

This project focuses on segmenting skin lesions from dermoscopic images using a Convolutional Neural Network (CNN) architecture. The encoder is based on **MobileNet (pretrained on ImageNet)** and the decoder is custom-built with transposed convolutions and dropout. Two models are compared:

- **Model 1:** Pretrained encoder with frozen weights  
- **Model 2:** Fine-tuned encoder with backpropagation

---

## 🧬 Dataset

- **Dataset:** ISIC 2016 Skin Lesion Dataset
- **Train Images:** 900  
- **Test Images:** 379  
- **Data Includes:** Images and segmentation masks

---

## 📊 Data Preprocessing

- Resized all images and masks to **128×128**
- Converted images and masks to **tensors**
- Applied necessary transforms using PyTorch's `Dataset` and `DataLoader` classes

---

## 🏗️ Model Architecture

- **Encoder:** MobileNet (Model 1 uses frozen encoder, Model 2 fine-tunes it)
- **Decoder:**
  - 4 Transposed Convolutional Layers
  - Dropout (p=0.5)
  - Final interpolation to restore output to 128×128

---

## ⚙️ Training Details

- **Epochs:** 25
- **Loss:** Binary Cross Entropy with Logits (BCEWithLogitsLoss)
- **Optimizer:** Nesterov Adam (`lr=0.001`)
- **Metrics:** IOU Score and Dice Coefficient
- **Device:** Trained on GPU

### 🚨 GPU Requirement
> **Note:** This project requires a **GPU** for efficient training. Training on CPU will be extremely slow and might cause memory issues.

```python
# Check GPU availability
import torch
print("CUDA Available:", torch.cuda.is_available())
```

---

## 📈 Results

### Model 1 (Frozen MobileNet)

- **Train Loss:** 0.4830  
- **Test Loss:** 0.3972  
- **IOU Score:** 0.2985  
- **Dice Score:** 0.7522

### Model 2 (Fine-Tuned MobileNet)

- **Train Loss:** 0.0927  
- **Test Loss:** 0.1578  
- **IOU Score:** 0.2985  
- **Dice Score:** 0.8620

---

## 📊 Comparison

- **Model 2** outperforms Model 1 in terms of loss and Dice Score.
- Freezing encoder in Model 1 led to poor learning and anomalies in loss/score trends.
- Fine-tuning the encoder in Model 2 improved segmentation quality significantly.

---

## 🔗 References

- [Google Colab Notebook](https://colab.research.google.com/drive/1x3NyjMtuv7OsZ9RuIeNPKpywiB_pIe8Z?usp=sharing)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Image Segmentation (Springer)](https://link.springer.com/article/10.1007/s10278-019-00227-x)
- [IoU and Dice Score Guide](https://www.v7labs.com/blog/intersection-over-union-guide)

---

## 🙌 Acknowledgements

- Special thanks to Dr. Deepak Mishra for course guidance and feedback.

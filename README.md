# Brain Tumor MRI Classifier

This app is a deep learning-powered tool for classifying brain MRI images into four categories: **glioma**, **meningioma**, **pituitary tumor**, or **no tumor**. It uses a custom convolutional neural network trained on public datasets and provides a simple web interface for predictions.

## Features

- Upload a brain MRI image and get instant predictions.
- Built with PyTorch and Gradio.
- Achieves high accuracy (F1-score ~97%) on validation data.

## Installation

1. **Clone the repository**  
   ```powershell
   git clone <your-repo-url>
   cd TumiroAI
   ```

2. **Create and activate a virtual environment**  
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies**  
   ```powershell
   pip install -r requirements.txt
   ```

## Usage

1. **Download the model**  
   Ensure `model.pth` is present in the `models/` directory. (Already included.)

2. **Run the app**  
   ```powershell
   python ui.py
   ```
   This will launch a Gradio web interface in your browser.

3. **Classify an MRI image**  
   - Click "Upload" and select a brain MRI image (JPG/PNG).
   - The app will predict one of: **glioma**, **meningioma**, **pituitary tumor**, or **no tumor**.

## Demo

**Placeholder images you should provide for the demo:**
- Example MRI image with a glioma tumor
- Example MRI image with a meningioma tumor
- Example MRI image with a pituitary tumor
- Example MRI image with no tumor

You can use anonymized sample images from public datasets or create your own synthetic examples.

**Demo:**

![Demo](./assets/brain_tumor_demo.mp4) 

> **Tip:** Place your demo images in the `assets/` folder and update the filenames above if needed.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

**References:**  
- Model architecture and training details are in [`brain-tumor-mri-classification-f1-score-97.ipynb`](brain-tumor-mri-classification-f1-score-97.ipynb).
- Model loading and prediction logic: [`utils.py`](utils.py)
- Gradio web interface: [`ui.py`](ui.py)

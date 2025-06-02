import gradio as gr
from utils import load_model, predict

# Load model once globally
model = load_model("./models/model.pth")

def classify_image(image):
    """
    Gradio wrapper for prediction.
    """
    return predict(model, image)

# Build the interface
interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=4),
    title="TumrioAi - Brain Tumor Classifier",
    description="Upload a brain MRI image and the model will predict: glioma, meningioma, pituitary tumor, or no tumor."
)

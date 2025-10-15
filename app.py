import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import os

# Load your trained model
model = YOLO('brain_tumor_best_model.pt')

def predict_brain_tumor(image):
    """Detect brain tumors from MRI image"""
    try:
        # Run prediction
        results = model.predict(image, conf=0.5)
        r = results[0]
        
        # Plot results on image
        output_image = r.plot()
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        
        # Count detections
        detection_count = len(r.boxes)
        tumor_types = {}
        
        if detection_count > 0:
            for box in r.boxes:
                class_name = model.names[int(box.cls[0])]
                tumor_types[class_name] = tumor_types.get(class_name, 0) + 1
        
        # Create result text
        if detection_count == 0:
            result_text = "✅ No tumors detected"
        else:
            result_text = f"🔍 Detected {detection_count} tumor(s):\n"
            for tumor_type, count in tumor_types.items():
                result_text += f"• {tumor_type.title()}: {count}\n"
            result_text += "\n🎯 Model Confidence: >50%"
        
        return output_image, result_text
    
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# Create the Gradio interface
with gr.Blocks(
    title="Brain Tumor Detection System",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:
    
    # Header
    gr.Markdown("""
    # 🧠 Brain Tumor Detection System
    ### AI-Powered MRI Analysis | 91.2% Accuracy
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📤 Upload MRI Scan")
            image_input = gr.Image(
                label="Drag & Drop MRI Image Here",
                type="filepath",
                height=300
            )
            gr.Markdown("**Supported formats:** JPG, PNG, JPEG")
            
        with gr.Column():
            gr.Markdown("### 🔍 Detection Results")
            image_output = gr.Image(
                label="Tumor Detection Visualization",
                height=300
            )
            text_output = gr.Textbox(
                label="Analysis Report",
                lines=4
            )
    
    # Auto-process when image is uploaded
    image_input.upload(
        fn=predict_brain_tumor,
        inputs=image_input,
        outputs=[image_output, text_output]
    )
    
    # Footer
    gr.Markdown("---")
    gr.Markdown("""
    **Medical Disclaimer:** This tool is for educational and research purposes. 
    Always consult healthcare professionals for medical diagnosis.
    
    **Model Performance:** 91.2% accuracy | Detects: Glioma, Meningioma, Pituitary
    """)

# Launch the application
if __name__ == "__main__":
    demo.launch()
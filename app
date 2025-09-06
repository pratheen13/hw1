import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline

st.title("Image Captioning & Keyword Tagging App")
st.write("Upload an image to see its AI-generated caption and keywords.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Load BLIP model & processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    # Prepare input and generate caption
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    alt_text = processor.decode(out, skip_special_tokens=True)
    st.markdown(f"**Generated Alt-Text:** {alt_text}")

    # Keyword Extraction (ViT)
    labeler = pipeline("image-classification", model="google/vit-base-patch16-224")
    # Save uploaded image for pipeline
    image_path = "temp_img.jpg"
    image.save(image_path)
    labels = labeler(image_path, top_k=5)
    keywords = [item['label'].lower() for item in labels if item['score'] > 0.3]
    st.markdown(f"**Extracted Keywords/Tags:** {', '.join(keywords)}")


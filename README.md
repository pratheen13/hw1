📸 Albumy / Moments (Customized Flask Photo Sharing App)

Albumy (a.k.a Moments) is a Flask-based photo-sharing app. This customized fork simplifies user access (no email/login required for uploads) and adds ML-powered features like alt-text generation, keyword extraction, and semantic search.

### Homepage
![Homepage Screenshot] [C:\Users\pprat\Desktop\moments\Screenshot (5).png] (https://drive.google.com/file/d/1MwW_Nzbo86lNOEg2ywWaYAaJZBLbVlGt/view?usp=sharing)

### Upload Page
![Upload Screenshot](C:\Users\pprat\Desktop\moments\Screenshot (6).png)

### Search Results
![Search Screenshot](C:\Users\pprat\Desktop\moments\Screenshot (7).png)

🚀 Features

User authentication with Flask-Login

Avatars via Flask-Avatars

Upload/manage photos (no email verification required ✅)

Optional anonymous uploads (login not required ✅)

Integrated ML models:

Auto-generate alt-text (captions) for images

Extract keywords for tagging & search

Semantic similarity search to find related images

⚙️ Setup Instructions (Windows)
1. Clone & Setup
git clone https://github.com/greyli/moments
cd moments
python -m venv venv
.\venv\Scripts\activate

2. Install Dependencies
python -m pip install --upgrade pip setuptools wheel
pip install flask==2.2.5
pip install -r requirements.txt
pip install torch torchvision torchaudio transformers pillow scikit-learn

3. Initialize DB
flask forge

4. Run
flask run


👉 Visit: http://127.0.0.1:5000

🧠 ML Integration
1. Auto Alt-Text for Uploaded Images

When a user uploads an image, generate a caption automatically:

# albumy/ml/caption.py
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)


Hook this into the photo upload route:

from albumy.ml.caption import generate_caption

@main_bp.route('/upload', methods=['GET', 'POST'])
def upload():
    # ... existing upload logic ...
    caption = generate_caption(file_path)
    photo.alt_text = caption
    db.session.commit()

2. Keyword Extraction for Search

Use a lightweight model to extract tags/keywords:

# albumy/ml/keywords.py
from transformers import pipeline

keyword_extractor = pipeline("feature-extraction", model="distilbert-base-uncased")

def extract_keywords(text: str, top_k=5):
    words = text.split()
    return words[:top_k]  # naive version — can replace with keyphrase model


Assign keywords during upload:

photo.keywords = ",".join(extract_keywords(photo.alt_text))
db.session.commit()

3. Semantic Search for Similar Images

Compute embeddings for captions/keywords:

# albumy/ml/embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str) -> np.ndarray:
    return embedder.encode(text)


Store embeddings in DB (as LargeBinary or JSON). At query time:

query_vec = get_embedding(user_query)
similarities = [np.dot(query_vec, get_embedding(photo.alt_text)) for photo in photos]


Sort results by similarity → return most relevant photos.

🔑 Customizations

Removed email verification → uploads available immediately.

Relaxed login requirement → optional login for uploads.

Added ML support → auto alt-text, keywords, semantic search.

📦 Tech Stack

Backend: Flask 2.2.5

Database: Flask-SQLAlchemy

Auth: Flask-Login

Uploads: Flask-Dropzone, Flask-Avatars

Search: Flask-Whooshee + ML embeddings

ML Models: Hugging Face Transformers, PyTorch, Sentence-Transformers

📌 Notes

Use Python 3.11 for stability

Keep Flask at 2.2.5 for compatibility

ML models require PyTorch (CPU or GPU)

Consider moderation if uploads are anonymous

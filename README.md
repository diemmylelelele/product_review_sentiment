# Product Review Sentiment (Multimodal)

AI system for sentiment analysis of niche e‑commerce product reviews (lip balm). Supports text and/or image input, retrieves similar examples, and predicts positive/neutral/negative using an LLM.

## System overview
- Ingests Amazon Reviews 2023 (All_Beauty) from Hugging Face and filters titles containing “lip balm”.
- Synthesizes labels from ratings: >3 positive, =3 neutral, <3 negative.
- Embeds text with SBERT (all-MiniLM-L6-v2, 384‑d) and images with CLIP ViT‑B/32 (512‑d); both L2‑normalized.
- Builds two FAISS indices (text.index, image.index) with JSON metadata for fast k-NN retrieval.
- Classifies via Ollama LLM (default granite3.2-vision).

## Report
Detailed report can be found [here](https://drive.google.com/file/d/1HA1x3jcnWxzX4ZewSNnvdCc2UA_fuCpm/view?usp=sharing)

## How to run

Python Environment
```powershell
# Clone repository
git clone https://github.com/diemmylelelele/product_review_sentiment.git
# Set Up a Python Virtual Environment
python -m venv venv
venv/Scrips/Activate
# Install dependencies
pip install -r requirements.txt
# Make sure to download Ollama on the machine and pull granite3.2-vision LLM model
# Run the app
python app.py
# Open http://127.0.0.1:5000
```

Docker Compose
```powershell
docker compose up --build

# (First run) pull model inside the Ollama container
docker exec -it prs-ollama ollama pull granite3.2-vision:latest
```

## Example Usage
This is the [demo video](https://drive.google.com/file/d/1WfZFhbY2cyVA1nnKhlVKpCHKmXD4K3Je/view?usp=sharing) 

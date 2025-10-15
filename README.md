# Product Review Sentiment (Multimodal)

AI system for sentiment analysis of niche e‑commerce product reviews (lip balm). Supports text and/or image input, retrieves similar examples, and predicts positive/neutral/negative using LLM.

## What it does
- Ingests Amazon Reviews 2023 (All_Beauty) from Hugging Face and filters titles containing “lip balm”.
- Synthesizes labels from ratings: >3 positive, =3 neutral, <3 negative.
- Embeds text with SBERT (all-MiniLM-L6-v2, 384‑d) and images with CLIP ViT‑B/32 (512‑d); both L2‑normalized.
- Builds two FAISS indices (text.index, image.index) with JSON metadata for fast k-NN retrieval.
- Classifies via Ollama LLM (default granite3.2-vision).

## How to run

Option A — Local (Windows/PowerShell)
```powershell
# Create venv and install
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# Ingest (downloads datasets)
python data_ingest.py

# Build FAISS indices
python embed.py

# Start Ollama in another terminal (for LLM classification)
ollama serve
ollama pull granite3.2-vision

# Run the app
python app.py
# Open http://127.0.0.1:5000
```

Option B — Docker Compose
```powershell
docker compose up --build

# App:     http://127.0.0.1:5000
# Ollama:  http://127.0.0.1:11434

# (First run) pull model inside the Ollama container
docker exec -it prs-ollama ollama pull gemma2:2b-instruct
```

Notes
- The app points to Ollama at `OLLAMA_BASE_URL` (default http://ollama:11434 in Compose; http://127.0.0.1:11434 locally). Set `OLLAMA_MODEL` to change models.
- Volumes: `artifacts/`, `uploads/`, and `data/` are mounted in Compose for persistence.

## Example Usage
This is the [demo video](https://drive.google.com/file/d/1WfZFhbY2cyVA1nnKhlVKpCHKmXD4K3Je/view?usp=sharing) 

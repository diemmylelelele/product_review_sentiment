import json, io, os, requests, uuid
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer
import open_clip
import faiss

DATA = Path("data/processed/lip_balm_reviews.jsonl")
OUT  = Path("artifacts/faiss")
OUT.mkdir(parents=True, exist_ok=True)

TXT_INDEX = OUT / "text.index"
IMG_INDEX = OUT / "image.index"
TXT_META  = OUT / "text_meta.json"
IMG_META  = OUT / "image_meta.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
# ---------- encoders ----------
txt_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model = clip_model.to(device)

def embed_text(s: str) -> np.ndarray:
    '''
    This function embeds a text string into a vector space using a pre-trained SentenceTransformer model.
    '''
    v = txt_model.encode([s or ""], normalize_embeddings=True)[0]  # cosine-ready
    return v.astype("float32")

@torch.no_grad()
def embed_image_from_url(url: str) -> Optional[np.ndarray]:
    '''
    This function fetches an image from a URL, processes it, and embeds it into a vector space using a pre-trained CLIP model.
    '''
    try:
        r = requests.get(url, timeout=8); r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        t = preprocess(img).unsqueeze(0).to(device)
        f = clip_model.encode_image(t)
        f /= f.norm(dim=-1, keepdim=True)  # cosine-ready
        return f[0].float().cpu().numpy()
    except Exception:
        return None

def main():
    '''
    This function reads a JSONL file containing product reviews, embeds the text and images,
    and builds FAISS indices for efficient similarity search.
    '''
    text_vecs, text_meta = [], []
    image_vecs, image_meta = [], []

    with open(DATA, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rid = str(uuid.uuid4())
            text = (r.get("text") or "").strip()
            img  = r.get("image_url") or None
            lab  = r.get("sentiment")

            payload = {"review_id": rid, "text": text or None, "image_url": img , "sentiment": lab}

            if text:
                text_vecs.append(embed_text(text))
                text_meta.append(payload)

            if img:
                v = embed_image_from_url(img)
                if v is not None:
                    image_vecs.append(v)
                    image_meta.append(payload)

    # FAISS indices
    if text_vecs:
        X = np.vstack(text_vecs).astype("float32") 
        idx_t = faiss.IndexFlatIP(X.shape[1])
        idx_t.add(X)
        faiss.write_index(idx_t, str(TXT_INDEX))
        with open(TXT_META, "w", encoding="utf-8") as f:
            json.dump(text_meta, f, ensure_ascii=False)
        print("Saved text index:", len(text_meta))

    if image_vecs:
        X = np.vstack(image_vecs).astype("float32")
        idx_i = faiss.IndexFlatIP(X.shape[1])
        idx_i.add(X)
        faiss.write_index(idx_i, str(IMG_INDEX))
        with open(IMG_META, "w", encoding="utf-8") as f:
            json.dump(image_meta, f, ensure_ascii=False)
        print("Saved image index:", len(image_meta))

if __name__ == "__main__":
    main()

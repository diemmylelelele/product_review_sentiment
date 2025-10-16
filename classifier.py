from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json, numpy as np
import os
import faiss
from sentence_transformers import SentenceTransformer
import open_clip, torch
from PIL import Image
import io, requests
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import re

# ---------- paths ----------
ART = Path("artifacts/faiss")
TXT_INDEX = ART / "text.index"
IMG_INDEX = ART / "image.index"
TXT_META  = ART / "text_meta.json"
IMG_META  = ART / "image_meta.json"

# ---------- encoders ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
txt_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model = clip_model.to(device)

# ---------- FAISS + meta, loaded on demand ----------
idx_t = None
idx_i = None
text_meta: List[Dict] = []
image_meta: List[Dict] = []

def _load_indices_if_needed():

    global idx_t, idx_i, text_meta, image_meta
    changed = False
    if idx_t is None and TXT_INDEX.exists():
        idx_t = faiss.read_index(str(TXT_INDEX))
        changed = True
    if idx_i is None and IMG_INDEX.exists():
        idx_i = faiss.read_index(str(IMG_INDEX))
        changed = True
    if (not text_meta) and TXT_META.exists():
        text_meta = json.loads(TXT_META.read_text("utf-8"))
        changed = True
    if (not image_meta) and IMG_META.exists():
        image_meta = json.loads(IMG_META.read_text("utf-8"))
        changed = True
    if changed:
        print("[faiss] indices/meta loaded")


def embed_text_q(s: str) -> np.ndarray:
    '''
    Embed a text string into a vector space using a pre-trained SentenceTransformer model.
    '''
    v = txt_model.encode([s or ""], normalize_embeddings=True)[0]
    return v.astype("float32").reshape(1, -1)

@torch.no_grad()
def embed_image_q(url: str) -> Optional[np.ndarray]:
    '''
    Fetch an image from a URL and embed it into a vector space using a pre-trained CLIP model.
    '''
    try:
        r = requests.get(url, timeout=8); r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        t = preprocess(img).unsqueeze(0).to(device)
        f = clip_model.encode_image(t)
        f /= f.norm(dim=-1, keepdim=True)
        return f.float().cpu().numpy()
    except Exception:
        return None

def search(index, vec: np.ndarray, k: int) -> Tuple[List[float], List[int]]:
    ''' 
    This function performs a FAISS search to find the top-k nearest neighbors for a given vector.
    '''
    if index is None: return [], []
    D, I = index.search(vec, k)
    return D[0].tolist(), I[0].tolist()

def build_neighbors(scores, ids, meta):
    '''
    This function constructs a list of neighbor dictionaries containing score and metadata for each neighbor.
    Input:
        scores: List of similarity scores from FAISS search.
        ids: List of indices corresponding to the scores.
    '''
    out = []
    for s, i in zip(scores, ids):
        if 0 <= i < len(meta):
            m = meta[i]
            out.append({
                "score": float(s),
                "text": m.get("text"),
                "image_url": m.get("image_url"),
                "sentiment": m.get("sentiment"),
            })
    return out

def merge(text_hits: List[Dict], image_hits: List[Dict], k: int) -> List[Dict]:
    '''
    This function merges text and image hits, normalizes their scores, and returns the top-k unique neighbors.
    Input: 
        text_hits: List of dictionaries containing text neighbor information.
        image_hits: List of dictionaries containing image neighbor information.
        k: The maximum number of unique neighbors to return.
    
    '''
    def norm(hs):
        if not hs: return []
        sc = np.array([h["score"] for h in hs], dtype="float32")
        sc = (sc - sc.min()) / (sc.max()-sc.min()+1e-8)
        for h, ns in zip(hs, sc): h["score_n"] = float(ns)
        return hs
    all_hits = norm(text_hits) + norm(image_hits)
    seen, merged = set(), []
    for h in sorted(all_hits, key=lambda x: x.get("score_n", 0.0), reverse=True):
        key = (h.get("text"), h.get("image_url"))
        if key in seen: continue
        seen.add(key); merged.append(h)
        if len(merged) == k: break
    return merged

def context(neighbors: List[Dict], k_limit: int = 6) -> str:
    '''
    This function constructs a context block string from the top-k_limit neighbors.
    Input: 
        neighbors: List of neighbor dictionaries containing review information.
        k_limit: The maximum number of neighbors to include in the context block.
    '''
    lines = []
    for i, n in enumerate(neighbors[:k_limit], 1):
        t = (n.get("text") or "").replace("\n", " ")[:200] 
        s = n.get("sentiment")
        im = n.get("image_url")
        line = f"[EX{i}] sentiment={s}; text={t}"
        if im: line += f"; image={im}"
        lines.append(line)
    return "\n".join(lines)

# ---------- LLM (LangChain + Ollama) ----------
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "granite3.2-vision:latest")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
os.environ.setdefault("OLLAMA_HOST", OLLAMA_BASE_URL)
llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.2, base_url=OLLAMA_BASE_URL) # use low temp for classification

PROMPT_TMPL = ChatPromptTemplate.from_template(
    """You are a sentiment classifier for LIP BALM product reviews.

Output:
- Return if the query is negative, neutral, or positive

QUERY:
{text_part}
IMAGE (if any):
{image_part}

SIMILAR_REVIEWS:
{context_block}
"""
)

LABELS = {"negative","neutral","positive"}

def classify_with_llm(query_text, query_image_url, context_block):
    '''
    This function classifies the sentiment of a review using a language model (LLM) with provided context.
    Input:
        query_text: The text of the review to classify.
        query_image_url: The URL of the image associated with the review (if any).
        context_block: A string containing context from similar reviews.
    Output:
        The predicted sentiment label as a string: "negative", "neutral", or "positive".
    '''
    text_part  = f"TEXT: {query_text[:1500]}" if query_text else "TEXT: <none>"
    image_part = f"IMAGE_URL: {query_image_url}" if query_image_url else "IMAGE_URL: <none>"

    prompt = PROMPT_TMPL.format_messages(
        text_part=text_part,
        image_part=image_part,
        context_block=context_block[:3000]
    )
    out = llm.invoke(prompt).strip().lower()

    return out

def classify(text: Optional[str]=None, image_url: Optional[str]=None, k: int = 5) -> Dict:
    '''
    This function reads a JSONL file containing product reviews, embeds the text and images,
    and builds FAISS indices for efficient similarity search.
    Input:
        text: The review text to classify.
        image_url: The URL of the review image to classify.
        k: The number of nearest neighbors to retrieve for context.
    '''
    _load_indices_if_needed()
    text_hits, image_hits = [], []
    if text:
        D, I = search(idx_t, embed_text_q(text), k)
        text_hits = build_neighbors(D, I, text_meta)
    if image_url:
        v = embed_image_q(image_url)
        if v is not None:
            D, I = search(idx_i, v, k)
            image_hits = build_neighbors(D, I, image_meta)

    neighbors = merge(text_hits, image_hits, k) # unique, sorted
    ctx = context(neighbors, k_limit=min(k, 3))  
    
    out = classify_with_llm(text, image_url, ctx)
    return {"neighbors": neighbors, "prediction": out}
    

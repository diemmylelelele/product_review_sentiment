from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import traceback

DATA_FILE = Path("data/processed/lip_balm_reviews.jsonl")
TXT_INDEX = Path("artifacts/faiss/text.index")
IMG_INDEX = Path("artifacts/faiss/image.index")

app = FastAPI(title="Ingest+Embed Service", version="1.0")


class EnsureResponse(BaseModel):
    data_exists: bool
    text_index_exists: bool
    image_index_exists: bool
    ran_data_ingest: bool
    ran_embed: bool
    message: str


def _run_data_ingest() -> None:
    # Import and run to avoid shelling out
    import data_ingest
    data_ingest.main()


def _run_embed() -> None:
    import embed
    embed.main()


def _status():
    return DATA_FILE.exists(), TXT_INDEX.exists(), IMG_INDEX.exists()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ensure", response_model=EnsureResponse)
def ensure():
    ran_ingest = False
    ran_embed = False
    msg = ""
    try:
        data_ok, txt_ok, img_ok = _status()
        if not data_ok:
            _run_data_ingest()
            ran_ingest = True
            data_ok, txt_ok, img_ok = _status()

        if not (txt_ok and img_ok):
            _run_embed()
            ran_embed = True
            data_ok, txt_ok, img_ok = _status()

        msg = "ready" if data_ok and txt_ok and img_ok else "incomplete"
        return EnsureResponse(
            data_exists=data_ok,
            text_index_exists=txt_ok,
            image_index_exists=img_ok,
            ran_data_ingest=ran_ingest,
            ran_embed=ran_embed,
            message=msg,
        )
    except Exception as e:
        tb = traceback.format_exc(limit=2)
        return EnsureResponse(
            data_exists=DATA_FILE.exists(),
            text_index_exists=TXT_INDEX.exists(),
            image_index_exists=IMG_INDEX.exists(),
            ran_data_ingest=ran_ingest,
            ran_embed=ran_embed,
            message=f"error: {e}; {tb}",
        )


@app.on_event("startup")
def on_startup():
    # Best-effort ensure at container boot
    try:
        ensure()
    except Exception:
        pass

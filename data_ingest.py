from datasets import load_dataset
import pandas as pd
import re
from pathlib import Path

# Precompile the regular expression for specific niche product category
LIP_BALM_PATTERN = re.compile(r"lip balm",flags=re.IGNORECASE)

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def data_fetching() -> pd.DataFrame:
    """
    Load Amazon All_Beauty reviews + metadata, keep lip-balm items,
    map rating->sentiment, and return a DataFrame with selected features and label.
    """
    # Load data from hugging face dataset
    reviews = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
    meta    = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_All_Beauty", split="full", trust_remote_code=True)

    df_rev = reviews["full"].to_pandas()[["rating", "text", "images", "parent_asin"]]
    df_meta = meta.select_columns(["parent_asin", "title"]).to_pandas().rename(columns={"title": "product_title"})

    # Merge reviews data and product metadata
    df = pd.merge(df_rev, df_meta, on="parent_asin", how="left", validate="m:1")

    # Filter lip-balm by product title based on the regular regression
    df = df[df["product_title"].str.contains(LIP_BALM_PATTERN, na=False)]

    # Label sentiment from rating: <3 neg, =3 neu, >3 pos
    df = df.assign(
        sentiment=pd.cut(
            df["rating"],
            bins=[-1, 2, 3, 5],
            labels=["negative", "neutral", "positive"],
            include_lowest=True,
            right=True,
        )
    )

    # Select useful columns (retain parent_asin for retrieval/dedup)
    df = df[["text", "images", "sentiment"]]
    # Create the image_url column
    df = df.reset_index(drop=True)
    df['image_url'] = None
    for i in df.index:
      if len(df.loc[i, 'images']) != 0:
          df.loc[i, 'image_url'] = df.loc[i, 'images'][0]['large_image_url']


    return df

def main():
    df = data_fetching()
    print("Fetched data sample...")
    print(f"Total samples of lip-balm products: {len(df)}")

    # Save the full dataframe to a JSONL file
    df.to_json(OUT_DIR / "lip_balm_reviews.jsonl", orient="records", lines=True)
    print(f"Saved full dataframe to {OUT_DIR / 'lip_balm_reviews.jsonl'}")

if __name__ == "__main__":
    main()




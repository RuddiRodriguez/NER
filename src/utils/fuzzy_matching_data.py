import pandas as pd
from rapidfuzz import process, fuzz

def normalize(s: str) -> str:
    return (s or "").strip()

data = pd.read_csv("/Users/ruddigarcia/Projects/ner/data/toy_hotel_ner_50.csv")
data.dropna(subset=["hotel_name","address","landmark","language"], how="any",  inplace=True)

# Keep hotel_names as a list for matching
hotel_names = data['hotel_name'].fillna("").tolist()
print(hotel_names)

def link_review_to_hotel(review_text: str, score_cutoff: int = 70, top_n: int = 3):
    matches = process.extract(
        query=review_text,
        choices=hotel_names,
        scorer=fuzz.token_set_ratio,
        score_cutoff=score_cutoff,
        limit=top_n
    )
    if not matches:
        return []
    
    candidates = []
    for match_name, score, idx in matches:
        row = data.iloc[idx]
        candidates.append({
            #"hotel_id": row.get("hotel_id", idx),
            "match_name": match_name,
            #"score": float(score),
            #"method": "rapidfuzz_name"
        })
    return candidates

links = data["description"].apply(lambda t: link_review_to_hotel(normalize(t)))

# Expand the list of candidates into separate columns
out = data.reset_index(drop=True).copy()
out["candidates"] = links

print(out.head(20))
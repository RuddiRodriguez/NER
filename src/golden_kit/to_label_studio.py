

import json
from pathlib import Path
import os
import sys 

repo_root = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(os.path.join(repo_root, "src"))


def to_labelstudio_tasks(rows) :
    """
    Produces Label Studio import tasks with preannotations.

    Assumes labeling config:
      <Labels name="label" toName="text"> ... </Labels>
      <Text name="text" value="$text"/>
    """
    tasks = []
    for r in rows:
        text = r["description"]
        ents = r.get("entities_pre", [])

        # Label Studio "result" objects
        result = []
        for i, e in enumerate(sorted(ents, key=lambda x: (x["start"], x["end"])), start=1):
            result.append({
                "id": f"r{i}",                 # region id (string)
                "from_name": "label",          # must match <Labels name="label">
                "to_name": "text",             # must match <Text name="text">
                "type": "labels",
                "value": {
                    "start": int(e["start"]),
                    "end": int(e["end"]),
                    "text": e.get("text", text[int(e["start"]):int(e["end"])]),
                    "labels": [e["type"]],
                },
            })

        task = {
            "data": {
                "text": text,
                "doc_id": r.get("doc_id"),
            },
            # Optional pre-annotations:
            "predictions": [{
                "model_version": "llm_preannot_v1",
                "score": 1.0,
                "result": result
            }] if result else []
        }
        tasks.append(task)

    return tasks


def read_jsonl(path: str) :
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path: str, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---- usage ----
in_jsonl = os.path.join(repo_root, "outputs", "toy_hotel_ner_50_extracted_noisy.jsonl")
out_json = Path(repo_root) / "outputs" / "toy_hotel_ner_50_labelstudio_tasks.json"

tasks = to_labelstudio_tasks(read_jsonl(in_jsonl))
out_json.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote: {out_json}")
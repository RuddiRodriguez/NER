import json
import os
import pandas as pd
from typing import Any, Dict, List, Optional
from openai import OpenAI
from dotenv import load_dotenv

import textwrap
from typing import Literal, TypedDict

load_dotenv(override=True)


api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
            print(
                "No API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!"
            )
            
if not api_key.startswith("sk-proj-"):
            print(
                "An API key was found, but it doesn't start sk-proj-; please check you're using the right key - see troubleshooting notebook"
            )
            
        
if api_key.strip() != api_key:
            print(
                "An API key was found, but it looks like it might have space or tab characters at the start or end - please remove them - see troubleshooting notebook"
            )
            

print("API key found and looks good so far!")






EntityType = Literal["HOTEL_NAME", "ADDRESS", "LANDMARK_POI"]

class Entity(TypedDict):
    start: int
    end: int
    type: EntityType
    text: str
    # Optional but useful for debugging / provenance
    candidate_id: str

class ExtractEntitiesArgs(TypedDict):
    entities: List[Entity]

class ExtractorPrompt:
   
   
   
    SYSTEM_MESSAGE = textwrap.dedent(
        """
        You are an information extraction engine that performs Named Entity Recognition (NER)
        on {language} hotel descriptions/reviews.

        Entity types:
        - HOTEL_NAME: the property name / brand name used to identify the hotel. Exclude generic "the hotel", "this property".
        - ADDRESS: a findability string (street/canal/square + number, postal code, city/region/country when used as address data).
        - LANDMARK_POI: named attractions, stations, airports, neighborhoods, venues, parks, squares, etc. Exclude generic "the station".

        Boundary rules:
        - Extract the smallest exact substring that uniquely identifies the entity.
        - No overlapping entities (prefer the longest specific span if conflicts exist).
        - Character offsets: start inclusive, end exclusive (Python slicing).
        - Every entity.text MUST equal original_text[start:end].

        Gazetteer hints may be provided. Use them ONLY as hints:
        - Prefer them when they appear verbatim in the text.
        - Do NOT output a hinted candidate unless it appears verbatim in the text.
        - You may still extract entities not in the hints.

        Output MUST be a function call only, matching the JSON schema.

    """
    ).strip()
   
    USER_MESSAGE = textwrap.dedent(
        """
    # Text to process
    {text}
    
            
        
        """).strip()
   
    TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "name": "extract_hotel_entities",
        "description": "Return named entities (HOTEL_NAME, ADDRESS, LANDMARK_POI) with character offsets.",
        # Structured Outputs guarantees schema adherence when strict=true. :contentReference[oaicite:3]{index=3}
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "integer"},
                            "end": {"type": "integer"},
                            "type": {"type": "string", "enum": ["HOTEL_NAME", "ADDRESS", "LANDMARK_POI"]},
                            "text": {"type": "string"},
                            "candidate_id": {"type": "string"},
                        },
                        "required": ["start", "end", "type", "text", "candidate_id"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["entities"],
            "additionalProperties": False,  # required for Structured Outputs tool schemas :contentReference[oaicite:4]{index=4}
        },
    }
]

    TOOL_CHOICE = {"type": "function", "name": "extract_hotel_entities"}











GPT_MODEL = "gpt-5.2-2025-12-11"
openai_client = OpenAI(
                api_key = api_key,
                
            )



import textwrap
from typing import Any, Dict, List, Literal, TypedDict


EntityType = Literal["HOTEL_NAME", "ADDRESS", "LANDMARK_POI"]

class Entity(TypedDict):
    start: int
    end: int
    type: EntityType
    text: str
    # Optional but useful for debugging / provenance
    candidate_id: str

class ExtractEntitiesArgs(TypedDict):
    entities: List[Entity]

class ExtractorPrompt:
   
   
   
    SYSTEM_MESSAGE = textwrap.dedent(
        """
        You are an information extraction engine that performs Named Entity Recognition (NER)
        on {language} hotel descriptions/reviews.

        Entity types:
        - HOTEL_NAME: the property name / brand name used to identify the hotel. Exclude generic "the hotel", "this property".
        - ADDRESS: a findability string (street/canal/square + number, postal code, city/region/country when used as address data).
        - LANDMARK_POI: named attractions, stations, airports, neighborhoods, venues, parks, squares, etc. Exclude generic "the station".
        - AMENITY: amenities/services/facilities and operational info mentioned in the description, e.g. "24h reception", "free Wi‑Fi",
          "breakfast included", "shower", "laundry", "parking", "pool", "gym", "check-in 24 hours", etc.

        Boundary rules:
        - Extract the smallest exact substring that uniquely identifies the entity.
        - No overlapping entities (prefer the longest specific span if conflicts exist).
        - Character offsets: start inclusive, end exclusive (Python slicing).
        - Every entity.text MUST equal original_text[start:end].

        Gazetteer hints may be provided. Use them ONLY as hints:
        - Prefer them when they appear verbatim in the text.
        - Do NOT output a hinted candidate unless it appears verbatim in the text.
        - You may still extract entities not in the hints.

        Output MUST be a function call only, matching the JSON schema.
        """
    ).strip()

    USER_MESSAGE = textwrap.dedent(
        """
        # Text to process
        {text}
        """
    ).strip()

    TOOLS: List[Dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "extract_hotel_entities",
                "description": "Return named entities (HOTEL_NAME, ADDRESS, LANDMARK_POI, AMENITY) with character offsets.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "start": {"type": "integer"},
                                    "end": {"type": "integer"},
                                    "type": {
                                        "type": "string",
                                        "enum": ["HOTEL_NAME", "ADDRESS", "LANDMARK_POI", "AMENITY"],
                                    },
                                    "text": {"type": "string"},
                                    "candidate_id": {"type": "string"},
                                },
                                "required": ["start", "end", "type", "text", "candidate_id"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["entities"],
                    "additionalProperties": False,  # required for Structured Outputs tool schemas
                },
            },
            # Structured Outputs guarantees schema adherence when strict=true.
            "strict": True,
        }
    ]

    TOOL_CHOICE = {
        "type": "function",
        "function": {"name": "extract_hotel_entities"},
    }








def build_messages(text: str, language: str) -> List[Dict[str, str]]:
    
    user_message_text = ExtractorPrompt.USER_MESSAGE.format(
        text = text
    )

    messages = []
    messages.append({"role": "system", "content": ExtractorPrompt.SYSTEM_MESSAGE.format(language=language)})
    messages.append({"role": "user", "content": user_message_text})
    return messages

def build_chat_params(messages):
    chat_params = {
        "model": GPT_MODEL,
        "messages": messages,
        "tools": ExtractorPrompt.TOOLS,
        "tool_choice": ExtractorPrompt.TOOL_CHOICE,
        "temperature": 0.0,
    }
    return chat_params

def _call_llm(**params):
    
        response = openai_client.chat.completions.create(**params)
        return response
    
data = pd.read_csv("/Users/ruddigarcia/Projects/ner/data/toy_hotel_ner_50.csv")
data.dropna(subset=["hotel_name","address","landmark","language"], how="any", inplace=True)
data = data.reset_index(drop=True)

def _validate_offsets(text: str, entities: List[Dict[str, Any]]) -> None:
    for ent in entities:
        s, e = int(ent["start"]), int(ent["end"])
        if not (0 <= s <= e <= len(text)):
            raise ValueError(f"Invalid offsets: {ent}")
        if text[s:e] != ent["text"]:
            raise ValueError(
                f"Offset mismatch: expected text[{s}:{e}]='{text[s:e]}' got '{ent['text']}'"
            )
            
from typing import Any, Dict, List

def repair_offsets(text: str, entities: List[Dict[str, Any]], window: int = 40) -> List[Dict[str, Any]]:
    """
    Returns a new list of entities with corrected (start,end) so that text[start:end] == ent['text'].
    Handles:
      - inclusive end bug (end should be +1)
      - small drift by searching near predicted span
    """
    fixed = []
    n = len(text)

    for ent in entities:
        s, e = int(ent["start"]), int(ent["end"])
        span = ent["text"]

        # Case 1: already correct (Python slice end-exclusive)
        if 0 <= s <= e <= n and text[s:e] == span:
            fixed.append({**ent, "start": s, "end": e})
            continue

        # Case 2: inclusive-end bug -> try end+1
        if 0 <= s <= e + 1 <= n and text[s:e+1] == span:
            fixed.append({**ent, "start": s, "end": e + 1})
            continue

        # Case 3: search locally near predicted region
        lo = max(0, s - window)
        hi = min(n, e + window)
        pos = text.find(span, lo, hi)
        if pos != -1:
            fixed.append({**ent, "start": pos, "end": pos + len(span)})
            continue

        # Case 4: last resort: search whole doc
        pos = text.find(span)
        if pos != -1:
            fixed.append({**ent, "start": pos, "end": pos + len(span)})
            continue

        # If we can't find the span text anywhere, it's a real model error
        raise ValueError(f"Span text not found in document: {ent}")

    return fixed

import json
from typing import Dict, Iterable, Iterator

def read_jsonl(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        

rows_out = []
for description , language, id in zip(data["description"], data["language"], data["id"]):
    messages = build_messages(
        text=description,
        language=language
    )
    chat_params = build_chat_params(messages)
    response = _call_llm(**chat_params)
    output_message = response.choices[0].message
    tool_call = output_message.tool_calls[0]
    parsed = json.loads(tool_call.function.arguments)
    ents = parsed.get("entities", [])
    ents = repair_offsets(description, ents)          # <-- add this
    _validate_offsets(description, ents)    
    
    
    rows_out.append({
        "doc_id": id,
        "description": description,
        "entities_pre": ents,
        
    })
path_jsonl_out = "/Users/ruddigarcia/Projects/ner/data/toy_hotel_ner_50_extracted.jsonl"
write_jsonl(path_jsonl_out, rows_out)




import json
from pathlib import Path
from typing import Dict, Iterator, List, Any

def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def to_labelstudio_tasks(rows: Iterator[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

# ---- usage ----
in_jsonl = "/Users/ruddigarcia/Projects/ner/data/toy_hotel_ner_50_extracted.jsonl"
out_json = Path("/Users/ruddigarcia/Projects/ner/data/labelstudio_tasks.json")

tasks = to_labelstudio_tasks(read_jsonl(in_jsonl))
out_json.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote: {out_json}")





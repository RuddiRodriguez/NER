

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
    
    # Hotel Name hints
    {hotel_name_hints}
    # LANDMARK_POI hints
    {landmark_hints}
    # ADDRESS hints
    {address_hints}
            
        
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
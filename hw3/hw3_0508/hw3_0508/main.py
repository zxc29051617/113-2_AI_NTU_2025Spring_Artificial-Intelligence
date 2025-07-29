from __future__ import annotations
from autogen import ConversableAgent, register_function
import os, sys, re, ast
from typing import Dict, List, get_type_hints

import json

# ────────────────────────────────────────────────────────────────
# 1. Utility data structures & helper functions
# ────────────────────────────────────────────────────────────────

SCORE_KEYWORDS: dict[str, int] = {
    "awful": 1, "horrible": 1, "horrid": 1, "disgusting": 1,
    "terrible": 1, "gross": 1, "foul": 1, "offensive": 1,
    "crummy": 1, "poor": 1, "lame": 1, "bland": 1,
    "subpar": 1, "yucky": 1, "appalling": 1, "trash": 1,
    "nasty": 1,
    "unpleasant": 2, "disinterested": 2, "unhelpful": 2,
    "slow": 2, "rude": 2, "forgettable": 2, "so-so": 2,
    "nothing special": 2, "uninspiring": 2, "mediocre": 2,
    "lackluster": 2, "fair": 2, "okay": 2, "passable": 2,
    "average": 3, "standard": 3, "decent": 3, "fine": 3,
    "regular": 3, "neutral": 3, "middling": 3,
    "good": 4, "enjoyable": 4, "satisfying": 4,
    "commendable": 4, "delightful": 4, "nice": 4, "pleasant": 4,
    "efficient": 4, "fresh": 4, "tasty": 4, "reliable": 4,
    "solid": 4, "attentive": 4, "quick": 4, "helpful": 4,
    "courteous": 4, "polite": 4,
    "awesome": 5, "amazing": 5, "incredible": 5,
    "fantastic": 5, "phenomenal": 5, "outstanding": 5,
    "superb": 5, "stellar": 5, "mind-blowing": 5,
    "blew my mind": 5, "god-tier": 5, "top-notch": 5,
    "wonderful": 5, "chef's kiss": 5, "legendary": 5,
    "peak": 5, "super": 5, "perfect": 5
}



# ────────────────────────────────────────────────────────────────
# 0. OpenAI API key setup ── *Do **not** modify this block.*
# ────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    sys.exit("❗ Set the OPENAI_API_KEY environment variable first.")
LLM_CFG = {"config_list": [{"model": "gpt-4o-mini", "api_key": OPENAI_API_KEY}]}

# ────────────────────────────────────────────────────────────────
# 1. Utility data structures & helper functions
# ────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()

def fetch_restaurant_data(restaurant_name: str) -> dict[str, list[str]]:
    data = {}
    target = normalize(restaurant_name)
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            name, review = line.split('.', 1)
            if normalize(name) == target:
                data.setdefault(name.strip(), []).append(review.strip())
    return data


def calculate_overall_score(restaurant_name: str, food_scores: List[int], customer_service_scores: List[int]) -> dict[str, str]:
    """Geometric-mean rating rounded to 3 dp."""
    n = len(food_scores)
    if n == 0 or n != len(customer_service_scores):
        raise ValueError("food_scores and customer_service_scores must be non-empty and same length")
    total = sum(((f**2 * s)**0.5) * (1 / (n * (125**0.5))) * 10 for f, s in zip(food_scores, customer_service_scores))
    return {restaurant_name: f"{total:.3f}"}

# register functions
fetch_restaurant_data.__annotations__ = get_type_hints(fetch_restaurant_data)
calculate_overall_score.__annotations__ = get_type_hints(calculate_overall_score)

# ──────────────────────────────────────────────
# 2. Agent setup
# ──────────────────────────────────────────────

def build_agent(name, msg):
    return ConversableAgent(name=name, system_message=msg, llm_config=LLM_CFG)

DATA_FETCH = build_agent(
    "fetch_agent",
    'Return JSON {"call":"fetch_restaurant_data","args":{"restaurant_name":"<name>"}}'
)
ANALYZER = build_agent(
    "review_analyzer_agent",
    (
        "You are given a dictionary: {Name: [list of reviews]}.\n"
        "Each review has exactly two parts: food description then service description.\n"
        "For each review, do these steps:\n"
        "1) Identify **one** adjective describing **food**.\n"
        "2) Identify **one** adjective describing **service**.\n"
        "3) Map each adjective to a score using this exact mapping:\n\n"
        f"{json.dumps(SCORE_KEYWORDS, indent=2)}\n\n"
        "If no listed adjective appears, assign score 3.\n"
        "All scores must be integers [1–5].\n"
        "Output **only** this JSON object, nothing else:\n"
        "{\n"
        '  "food_scores": [int, ...],\n'
        '  "customer_service_scores": [int, ...]\n'
        "}\n"
        "The i-th food_score and customer_service_score correspond to the i-th review.\n"
        "Example:\n"
        '{\n'
        '  "food_scores": [4, 3],\n'
        '  "customer_service_scores": [5, 2]\n'
        "}\n"
        "Any deviation from valid JSON in this exact shape is considered wrong."
    )
)



SCORER = build_agent(
    "scoring_agent",
    "Given name + two lists. Reply only: calculate_overall_score(...)"
)
ENTRY = build_agent("entry", "Coordinator")

# register functions
register_function(
    fetch_restaurant_data,
    caller=DATA_FETCH,
    executor=ENTRY,
    name="fetch_restaurant_data",
    description="Fetch reviews from specified data file by name.",
)
register_function(
    calculate_overall_score,
    caller=SCORER,
    executor=ENTRY,
    name="calculate_overall_score",
    description="Compute final rating via geometric mean.",
)


# ────────────────────────────────────────────────────────────────
# 3. Conversation helpers
# ────────────────────────────────────────────────────────────────

def run_chat_sequence(entry: ConversableAgent, sequence: list[dict]) -> str:
    ctx = {**getattr(entry, "_initiate_chats_ctx", {})}
    for step in sequence:
        msg = step["message"].format(**ctx)
        chat = entry.initiate_chat(
            step["recipient"], message=msg,
            summary_method=step.get("summary_method", "last_msg"),
            max_turns=step.get("max_turns", 2),
        )
        out = chat.summary
        # Data fetch output
        if step["recipient"] is DATA_FETCH:
            for past in reversed(chat.chat_history):
                try:
                    data = ast.literal_eval(past["content"])
                    if isinstance(data, dict) and data and not ("call" in data):
                        ctx.update({"reviews_dict": data, "restaurant_name": next(iter(data))})
                        break
                except:
                    continue
        # Analyzer output passed directly
        elif step["recipient"] is ANALYZER:
            ctx["analyzer_output"] = out
    return out

ConversableAgent.initiate_chats = lambda self, seq: run_chat_sequence(self, seq)

# ──────────────────────────────────────────────
# 4. Main entry
# ──────────────────────────────────────────────

def main(user_query: str, data_path: str = "restaurant-data.txt"):
    global DATA_PATH
    DATA_PATH = data_path
    agents = {"data_fetch": DATA_FETCH, "analyzer": ANALYZER, "scorer": SCORER}
    chat_sequence = [
        {"recipient": agents["data_fetch"], 
        "message": "Find reviews for this query: {user_query}", 
        "summary_method": "last_msg", 
        "max_turns": 2},

        {"recipient": agents["analyzer"], 
        "message": "Here are the reviews from the data fetch agent:\n{reviews_dict}\n\nExtract food and service scores for each review.", 
        "summary_method": "last_msg", 
        "max_turns": 1},

        {"recipient": agents["scorer"], 
        "message": "{analyzer_output}", 
        "summary_method": "last_msg", 
        "max_turns": 2},
    ]
    ENTRY._initiate_chats_ctx = {"user_query": user_query}
    result = ENTRY.initiate_chats(chat_sequence)
    print(f"result: {result}")
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python main.py path/to/data.txt "How good is Subway?" ')
        sys.exit(1)

    path = sys.argv[1]
    query = sys.argv[2]
    main(query, path)

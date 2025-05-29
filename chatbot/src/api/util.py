from functools import lru_cache
from typing import Dict, Any

import yaml

from src.config import AI_ACT_YAML_PATH

with open(AI_ACT_YAML_PATH, "r", encoding="utf-8") as f:
    ai_act_data = yaml.safe_load(f)

    ai_act_index = {
        item["id_elementa"]: item
        for section in ("cleni", "tocke")
        for item in ai_act_data.get(section, [])
    }


def format_sse(data: str, event: str = None) -> str:
    msg = ""
    if event:
        msg += f"event: {event}\n"
    msg += f"data: {data}\n\n"
    return msg


@lru_cache(maxsize=1024)
def get_ai_act_part_by_id(part_id: str) -> Dict[str, Any]:
    if part_id is None:
        return {}
    return ai_act_index.get(part_id, {})

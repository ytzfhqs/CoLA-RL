import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


def read_txt(txt_path: str, split="\n") -> List[str]:
    p = Path(txt_path).read_text(encoding="utf-8").strip().split(split)
    return [x.strip() for x in p if x]


def save_txt(txt_path: str, datas: List[str]):
    Path(txt_path).write_text("\n".join(datas))


def read_jsonl(
    jsonl_path: str, text_column: Optional[str] = None
) -> Iterator[Dict[str, Any]]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if text_column:
                yield json.loads(line)[text_column]
            else:
                yield json.loads(line)


def read_jsonl_offline(jsonl_path: str) -> List[Dict[str, str]]:
    json_data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            json_data.append(json.loads(line))
    return json_data


def read_json_offline(json_path: str) -> List[Dict[str, Any]]:
    json_data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return json_data


def save_json(path: str, datas: List[Dict[str, str]]):
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(datas, json_file, indent=2, ensure_ascii=False)


def save_jsonl(path: str, datas: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as file:
        for data in datas:
            json_line = json.dumps(data, ensure_ascii=False)
            file.write(json_line + "\n")

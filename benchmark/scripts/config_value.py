#!/usr/bin/env python3
"""读取 benchmark JSON 配置中的单个值。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def nested_value(document: dict[str, Any], dotted_key: str) -> Any:
    value: Any = document
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(dotted_key)
        value = value[key]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("key")
    parser.add_argument("--path", action="store_true", help="相对路径按项目根目录展开")
    parser.add_argument("--json", action="store_true", help="使用 JSON 格式输出")
    args = parser.parse_args()

    config_path = args.config.resolve()
    document = json.loads(config_path.read_text(encoding="utf-8"))
    value = nested_value(document, args.key)

    if args.path:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            project_root = config_path.parent.parent
            path = project_root / path
        value = str(path.resolve())

    if args.json or isinstance(value, (dict, list, bool)) or value is None:
        print(json.dumps(value, ensure_ascii=False))
    else:
        print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""JSON that stays readable: any object or array that fits within `width` columns is kept on one line."""
import json


def dumps_compact(obj, indent=2, width=100):
    def fmt(o, level):
        flat = json.dumps(o)
        if not isinstance(o, (dict, list)) or not o or len(flat) + level * indent <= width:
            return flat
        pad, end = " " * (indent * (level + 1)), " " * (indent * level)
        if isinstance(o, dict):
            body = ",\n".join(f"{pad}{json.dumps(k)}: {fmt(v, level + 1)}" for k, v in o.items())
            return "{\n" + body + "\n" + end + "}"
        return "[\n" + ",\n".join(pad + fmt(v, level + 1) for v in o) + "\n" + end + "]"
    return fmt(obj, 0)

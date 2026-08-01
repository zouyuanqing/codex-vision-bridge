#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Codex Vision Bridge - 交互式视觉原语 MCP Server
让纯文本模型（如 deepseek-v4-flash）通过 MCP 工具获得"看图 + 视觉原语"能力：
  describe / analyze / locate(输出坐标) / OCR(带坐标) / annotate(圈画) / crop(裁切) / zoom(放大) / health
视觉后端：小米 MiMo V2.5（OpenAI 兼容 /v1/chat/completions，图片走 data URL）
依赖：Python 3.10+、Pillow（本机已装 12.2.0）。零第三方运行时依赖。
环境变量：
  VISION_API_BASE     默认 https://api.xiaomimimo.com/v1
  VISION_API_KEY      必填
  VISION_MODEL        默认 mimo-v2.5（grounding 要求高可切 mimo-v2.5-pro）
  VISION_MAX_TOKENS   默认 4096（MiMo 推理型耗 token）
  VISION_TIMEOUT_S    默认 120
  VISION_MAX_IMAGE_MB 默认 20
  VISION_CACHE        默认 1；设 0 关闭缓存
  VISION_SAMPLES      默认 1；>1 时 locate/analyze 多次取样取坐标中位数
  VISION_OUTPUT_DIR   生成图片输出目录（默认本目录 generated/）
  VISION_DEBUG        设 1 输出日志到 stderr
"""
import base64
import hashlib
import io
import json
import math
import os
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:  # pragma: no cover
    HAS_PIL = False

# ----------------------------- 配置 -----------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

def _env(name, default=""):
    return os.environ.get(name, default).strip()

def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default

API_BASE = _env("VISION_API_BASE", "https://api.xiaomimimo.com/v1").rstrip("/")
API_KEY = _env("VISION_API_KEY")
MODEL = _env("VISION_MODEL", "mimo-v2.5")
MAX_TOKENS = _env_int("VISION_MAX_TOKENS", 4096)
TIMEOUT_S = _env_int("VISION_TIMEOUT_S", 120)
MAX_IMAGE_BYTES = _env_int("VISION_MAX_IMAGE_MB", 20) * 1024 * 1024
CACHE_ENABLED = _env("VISION_CACHE", "1") != "0"
CACHE_MAX_ENTRIES = _env_int("VISION_CACHE_MAX", 256)
CACHE_TTL_S = _env_int("VISION_CACHE_TTL_S", 7 * 24 * 3600)
SAMPLES = max(1, _env_int("VISION_SAMPLES", 1))
DEBUG = _env("VISION_DEBUG", "0") == "1"
OUTPUT_DIR = Path(_env("VISION_OUTPUT_DIR", str(SCRIPT_DIR / "generated")))
CACHE_DIR = Path(_env("VISION_CACHE_DIR", str(SCRIPT_DIR / ".cache")))
CACHE_FILE = CACHE_DIR / "vision-cache.json"
ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

AUX_VISION_SYSTEM = (
    "你是辅助视觉分析模型。请仔细分析用户提供的图像，并严格按要求的输出格式作答。"
    "不要提及隐藏的推理过程、内部工具，也不要提到会有另一个模型阅读你的回答。"
    "直接给出分析与结论。"
)

# ----------------------------- 基础工具 -----------------------------

class VisionError(Exception):
    """用户可见的工具错误。"""

class McpParamError(Exception):
    """参数校验错误 -> JSON-RPC -32602。"""

def log(*args):
    if DEBUG:
        print("[vision-bridge]", *args, file=sys.stderr, flush=True)

def load_image(src):
    """返回 (PIL.Image(RGB), 原始字节, 来源标签)。支持本地路径或 http(s) URL。"""
    if src.startswith(("http://", "https://")):
        req = urllib.request.Request(src, headers={"User-Agent": "codex-vision-bridge/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
        except urllib.error.URLError as e:
            raise VisionError(f"无法下载图片: {e.reason}")
        if len(data) > MAX_IMAGE_BYTES:
            raise VisionError(f"图片超过大小限制（{MAX_IMAGE_BYTES // (1024*1024)}MB）")
        label = src
    else:
        p = Path(src).expanduser()
        if not p.is_file():
            raise VisionError(f"图片文件不存在: {p}")
        if p.suffix.lower() not in ALLOWED_EXTS:
            raise VisionError(f"不支持的图片格式: {p.suffix}（支持: {', '.join(sorted(ALLOWED_EXTS))}）")
        size = p.stat().st_size
        if size > MAX_IMAGE_BYTES:
            raise VisionError(f"图片 {size // (1024*1024)}MB 超过大小限制（{MAX_IMAGE_BYTES // (1024*1024)}MB）")
        data = p.read_bytes()
        label = str(p.resolve())
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:
        raise VisionError(f"无法解析图片: {e}")
    return img.convert("RGB"), data, label

def encode_png(img):
    """统一转 PNG data URL（兼容各种输入格式）。返回 (data_url, bytes)。"""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw = buf.getvalue()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii"), raw

def _clamp_list(pts, w, h):
    """钳制 2 或 4 元素坐标到图像边界。返回 (钳制后列表, 是否发生钳制)。"""
    n = len(pts)
    changed = False
    out = []
    for i, v in enumerate(pts):
        limit = w if i % 2 == 0 else h
        cv = max(0, min(int(round(v)), limit))
        if cv != v:
            changed = True
        out.append(cv)
    if n == 4:
        x1, y1, x2, y2 = out
        if x1 > x2:
            x1, x2, changed = x2, x1, True
        if y1 > y2:
            y1, y2, changed = y2, y1, True
        out = [x1, y1, x2, y2]
    return out, changed

def to_pixel(value, w, h, coords):
    """把 box(4)/point(2) 从 pixel 或 norm(0-1000) 转像素并钳制。返回 (list, clamped)。"""
    if not isinstance(value, (list, tuple)) or len(value) not in (2, 4):
        raise VisionError(f"坐标必须是长度为 2 或 4 的数组，收到: {value!r}")
    coords = (coords or "pixel").lower()
    if coords == "norm":
        pts = [v * (w / 1000.0 if i % 2 == 0 else h / 1000.0) for i, v in enumerate(value)]
    elif coords == "pixel":
        pts = list(value)
    else:
        raise VisionError(f"coords 必须是 'pixel' 或 'norm'，收到: {coords!r}")
    return _clamp_list(pts, w, h)

def to_norm(value, w, h):
    """像素坐标 -> 0-1000 归一化。"""
    return [round(v * 1000.0 / (w if i % 2 == 0 else h)) for i, v in enumerate(value)]

def clamp_box(box, w, h):
    return _clamp_list(box, w, h)

def extract_json(text):
    """从模型输出中稳健提取 JSON 对象或数组。"""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    for pat in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pat, text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    raise VisionError(f"视觉模型未返回有效 JSON: {text[:400]}")

def _font(size):
    for p in [
        "C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/arialbd.ttf",
        "/System/Library/Fonts/PingFang.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()
# ----------------------------- 视觉后端调用 -----------------------------

def call_chat(messages, max_tokens=None, retries=1):
    """调用 OpenAI 兼容 chat/completions。返回文本 content。"""
    if not API_KEY:
        raise VisionError("未配置 VISION_API_KEY（视觉后端密钥）")
    url = f"{API_BASE}/chat/completions"
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens or MAX_TOKENS,
        "temperature": 0,
        "stream": False,
    }
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                resp = json.loads(r.read().decode("utf-8"))
            try:
                return resp["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError):
                raise VisionError(f"视觉后端响应异常: {json.dumps(resp, ensure_ascii=False)[:500]}")
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "ignore")[:500]
            except Exception:
                pass
            if e.code == 429 or 500 <= e.code < 600:
                if attempt < retries:
                    time.sleep(2 * (attempt + 1))
                    continue
            raise VisionError(f"视觉后端 HTTP {e.code}: {detail}")
        except urllib.error.URLError as e:
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            raise VisionError(f"无法连接视觉后端（已重试 {retries} 次）: {e.reason}")
        except TimeoutError:
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            raise VisionError(f"视觉后端请求超时（已重试 {retries} 次）")
        except VisionError:
            raise
        except Exception as e:
            last = e
    raise VisionError(f"视觉后端请求失败: {last}")

def image_message(text, img):
    """构造带图片的 user 消息；统一转 PNG data URL。"""
    data_url, _ = encode_png(img)
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }

# ----------------------------- 缓存 -----------------------------

_lock = threading.Lock()

def cache_get(key):
    if not CACHE_ENABLED:
        return None
    try:
        with _lock:
            data = json.loads(CACHE_FILE.read_text("utf-8")) if CACHE_FILE.exists() else {}
        entry = data.get("entries", {}).get(key)
        if not entry:
            return None
        if time.time() - entry.get("ts", 0) > CACHE_TTL_S:
            return None
        return entry.get("value")
    except Exception:
        return None

def cache_set(key, value):
    if not CACHE_ENABLED:
        return
    try:
        with _lock:
            data = json.loads(CACHE_FILE.read_text("utf-8")) if CACHE_FILE.exists() else {}
            entries = data.setdefault("entries", {})
            entries[key] = {"ts": time.time(), "value": value}
            if len(entries) > CACHE_MAX_ENTRIES:
                for k in sorted(entries, key=lambda k: entries[k].get("ts", 0))[: len(entries) - CACHE_MAX_ENTRIES]:
                    del entries[k]
            CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
    except Exception as e:
        log("cache write failed:", e)

def cache_key(img_bytes, tool, *parts):
    h = hashlib.sha256(img_bytes).hexdigest()
    return "|".join([h, tool, *[str(p) for p in parts], MODEL])

# ----------------------------- 视觉原语 -----------------------------

def normalize_primitives(raw_primitives, w, h, fmt="generic"):
    """把视觉模型输出的 primitives 规范化为统一结构（像素 + 归一化 + 钳制标记）。"""
    out = []
    for i, p in enumerate(raw_primitives or []):
        if not isinstance(p, dict):
            continue
        box = p.get("box") or p.get("bbox") or p.get("bbox_2d") or p.get("box_2d")
        point = p.get("point") or p.get("point_2d") or p.get("center")
        label = p.get("label") or p.get("ref") or p.get("text") or p.get("name") or f"item{i+1}"
        conf = p.get("confidence")
        entry = {
            "id": str(p.get("id") or f"v{i+1}"),
            "label": str(label).strip()[:96] or f"item{i+1}",
            "confidence": conf,
            "rotation": p.get("rotation") if p.get("rotation") is not None else p.get("angle"),
        }
        if fmt == "gemini" and isinstance(box, (list, tuple)) and len(box) == 4:
            ymin, xmin, ymax, xmax = box
            box = [xmin, ymin, xmax, ymax]
        if isinstance(box, (list, tuple)) and len(box) == 4:
            pts, clamped = to_pixel(box, w, h, "pixel")
            entry["type"] = "box"
            entry["box_pixel"] = pts
            entry["box_norm"] = to_norm(pts, w, h)
            entry["clamped"] = clamped
        elif isinstance(point, (list, tuple)) and len(point) == 2:
            pts, clamped = to_pixel(point, w, h, "pixel")
            entry["type"] = "point"
            entry["point_pixel"] = pts
            entry["point_norm"] = to_norm(pts, w, h)
            entry["clamped"] = clamped
        else:
            continue
        out.append(entry)
    return out

def median_primitives(batches):
    """Multi-sample aggregation: cluster by spatial center, median per cluster."""
    all_items = [p for batch in batches for p in batch]
    groups = []
    for p in all_items:
        c = _box_center(p)
        if c is None:
            groups.append([p])
            continue
        placed = False
        for g in groups:
            cg = _box_center(g[0])
            if cg is None:
                continue
            dx = abs(c[0] - cg[0])
            dy = abs(c[1] - cg[1])
            w = max(c[2], cg[2], 1.0)
            h = max(c[3], cg[3], 1.0)
            if dx <= max(40.0, w * 0.5) and dy <= max(40.0, h * 0.5):
                g.append(p)
                placed = True
                break
        if not placed:
            groups.append([p])
    from collections import Counter
    out = []
    for g in groups:
        base = dict(g[0])
        for field in ("box_pixel", "box_norm", "point_pixel", "point_norm", "confidence"):
            vals = [i.get(field) for i in g if i.get(field) is not None]
            if not vals:
                continue
            if isinstance(vals[0], list):
                base[field] = [round(statistics.median([v[j] for v in vals])) for j in range(len(vals[0]))]
            elif isinstance(vals[0], (int, float)):
                base[field] = round(statistics.median(vals), 3)
        labels = [i.get("label") for i in g if i.get("label")]
        if labels:
            base["label"] = Counter(labels).most_common(1)[0][0]
        out.append(base)
    return out

def _box_center(p):
    if p.get("box_pixel"):
        x1, y1, x2, y2 = p["box_pixel"]
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, float(x2 - x1), float(y2 - y1))
    if p.get("point_pixel"):
        x, y = p["point_pixel"]
        return (float(x), float(y), 0.0, 0.0)
    return None
PRIMITIVE_PROMPT = (
    "请定位并输出 JSON（不要输出任何其他文字，不要用代码块）:\n"
    '{"visual_primitives":[{"id":"v1","type":"box","label":"简短标签","box":[x1,y1,x2,y2],"confidence":0.9,"rotation":0}]}\n'
    "- box 为像素坐标 [x1,y1,x2,y2]（左上角、右下角），必须包含目标元件本体和全部引脚焊盘的完整边界框，图像实际宽 W px、高 H px，坐标必须在 0..W / 0..H 范围内。\n"
    "- 列出所有可疑目标（不只一个），每个目标一条。\n"
    "- rotation：若目标相对水平/垂直轴有旋转，估计旋转角度（度）；无旋转填 0。\n"
    "- 找不到目标时返回 {\"visual_primitives\":[]}。\n"
)
# ----------------------------- 工具实现 -----------------------------

def tool_describe_image(args):
    img, raw, label = load_image(args["image"])
    question = str(args.get("question") or "").strip()
    detail = str(args.get("detail") or "balanced").lower()
    if detail not in ("brief", "balanced", "detailed"):
        detail = "balanced"
    key = cache_key(raw, "describe", question, detail)
    hit = cache_get(key)
    if hit is not None:
        log("cache hit: describe")
        return hit
    prompt = f"请描述这张图像（宽 {img.width}px，高 {img.height}px）。"
    prompt += {"brief": "用 2-3 句话简要概括。", "balanced": "描述主要内容：布局、对象、文字、颜色。", "detailed": "尽可能详细地描述所有元素、文字内容、位置关系与颜色。"}[detail]
    if question:
        prompt += f"\n用户问题：{question}"
    text = call_chat([
        {"role": "system", "content": AUX_VISION_SYSTEM},
        image_message(prompt, img),
    ]).strip()
    cache_set(key, text)
    return text

def tool_analyze_image(args):
    img, raw, label = load_image(args["image"])
    question = str(args.get("question") or "").strip()
    fmt = str(args.get("format") or "generic").lower()
    if fmt not in ("generic", "gemini", "qwen"):
        raise VisionError(f"format 必须是 generic/gemini/qwen，收到: {fmt}")
    key = cache_key(raw, "analyze", question, fmt)
    hit = cache_get(key)
    if hit is not None:
        log("cache hit: analyze")
        return hit

    field = {"generic": "box", "gemini": "box_2d", "qwen": "bbox_2d"}[fmt]
    shape = {"generic": "box 为 [x1,y1,x2,y2] 像素坐标", "gemini": "box_2d 为 [ymin,xmin,ymax,xmax]，坐标 0-1000 归一化", "qwen": "bbox_2d 为 [x1,y1,x2,y2] 像素坐标"}[fmt]
    prompt = (
        f"分析这张图像（宽 {img.width}px，高 {img.height}px），输出 JSON（不要输出其他文字）:\n"
        '{"description":"对图像的整体描述","visual_primitives":[{"id":"v1","type":"box","label":"简短标签","%s":[0,0,0,0],"confidence":0.0}]}\n' % field
        + f"- {shape}。\n"
        + "- 列出重要对象/文字区域/按钮等，最多 12 个；没有框的就不输出。\n"
        + (f"- 用户关注点：{question}\n" if question else "")
        + "- confidence 为 0-1 的置信度；若目标有旋转，请添加 rotation 字段（估计角度，度）。"
    )
    batches = []
    for _ in range(SAMPLES):
        text = call_chat([
            {"role": "system", "content": AUX_VISION_SYSTEM},
            image_message(prompt, img),
        ])
        obj = extract_json(text)
        batches.append(normalize_primitives(obj.get("visual_primitives"), img.width, img.height, fmt))
    prims = median_primitives(batches) if SAMPLES > 1 else batches[0]
    result = {
        "description": str(obj.get("description") or "") if SAMPLES == 1 else "",
        "visual_primitives": prims,
        "image_size": [img.width, img.height],
    }
    if SAMPLES > 1:
        result["samples"] = SAMPLES
    cache_set(key, result)
    return result

def tool_locate_object(args):
    img, raw, label = load_image(args["image"])
    target = str(args.get("target") or "").strip()
    if not target:
        raise VisionError("缺少参数: target")
    coords = str(args.get("coords") or "pixel").lower()
    if coords not in ("pixel", "norm"):
        raise VisionError(f"coords 必须是 'pixel' 或 'norm'，收到: {coords}")
    key = cache_key(raw, "locate", target, coords)
    hit = cache_get(key)
    if hit is not None:
        log("cache hit: locate")
        return hit
    prompt = f"在图像（宽 {img.width}px，高 {img.height}px）中定位目标：{target}\n" + PRIMITIVE_PROMPT
    batches = []
    for _ in range(SAMPLES):
        text = call_chat([
            {"role": "system", "content": AUX_VISION_SYSTEM},
            image_message(prompt, img),
        ])
        obj = extract_json(text)
        batches.append(normalize_primitives(obj.get("visual_primitives"), img.width, img.height, "generic"))
    prims = median_primitives(batches) if SAMPLES > 1 else batches[0]
    for p in prims:
        if coords == "norm":
            p["box"] = p.get("box_norm")
            p["point"] = p.get("point_norm")
        else:
            p["box"] = p.get("box_pixel")
            p["point"] = p.get("point_pixel")
    result = {
        "target": target,
        "count": len(prims),
        "primitives": prims,
        "image_size": [img.width, img.height],
        "coords": coords,
    }
    if not prims:
        result["note"] = "视觉模型未找到该目标，请核对描述或换一种说法重试"
    cache_set(key, result)
    return result

def tool_ocr_image(args):
    img, raw, label = load_image(args["image"])
    language = str(args.get("language") or "auto")
    key = cache_key(raw, "ocr", language)
    hit = cache_get(key)
    if hit is not None:
        log("cache hit: ocr")
        return hit
    prompt = (
        f"请对这张图像（宽 {img.width}px，高 {img.height}px）做 OCR：提取图像中所有文字块，输出 JSON 数组（不要输出其他文字）:\n"
        '[{"text":"文字内容","box":[x1,y1,x2,y2]}]'
        f"\n- box 为像素坐标 [x1,y1,x2,y2]，图像宽 {img.width}px 高 {img.height}px。\n"
        "- 每行/每个独立文本块一条；没有文字返回 []。"
        + (f"\n- 语言提示：{language}" if language != "auto" else "")
    )
    text = call_chat([
        {"role": "system", "content": AUX_VISION_SYSTEM},
        image_message(prompt, img),
    ])
    rows = extract_json(text)
    if not isinstance(rows, list):
        raise VisionError(f"OCR 结果格式异常: {str(rows)[:300]}")
    items = []
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            continue
        box = r.get("box") or r.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        pts, clamped = to_pixel(box, img.width, img.height, "pixel")
        items.append({
            "text": str(r.get("text") or "").strip(),
            "box_pixel": pts,
            "box_norm": to_norm(pts, img.width, img.height),
            "confidence": r.get("confidence"),
            "clamped": clamped,
        })
    result = {"count": len(items), "items": items, "image_size": [img.width, img.height]}
    cache_set(key, result)
    return result

def _resolve_out_path(out_path):
    if not out_path:
        return None
    p = Path(out_path).expanduser()
    try:
        p = p.resolve()
    except Exception:
        p = Path(str(p))
    out_root = OUTPUT_DIR.resolve()
    if not (p == out_root or out_root in p.parents):
        raise VisionError(f"out_path 必须位于输出目录内: {OUTPUT_DIR}")
    return p

def _unique_path(prefix):
    ts = time.strftime("%Y%m%d-%H%M%S")
    return OUTPUT_DIR / f"{prefix}_{ts}_{os.getpid()}.png"

def tool_annotate_image(args):
    img, raw, label = load_image(args["image"])
    items = args.get("items")
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list) or not items:
        raise VisionError("items 必须是标注项数组（或单个对象）")
    coords = str(args.get("coords") or "pixel").lower()
    if coords not in ("pixel", "norm"):
        raise VisionError(f"coords 必须是 'pixel' 或 'norm'，收到: {coords}")
    style = args.get("style") or {}
    if not isinstance(style, dict):
        raise VisionError("style 必须是对象")
    out_path = _resolve_out_path(args.get("out_path")) or _unique_path("annotate")
    draw = ImageDraw.Draw(img)
    lw = max(1, int(style.get("line_width") or 3))
    fs = int(style.get("font_size") or max(14, img.width // 50))
    default_color = str(style.get("color") or "#ff3b30")
    fnt = _font(fs)
    clamped_any = False
    count = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        color = str(item.get("color") or default_color)
        box = item.get("box")
        point = item.get("point")
        label_text = str(item.get("label") or "").strip()
        if box is not None:
            pts, clamped = to_pixel(box, img.width, img.height, coords)
            clamped_any = clamped_any or clamped
            x1, y1, x2, y2 = pts
            draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
            if label_text:
                tb = draw.textbbox((0, 0), label_text, font=fnt)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
                ly = max(0, y1 - th - 6)
                draw.rectangle([x1, ly, x1 + tw + 8, ly + th + 6], fill=color)
                draw.text((x1 + 4, ly + 2), label_text, fill="white", font=fnt)
            count += 1
        elif point is not None:
            pts, clamped = to_pixel(point, img.width, img.height, coords)
            clamped_any = clamped_any or clamped
            px, py = pts
            r = max(6, lw + 3)
            draw.ellipse([px - r, py - r, px + r, py + r], fill=color)
            if label_text:
                tb = draw.textbbox((0, 0), label_text, font=fnt)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
                draw.rectangle([px + r + 2, py - th // 2 - 3, px + r + 2 + tw + 8, py + th // 2 + 3], fill=color)
                draw.text((px + r + 6, py - th // 2 - 1), label_text, fill="white", font=fnt)
            count += 1
    img.save(out_path, "PNG")
    return {
        "path": str(out_path),
        "image_size": [img.width, img.height],
        "annotations": count,
        "clamped": clamped_any,
        "note": "clamped=true 表示部分坐标超出图像边界已被自动钳制" if clamped_any else "",
    }

def tool_crop_image(args):
    img, raw, label = load_image(args["image"])
    box = args.get("box")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise VisionError("box 必须是 [x1,y1,x2,y2]（长度 4 的数组）")
    coords = str(args.get("coords") or "pixel").lower()
    if coords not in ("pixel", "norm"):
        raise VisionError(f"coords 必须是 'pixel' 或 'norm'，收到: {coords}")
    expand = int(args.get("expand_px") or 0)
    if expand < 0:
        raise VisionError("expand_px 不能为负数")
    out_path = _resolve_out_path(args.get("out_path")) or _unique_path("crop")
    pts, clamped = to_pixel(box, img.width, img.height, coords)
    if expand:
        x1, y1, x2, y2 = pts
        pts = [x1 - expand, y1 - expand, x2 + expand, y2 + expand]
        pts, clamped2 = _clamp_list(pts, img.width, img.height)
        clamped = clamped or clamped2
    x1, y1, x2, y2 = pts
    if x2 - x1 < 1 or y2 - y1 < 1:
        raise VisionError(f"裁切区域为空或过小: {pts}")
    region = img.crop((x1, y1, x2, y2))
    region.save(out_path, "PNG")
    return {
        "path": str(out_path),
        "box_used": pts,
        "size": [region.width, region.height],
        "expand_px": expand,
        "clamped": clamped,
    }

def tool_zoom_region(args):
    img, raw, label = load_image(args["image"])
    coords = str(args.get("coords") or "pixel").lower()
    if coords not in ("pixel", "norm"):
        raise VisionError(f"coords 必须是 'pixel' 或 'norm'，收到: {coords}")
    scale = int(args.get("scale") or 2)
    if not 1 <= scale <= 8:
        raise VisionError("scale 必须是 1-8 的整数")
    out_path = _resolve_out_path(args.get("out_path")) or _unique_path("zoom")
    box = args.get("box")
    if box is None:
        region = img
        pts = [0, 0, img.width, img.height]
        clamped = False
    else:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise VisionError("box 必须是 [x1,y1,x2,y2]（长度 4 的数组）")
        pts, clamped = to_pixel(box, img.width, img.height, coords)
        x1, y1, x2, y2 = pts
        if x2 - x1 < 1 or y2 - y1 < 1:
            raise VisionError(f"放大区域为空或过小: {pts}")
        region = img.crop((x1, y1, x2, y2))
    region = region.resize((region.width * scale, region.height * scale), Image.LANCZOS)
    region.save(out_path, "PNG")
    return {
        "path": str(out_path),
        "box_used": pts,
        "scale": scale,
        "size": [region.width, region.height],
        "clamped": clamped,
    }

def tool_vision_health(args=None):
    problems = []
    if not API_KEY:
        problems.append("VISION_API_KEY 未配置")
    if not API_BASE:
        problems.append("VISION_API_BASE 未配置")
    if not HAS_PIL:
        problems.append("Pillow 未安装（无法执行 annotate/crop/zoom）")
    backend = None
    try:
        req = urllib.request.Request(f"{API_BASE}/models", headers={"Authorization": f"Bearer {API_KEY}"})
        with urllib.request.urlopen(req, timeout=20) as r:
            backend = r.status
            data = json.loads(r.read().decode("utf-8"))
        ids = [m.get("id") for m in data.get("data", [])] if isinstance(data, dict) else []
        if MODEL not in ids:
            problems.append(f"模型 {MODEL} 不在后端模型列表: {', '.join(ids)}")
    except Exception as e:
        problems.append(f"后端连通性探测失败: {e}")
        backend = None
    return {
        "ok": not problems,
        "api_base": API_BASE,
        "model": MODEL,
        "backend_status": backend,
        "problems": problems,
        "pillow": HAS_PIL,
        "cache_enabled": CACHE_ENABLED,
        "samples": SAMPLES,
        "output_dir": str(OUTPUT_DIR),
        "max_image_mb": MAX_IMAGE_BYTES // (1024 * 1024),
    }






# ----------------------------- 虚拟标注推理（annotate_infer） -----------------------------

ANNOT_TYPES = ("box", "point", "line", "arrow", "circle")

def _parse_annot_item(item, w, h):
    """解析标注项 -> (type, label, color, geometry_pixel)。geometry_pixel: dict。"""
    if not isinstance(item, dict):
        raise VisionError("标注项必须是对象")
    typ = str(item.get("type") or "box").lower()
    if typ not in ANNOT_TYPES:
        raise VisionError(f"不支持的标注类型: {typ}（支持: {', '.join(ANNOT_TYPES)}）")
    label = str(item.get("label") or item.get("id") or "").strip()
    color = str(item.get("color") or "#ff3b30")
    coords = str(item.get("coords") or "pixel").lower()
    if coords not in ("pixel", "norm"):
        raise VisionError(f"coords 必须是 'pixel' 或 'norm'，收到: {coords!r}")

    def px(v, n):
        if not isinstance(v, (list, tuple)) or len(v) != n:
            raise VisionError(f"标注几何长度应为 {n}: {v!r}")
        if coords == "norm":
            return [round(float(v[0]) * w / 1000), round(float(v[1]) * h / 1000)] if n == 2 else                    [round(float(v[0]) * w / 1000), round(float(v[1]) * h / 1000),
                    round(float(v[2]) * w / 1000), round(float(v[3]) * h / 1000)]
        return [round(float(x)) for x in v]

    if typ == "box":
        box = px(item.get("box"), 4)
        box, _ = _clamp_list(box, w, h)
        return typ, label, color, {"box": box}
    if typ == "point":
        pt = px(item.get("point"), 2)
        pt, _ = _clamp_list(pt, w, h)
        return typ, label, color, {"point": pt}
    if typ in ("line", "arrow"):
        frm = px(item.get("from"), 2)
        to = px(item.get("to"), 2)
        frm, _ = _clamp_list(frm, w, h)
        to, _ = _clamp_list(to, w, h)
        return typ, label, color, {"from": frm, "to": to}
    if typ == "circle":
        c = px(item.get("center"), 2)
        c, _ = _clamp_list(c, w, h)
        try:
            r = int(item.get("radius") or 20)
        except (TypeError, ValueError):
            r = 20
        r = max(1, min(r, w, h))
        return typ, label, color, {"center": c, "radius": r}
    raise VisionError(f"未实现的标注类型: {typ}")

def _annot_to_text(typ, label, color, geo):
    name = label or {"box": "框", "point": "点", "line": "连线", "arrow": "箭头连线", "circle": "圆"}[typ]
    if typ == "box":
        b = geo["box"]
        return f"{name}：框 [({b[0]},{b[1]}) -> ({b[2]},{b[3]})]（左上到右下，像素坐标）"
    if typ == "point":
        p = geo["point"]
        return f"{name}：点 ({p[0]},{p[1]})"
    if typ in ("line", "arrow"):
        f, t = geo["from"], geo["to"]
        return f"{name}：{'箭头连线' if typ == 'arrow' else '连线'} 从 ({f[0]},{f[1]}) 到 ({t[0]},{t[1]})"
    if typ == "circle":
        c, r = geo["center"], geo["radius"]
        return f"{name}：圆 圆心 ({c[0]},{c[1]}) 半径 {r}px"
    return ""

def _draw_annot_overlay(draw, typ, label, color, geo, lw, fnt):
    if typ == "box":
        b = geo["box"]
        draw.rectangle(b, outline=color, width=lw, fill=color + "33")
        if label:
            tb = draw.textbbox((0, 0), label, font=fnt)
            draw.rectangle([b[0], max(0, b[1] - (tb[3] - tb[1]) - 6), b[0] + (tb[2] - tb[0]) + 8, b[1]], fill=color)
            draw.text((b[0] + 4, max(0, b[1] - (tb[3] - tb[1]) - 4)), label, fill="white", font=fnt)
    elif typ == "point":
        p = geo["point"]
        r = max(6, lw + 3)
        draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=color)
    elif typ in ("line", "arrow"):
        f, t = geo["from"], geo["to"]
        draw.line([f, t], fill=color, width=lw)
        if typ == "arrow":
            import math as _m
            ang = _m.atan2(t[1] - f[1], t[0] - f[0])
            hlen = 12
            for da in (0.5, -0.5):
                draw.line([t, (round(t[0] - hlen * _m.cos(ang + da)), round(t[1] - hlen * _m.sin(ang + da)))], fill=color, width=lw)
    elif typ == "circle":
        c, r = geo["center"], geo["radius"]
        draw.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], outline=color, width=lw, fill=color + "22")

def tool_annotate_infer(args):
    src = args["image"]
    items = args.get("items")
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list) or not items:
        raise VisionError("items 必须是标注数组（或单个对象）")
    question = str(args.get("question") or "").strip()
    if not question:
        raise VisionError("缺少参数: question（推理问题）")
    mode = str(args.get("mode") or "virtual").lower()
    if mode not in ("virtual", "overlay"):
        raise VisionError("mode 必须是 'virtual' 或 'overlay'")
    try:
        alpha = float(args.get("alpha") or 0.35)
    except (TypeError, ValueError):
        alpha = 0.35
    if not 0 < alpha <= 1:
        raise VisionError("alpha 必须在 (0, 1] 之间")

    img, raw, label = load_image(src)
    w, h = img.size
    parsed = [_parse_annot_item(it, w, h) for it in items]
    descs = [_annot_to_text(*p) for p in parsed]
    annot_text = "\n".join(descs)

    key = cache_key(raw, "annotate_infer", mode, question, annot_text, str(alpha))
    hit = cache_get(key)
    if hit is not None:
        log("cache hit: annotate_infer")
        return hit

    base_prompt = (
        "请分析这张图像，并结合以下**虚拟标注**（这些标注不是图像上的实际内容，"
        "仅用于指示位置、区域和关系；请以标注为参考进行空间推理，并区分图像实际内容与标注）：\n"
        + annot_text + f"\n推理问题：{question}\n请结合图像内容与标注关系给出分析。"
    )

    if mode == "virtual":
        text = call_chat([
            {"role": "system", "content": AUX_VISION_SYSTEM},
            image_message(base_prompt, img),
        ]).strip()
        out = {"mode": "virtual", "answer": text, "annotations": len(items)}
    else:
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        lw = max(2, min(6, w // 300))
        fnt = _font(max(12, w // 60))
        for typ, lbl, color, geo in parsed:
            _draw_annot_overlay(od, typ, lbl, color, geo, lw, fnt)
        overlay = overlay.point(lambda px: px)  # no-op 保持类型
        import math  # noqa: F401
        alpha_img = overlay
        # 应用 alpha 缩放（通过色值强度近似：填充色已带 alpha hex，这里控制整体）
        if alpha < 1.0:
            a_layer = alpha_img.getchannel("A").point(lambda a: round(a * alpha))
            alpha_img = alpha_img.copy()
            alpha_img.putalpha(a_layer)
        composite = Image.alpha_composite(img.convert("RGBA"), alpha_img).convert("RGB")
        out_path = _unique_path("annotate_infer")
        composite.save(out_path, "PNG")
        prompt = (
            "这张图像上已叠加半透明标注层（框/点/线/圆，叠加不会遮挡原图内容）。"
            + base_prompt
        )
        text = call_chat([
            {"role": "system", "content": AUX_VISION_SYSTEM},
            image_message(prompt, composite),
        ]).strip()
        out = {
            "mode": "overlay",
            "answer": text,
            "annotations": len(items),
            "overlay_path": str(out_path),
            "alpha": alpha,
        }
    cache_set(key, out)
    return out

# ----------------------------- 多图对比（compare_images） -----------------------------

def tool_compare_images(args):
    images = args.get("images")
    if not isinstance(images, list) or not (2 <= len(images) <= 4):
        raise VisionError("images 必须是 2-4 张图片（本地路径或 http(s) URL）的数组")
    question = str(args.get("question") or "").strip()
    detail = str(args.get("detail") or "balanced").lower()
    if detail not in ("brief", "balanced", "detailed"):
        detail = "balanced"
    imgs, raws = [], []
    for src in images:
        img, raw, label = load_image(src)
        imgs.append(img)
        raws.append(raw)
    key = cache_key(b"|".join(raws), "compare", question, detail)
    hit = cache_get(key)
    if hit is not None:
        log("cache hit: compare")
        return hit
    names = "、".join(f"图{i+1}" for i in range(len(imgs)))
    prompt = (
        f"请对比分析以下 {len(imgs)} 张图像（编号：{names}）。逐项对比："
        "1) 整体内容与布局；2) 相同点；3) 差异点（文字、元素、颜色、位置、状态等，尽量具体）；4) 结论/判断。\n"
    )
    prompt += {"brief": "简要回答，每项 1-2 句。", "balanced": "每项给出要点即可。", "detailed": "尽可能详细，逐条列出差异。"}[detail]
    if question:
        prompt += f"\n用户关注点（重点回答）：{question}"
    content = [{"type": "text", "text": prompt}]
    for img in imgs:
        data_url, _ = encode_png(img)
        content.append({"type": "image_url", "image_url": {"url": data_url}})
    text = call_chat([
        {"role": "system", "content": AUX_VISION_SYSTEM},
        {"role": "user", "content": content},
    ]).strip()
    cache_set(key, text)
    return text

# ----------------------------- 异常元件扫描（scan_anomalies） -----------------------------

def _build_tiles(region, tile_size, overlap, max_tiles):
    """把 region 切成带重叠的块。返回 [(x0,y0,x1,y1), ...]。"""
    x0, y0, x1, y1 = region
    rw, rh = x1 - x0, y1 - y0
    if tile_size and tile_size > 0:
        cols = max(1, min(max_tiles, math.ceil(rw / tile_size)))
        rows = max(1, min(max_tiles // cols if cols else 1, math.ceil(rh / tile_size)))
    else:
        ratio = rw / max(rh, 1)
        if ratio >= 2.0:
            cols, rows = max(1, min(max_tiles, max(1, round(rw / max(rh, 1))))), 1
        elif ratio <= 0.5:
            cols, rows = 1, max(1, min(max_tiles, max(1, round(rh / max(rw, 1)))))
        else:
            cols = rows = max(1, int(math.sqrt(max_tiles)))
            while cols * rows > max_tiles:
                cols -= 1
            cols = max(1, cols)
            rows = max(1, min(rows, max_tiles // cols))
    step_x = rw / cols if cols > 1 else rw
    step_y = rh / rows if rows > 1 else rh
    tiles = []
    for r in range(rows):
        for c in range(cols):
            bx0 = x0 + round(c * step_x)
            by0 = y0 + round(r * step_y)
            bx1 = x0 + round((c + 1) * step_x) if c < cols - 1 else x1
            by1 = y0 + round((r + 1) * step_y) if r < rows - 1 else y1
            if cols > 1 and c > 0:
                bx0 = max(x0, bx0 - overlap)
            if rows > 1 and r > 0:
                by0 = max(y0, by0 - overlap)
            if cols > 1 and c < cols - 1:
                bx1 = min(x1, bx1 + overlap)
            if rows > 1 and r < rows - 1:
                by1 = min(y1, by1 + overlap)
            if bx1 - bx0 >= 200 and by1 - by0 >= 200:
                tiles.append((bx0, by0, bx1, by1))
    return tiles

def _box_center_xy(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

def _boxes_close(a, b, min_dist=150):
    ca, cb = _box_center_xy(a), _box_center_xy(b)
    if abs(ca[0] - cb[0]) <= min_dist and abs(ca[1] - cb[1]) <= min_dist:
        return True
    # IoU 判定
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return False
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(union, 1) >= 0.2

def _merge_candidates(cands):
    merged = []
    for c in cands:
        box = c["box"]
        placed = False
        for m in merged:
            if _boxes_close(box, m["box"]):
                m["box"] = [
                    min(box[0], m["box"][0]), min(box[1], m["box"][1]),
                    max(box[2], m["box"][2]), max(box[3], m["box"][3]),
                ]
                m["confidence"] = max(m.get("confidence") or 0, c.get("confidence") or 0)
                if c.get("label"):
                    m["labels"].append(c["label"])
                if c.get("tile"):
                    m["tiles"].append(c["tile"])
                placed = True
                break
        if not placed:
            merged.append({
                "box": box,
                "confidence": c.get("confidence"),
                "labels": [c.get("label")] if c.get("label") else [],
                "tiles": [c.get("tile")] if c.get("tile") else [],
            })
    return merged

VERIFY_PROMPT = (
    "请客观检查这张 PCB 局部放大图，只描述你实际看到的，不要猜测：\n"
    "1) 图中是否存在明显歪斜/旋转摆放的元件（相对图像水平或垂直轴明显偏转，且与周边元件方向不一致）？回答 是/否。\n"
    "2) 若有，旋转角度约多少度？\n"
    "3) 该元件的丝印字符是什么（如无则写：无）？\n"
    "4) 元件类型（三极管/MOS管/LDO稳压器/电阻/电容/其他）？\n"
    "如果没有任何明显歪斜的元件，直接回答：没有明显歪斜元件。"
)

def _parse_verdict(text):
    t = text or ""
    verdict = "unclear"
    neg = re.search(r"没有明显歪斜|未发现歪斜|不存在歪斜|无歪斜|没有歪斜|没有明显|没有可识别|无法判断", t)
    pos = re.search(r"(歪斜|倾斜|旋转|偏转)", t) or re.search(r"^\s*1\)\s*是", t, re.M)
    if neg:
        verdict = "not_skewed"
    elif pos or re.search(r"\d+\s*(?:°|度)", t):
        verdict = "skewed"
    rotation = None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:°|度)", t)
    if m:
        try:
            rotation = float(m.group(1))
        except ValueError:
            rotation = None
    silkscreen = None
    m = re.search(r"(?:丝印|字符|marking)[：:为是]?\s*([A-Za-z0-9_\-]{1,16})", t, re.I)
    if not m:
        m = re.search(r"^\s*\d+\)\s*([A-Za-z0-9_\-]{1,16})\s*$", t, re.M)
    if m:
        silkscreen = m.group(1)
    etype = None
    for kw in ("LDO", "三极管", "MOS", "稳压", "电阻", "电容", "电感"):
        if kw in t:
            etype = kw
            break
    return {"verdict": verdict, "rotation": rotation, "silkscreen": silkscreen, "component_type": etype}

def tool_scan_anomalies(args):
    src = args["image"]
    target = str(args.get("target") or "摆放歪斜、方向与周边不一致的元件").strip()
    verify = args.get("verify", True)
    if not isinstance(verify, bool):
        verify = True
    max_tiles = max(1, min(12, int(args.get("max_tiles") or 6)))
    overlap = max(0, int(args.get("overlap") or 250))
    tile_size = max(0, int(args.get("tile_size") or 0))
    img, raw, label = load_image(src)
    W, H = img.size
    region = args.get("region")
    if region:
        if not isinstance(region, (list, tuple)) or len(region) != 4:
            raise VisionError("region 必须是 [x1,y1,x2,y2]")
        region, _ = _clamp_list([int(v) for v in region], W, H)
        if region[2] - region[0] < 200 or region[3] - region[1] < 200:
            raise VisionError("region 过小（至少 200x200 像素）")
    else:
        region = [0, 0, W, H]

    tiles = _build_tiles(region, tile_size, overlap, max_tiles)
    tmp_dir = CACHE_DIR / "scan_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    raw_cands = []
    try:
        for i, tbox in enumerate(tiles):
            crop = img.crop(tbox)
            tp = tmp_dir / f"tile_{os.getpid()}_{i}.png"
            crop.save(tp, "PNG")
            log("scan tile", i, tbox)
            try:
                loc = tool_locate_object({"image": str(tp), "target": target})
            finally:
                try:
                    tp.unlink()
                except OSError:
                    pass
            for pr in loc.get("primitives", []):
                if pr.get("box"):
                    raw_cands.append({
                        "tile": str(i),
                        "label": pr.get("label"),
                        "confidence": pr.get("confidence"),
                        "rotation": pr.get("rotation"),
                        "box": [tbox[0] + pr["box"][0], tbox[1] + pr["box"][1], tbox[0] + pr["box"][2], tbox[1] + pr["box"][3]],
                    })
        merged = _merge_candidates(raw_cands)
    finally:
        try:
            for f in tmp_dir.glob(f"tile_{os.getpid()}_*.png"):
                f.unlink()
        except OSError:
            pass

    out = []
    for m in merged:
        entry = {
            "box": m["box"],
            "box_norm": [to_norm([m["box"][0], m["box"][1]], W, H)[0], to_norm([m["box"][0], m["box"][1]], W, H)[1],
                          to_norm([m["box"][2], m["box"][3]], W, H)[0], to_norm([m["box"][2], m["box"][3]], W, H)[1]],
            "confidence": m.get("confidence"),
            "labels": m.get("labels") or [],
            "tiles": m.get("tiles") or [],
            "verified": None,
        }
        if verify:
            pad = max(150, int((m["box"][2] - m["box"][0]) * 0.3))
            b = (max(0, m["box"][0] - pad), max(0, m["box"][1] - pad),
                 min(W, m["box"][2] + pad), min(H, m["box"][3] + pad))
            crop = img.crop(b)
            if max(crop.size) < 600:
                crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
            vp = tmp_dir / f"verify_{os.getpid()}_{len(out)}.png"
            crop.save(vp, "PNG")
            try:
                text = tool_describe_image({"image": str(vp), "question": VERIFY_PROMPT, "detail": "balanced"})
            finally:
                try:
                    vp.unlink()
                except OSError:
                    pass
            entry["verified"] = _parse_verdict(text)
            entry["verified"]["raw"] = text[:400]
        out.append(entry)

    # 排序：歪斜的排前面
    def rank(e):
        v = e.get("verified") or {}
        return 0 if v.get("verdict") == "skewed" else (1 if v.get("verdict") == "unclear" else 2)
    out.sort(key=rank)

    result = {
        "target": target,
        "region": region,
        "image_size": [W, H],
        "tiles": len(tiles),
        "candidates_found": len(raw_cands),
        "candidates_merged": len(out),
        "candidates": out,
    }
    if not out:
        result["note"] = "未找到候选目标，可尝试：换一种 target 描述、缩小 region、增大 max_tiles"
    return result

HANDLERS = {
    "describe_image": tool_describe_image,
    "analyze_image": tool_analyze_image,
    "locate_object": tool_locate_object,
    "ocr_image": tool_ocr_image,
    "annotate_image": tool_annotate_image,
    "crop_image": tool_crop_image,
    "zoom_region": tool_zoom_region,
    "vision_health": tool_vision_health,
    "scan_anomalies": tool_scan_anomalies,
    "compare_images": tool_compare_images,
    "annotate_infer": tool_annotate_infer,
}

TOOLS = [
    {
        "name": "describe_image",
        "description": "用视觉模型描述图片内容，返回文字描述。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image": {"type": "string", "description": "本地图片绝对路径或 http(s) 图片 URL"},
                "question": {"type": "string", "description": "可选的针对性问题"},
                "detail": {"type": "string", "enum": ["brief", "balanced", "detailed"], "description": "细节程度，默认 balanced"},
            },
            "required": ["image"],
        },
    },
    {
        "name": "analyze_image",
        "description": "结构化分析：返回 description + visual_primitives（box/point 坐标与标签）。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image": {"type": "string"},
                "question": {"type": "string", "description": "可选的关注点"},
                "format": {"type": "string", "enum": ["generic", "gemini", "qwen"], "description": "primitives 字段风格，默认 generic"},
            },
            "required": ["image"],
        },
    },
    {
        "name": "locate_object",
        "description": "在图片中定位目标对象，返回坐标 primitives（让 LLM 输出坐标）。找不到会返回 count=0。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image": {"type": "string"},
                "target": {"type": "string", "description": "要定位的目标，如：蓝色提交按钮 / 红色圆形 / 报错文字"},
                "coords": {"type": "string", "enum": ["pixel", "norm"], "description": "返回坐标单位：pixel（默认，像素）或 norm（0-1000 归一化）"},
            },
            "required": ["image", "target"],
        },
    },
    {
        "name": "ocr_image",
        "description": "OCR 提取图片中所有文字块，返回 text + bbox（像素与归一化坐标）。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image": {"type": "string"},
                "language": {"type": "string", "description": "语言提示，默认 auto"},
            },
            "required": ["image"],
        },
    },
    {
        "name": "annotate_image",
        "description": "在图片上画矩形框/圆点/标签（圈画标记），保存标注图并返回路径。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image": {"type": "string"},
                "items": {"description": "标注项数组或单个对象：[{label, box 或 point, color}]，box=[x1,y1,x2,y2]，point=[x,y]"},
                "coords": {"type": "string", "enum": ["pixel", "norm"], "description": "坐标单位，默认 pixel"},
                "out_path": {"type": "string", "description": "输出路径（必须位于输出目录内），默认自动命名"},
                "style": {"type": "object", "description": "{line_width, font_size, color}"},
            },
            "required": ["image", "items"],
        },
    },
    {
        "name": "crop_image",
        "description": "按坐标裁切图片（可边缘外扩），保存并返回路径与新尺寸。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image": {"type": "string"},
                "box": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4, "description": "[x1,y1,x2,y2]"},
                "coords": {"type": "string", "enum": ["pixel", "norm"], "description": "坐标单位，默认 pixel"},
                "expand_px": {"type": "integer", "description": "四边外扩像素数，默认 0"},
                "out_path": {"type": "string"},
            },
            "required": ["image", "box"],
        },
    },
    {
        "name": "zoom_region",
        "description": "放大图片指定区域（默认整图 2 倍），保存并返回路径。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image": {"type": "string"},
                "box": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4, "description": "[x1,y1,x2,y2]，省略则放大整图"},
                "coords": {"type": "string", "enum": ["pixel", "norm"], "description": "坐标单位，默认 pixel"},
                "scale": {"type": "integer", "description": "放大倍数 1-8，默认 2"},
                "out_path": {"type": "string"},
            },
            "required": ["image"],
        },
    },
    {
        "name": "vision_health",
        "description": "检查视觉后端配置与连通性。",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "annotate_infer",
        "description": "虚拟标注 + 增强图形推理：把框/点/连线/箭头/圆等标注（不修改原图）注入视觉模型，引导空间关系推理。mode=virtual 用坐标文本注入；mode=overlay 生成半透明叠加图。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image": {"type": "string"},
                "items": {"description": "标注数组或单个对象：[{type: box|point|line|arrow|circle, label, color, coords, box/point/from/to/center/radius}]"},
                "question": {"type": "string", "description": "推理问题，如：框A中的元件是什么？A到B的连线代表什么连接关系？"},
                "mode": {"type": "string", "enum": ["virtual", "overlay"], "description": "virtual=坐标文本注入（默认，原图零修改）；overlay=半透明叠加图"},
                "alpha": {"type": "number", "description": "overlay 模式叠加透明度 (0,1]，默认 0.35"},
                "detail": {"type": "string", "enum": ["brief", "balanced", "detailed"], "description": "细节程度"},
            },
            "required": ["image", "items", "question"],
        },
    },
    {
        "name": "compare_images",
        "description": "多图对比分析（2-4 张）：A/B 截图对比、设计稿一致性、多帧分析，返回逐项对比结果。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "images": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 4, "description": "2-4 张本地图片路径或 http(s) URL"},
                "question": {"type": "string", "description": "对比重点，如：UI 有什么变化"},
                "detail": {"type": "string", "enum": ["brief", "balanced", "detailed"], "description": "细节程度，默认 balanced"},
            },
            "required": ["images"],
        },
    },
    {
        "name": "scan_anomalies",
        "description": "自动扫描图片中的异常/歪斜元件：把区域切成带重叠的块逐块定位候选，再从原图高清裁切逐个验证，输出带置信度与角度/丝印的报告。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image": {"type": "string"},
                "target": {"type": "string", "description": "要找的异常特征描述，默认：摆放歪斜、方向与周边不一致的元件"},
                "region": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4, "description": "可选：限定扫描区域 [x1,y1,x2,y2] 像素，默认全图"},
                "verify": {"type": "boolean", "description": "是否自动高清验证候选，默认 true"},
                "max_tiles": {"type": "integer", "description": "切块数上限（1-12），默认 6"},
                "overlap": {"type": "integer", "description": "切块重叠像素，默认 250"},
                "tile_size": {"type": "integer", "description": "切块边长（像素），默认自动"},
            },
            "required": ["image"],
        },
    },
]

# ----------------------------- MCP stdio 协议 -----------------------------

def write_frame(stream, obj):
    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    stream.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("utf-8"))
    stream.write(payload)
    stream.flush()

def read_frame(stream):
    headers = {}
    while True:
        line = stream.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        if b":" in line:
            k, v = line.split(b":", 1)
            headers[k.strip().lower()] = v.strip()
    try:
        length = int(headers.get(b"content-length", b"0"))
    except ValueError:
        length = 0
    if length <= 0:
        return None
    payload = stream.read(length)
    try:
        return json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError:
        return None

def _text_result(text, is_error=False):
    return {"result": {"content": [{"type": "text", "text": text}], "isError": is_error}}

def _error(rid, code, message):
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}

def handle_message(msg):
    if not isinstance(msg, dict):
        return None
    method = msg.get("method")
    rid = msg.get("id")
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "codex-vision-bridge", "version": "1.0.0"},
            },
        }
    if method == "notifications/initialized":
        return None
    if method == "ping":
        return {"jsonrpc": "2.0", "id": rid, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        tool = next((t for t in TOOLS if t["name"] == name), None)
        if tool is None:
            return _error(rid, -32601, f"未知工具: {name}")
        required = tool["inputSchema"].get("required", [])
        missing = [r for r in required if r not in args or args[r] in (None, "")]
        if missing:
            return _error(rid, -32602, f"缺少必需参数: {', '.join(missing)}")
        handler = HANDLERS.get(name)
        if handler is None:
            return _error(rid, -32601, f"工具未实现: {name}")
        try:
            result = handler(args)
            if isinstance(result, str):
                return _text_result(result)
            return _text_result(json.dumps(result, ensure_ascii=False, indent=2))
        except McpParamError as e:
            return _error(rid, -32602, f"Invalid params: {e}")
        except VisionError as e:
            return _text_result(f"错误: {e}", is_error=True)
        except Exception as e:
            log("tool crash:", name, e)
            return _text_result(f"内部错误: {e}", is_error=True)
    if rid is not None:
        return _error(rid, -32601, f"未知方法: {method}")
    return None

def main():
    if "--health" in sys.argv:
        print(json.dumps(tool_vision_health(), ensure_ascii=False, indent=2))
        return
    while True:
        msg = read_frame(sys.stdin.buffer)
        if msg is None:
            break
        resp = handle_message(msg)
        if resp is not None:
            write_frame(sys.stdout.buffer, resp)

if __name__ == "__main__":
    main()

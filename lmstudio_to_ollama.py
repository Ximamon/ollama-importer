#!/usr/bin/env python3
"""LM Studio GGUF -> Ollama import assistant.

Supports two modes:
- CLI batch mode (flags-driven)
- Interactive terminal mode (--tui) using curses
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import curses
except Exception:  # pragma: no cover
    curses = None  # type: ignore[assignment]

DEFAULT_LMSTUDIO_DIR = Path("~/.lmstudio/models").expanduser()
DEFAULT_OUTPUT_DIR = Path("./ollama_imports")
DEFAULT_PREFERRED_QUANTS = ["q5_k_m", "q4_k_m", "q4_k_s", "q8_0", "q6_k", "f16", "fp16"]
SEMANTIC_TAGS = ["instruct", "coder", "vision", "chat", "reasoning", "embed", "multilingual"]
NOISE_TOKENS = {
    "gguf",
    "model",
    "models",
    "quant",
    "quantized",
    "ml",
    "studio",
}


@dataclass
class VariantRecord:
    source_path: Path
    source_path_abs: Path
    exists: bool
    publisher: Optional[str]
    model_family: Optional[str]
    filename: str
    stem: str
    quant_guess: Optional[str]
    tags_detected: List[str]
    group_key: str = ""
    selected_in_group: bool = False
    selection_reason: str = ""
    proposed_ollama_name: str = ""
    modelfile_path: Optional[Path] = None
    readme_path: Optional[Path] = None
    generated_at: Optional[str] = None
    create_attempted: bool = False
    create_success: Optional[bool] = None
    create_command: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    warnings: List[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: Optional[str] = None
    selected_by_user: bool = False
    overrides_applied: Dict[str, Any] = field(default_factory=dict)
    effective_parameters: Dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "source_path_abs": str(self.source_path_abs),
            "exists": self.exists,
            "publisher": self.publisher,
            "model_family": self.model_family,
            "filename": self.filename,
            "stem": self.stem,
            "quant_guess": self.quant_guess,
            "tags_detected": self.tags_detected,
            "group_key": self.group_key,
            "selected_in_group": self.selected_in_group,
            "selected_by_user": self.selected_by_user,
            "selection_reason": self.selection_reason,
            "overrides_applied": self.overrides_applied,
            "effective_parameters": self.effective_parameters,
            "proposed_ollama_name": self.proposed_ollama_name,
            "modelfile_path": str(self.modelfile_path) if self.modelfile_path else None,
            "readme_path": str(self.readme_path) if self.readme_path else None,
            "generated_at": self.generated_at,
            "create_attempted": self.create_attempted,
            "create_success": self.create_success,
            "create_command": self.create_command,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "warnings": self.warnings,
        }


@dataclass
class Summary:
    detected_models: int = 0
    grouped_models: int = 0
    groups_detected: int = 0
    variants_processed: int = 0
    selected_variants: int = 0
    modelfiles_generated: int = 0
    created_ok: int = 0
    created_failed: int = 0
    omitted: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "models_detected": self.detected_models,
            "groups_detected": self.groups_detected,
            "variants_grouped": self.grouped_models,
            "variants_processed": self.variants_processed,
            "selected": self.selected_variants,
            "modelfiles_generated": self.modelfiles_generated,
            "created_ok": self.created_ok,
            "created_failed": self.created_failed,
            "omitted": self.omitted,
        }


@dataclass
class DiscoveryResult:
    variants: List[VariantRecord]
    groups: Dict[str, List[VariantRecord]]
    selected_map: Dict[str, VariantRecord]
    preferred_quants: List[str]
    summary: Summary


@dataclass
class VariantJob:
    variant: VariantRecord
    requested_name: Optional[str] = None
    final_name: str = ""
    selected_by_user: bool = False
    overrides_applied: Dict[str, Any] = field(default_factory=dict)
    effective_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactResult:
    summary: Summary
    index_path: Path
    selected_path: Path
    batch_path: Path
    processed_metadata: List[Dict[str, Any]]
    selected_metadata: List[Dict[str, Any]]


@dataclass
class GlobalConfig:
    preferred_quants: List[str]
    prefer_small: bool = False
    prefer_quality: bool = False
    name_mode: str = "variant"
    prefix: str = ""
    suffix: str = ""
    num_ctx: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    system: Optional[str] = None
    only: Optional[str] = None
    exclude: Optional[str] = None
    limit: Optional[int] = None
    output_dir: Path = DEFAULT_OUTPUT_DIR
    force: bool = False


@dataclass
class ModelOverride:
    enabled: bool = True
    variant_index: int = 0
    selected_source_path: Optional[str] = None
    name_override: Optional[str] = None
    num_ctx: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    system: Optional[str] = None


@dataclass
class InteractiveSelection:
    global_config: GlobalConfig
    lmstudio_root: Path
    groups: Dict[str, List[VariantRecord]] = field(default_factory=dict)
    group_order: List[str] = field(default_factory=list)
    overrides: Dict[str, ModelOverride] = field(default_factory=dict)
    discovery_summary: Summary = field(default_factory=Summary)
    status: str = ""
    last_result: Optional[ArtifactResult] = None


class Console:
    def __init__(self, verbose: bool, json_mode: bool):
        self.verbose = verbose
        self.json_mode = json_mode

    def info(self, msg: str) -> None:
        if not self.json_mode:
            print(msg)

    def debug(self, msg: str) -> None:
        if self.verbose and not self.json_mode:
            print(msg)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect LM Studio GGUF models and prepare Ollama import artifacts."
    )
    parser.add_argument(
        "--lmstudio-dir",
        default=str(DEFAULT_LMSTUDIO_DIR),
        help="LM Studio models directory (default: ~/.lmstudio/models)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for generated artifacts (default: ./ollama_imports)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite generated files if they already exist")
    parser.add_argument("--tui", action="store_true", help="Run interactive terminal UI (curses)")

    dry_create_group = parser.add_mutually_exclusive_group()
    dry_create_group.add_argument("--dry-run", action="store_true", help="Do not execute ollama create (default)")
    dry_create_group.add_argument("--create", action="store_true", help="Execute ollama create for prepared variants")

    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument("--selected-only", action="store_true", help="Process only selected variant per group (default)")
    selection_group.add_argument("--all-variants", action="store_true", help="Process all detected variants")

    parser.add_argument("--only", default=None, help="Include only entries containing this case-insensitive text")
    parser.add_argument("--exclude", default=None, help="Exclude entries containing this case-insensitive text")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N discovered GGUF entries")

    parser.add_argument(
        "--preferred-quants",
        default=",".join(DEFAULT_PREFERRED_QUANTS),
        help="Comma-separated quant priority list",
    )

    quality_group = parser.add_mutually_exclusive_group()
    quality_group.add_argument("--prefer-small", action="store_true", help="Favor lighter quantizations")
    quality_group.add_argument("--prefer-quality", action="store_true", help="Favor quality-oriented variants")

    parser.add_argument(
        "--name-mode",
        choices=["variant", "best"],
        default="variant",
        help="Ollama name strategy (default: variant)",
    )
    parser.add_argument("--prefix", default="", help="Optional prefix for generated Ollama names")
    parser.add_argument("--suffix", default="", help="Optional suffix for generated Ollama names")

    parser.add_argument("--num-ctx", type=int, default=None, help="Optional PARAMETER num_ctx for Modelfile")
    parser.add_argument("--temperature", type=float, default=None, help="Optional PARAMETER temperature for Modelfile")
    parser.add_argument("--top-p", type=float, default=None, help="Optional PARAMETER top_p for Modelfile")
    parser.add_argument("--system", default=None, help="Optional SYSTEM text for Modelfile")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose console output")
    parser.add_argument("--json", action="store_true", help="Emit final summary as JSON")

    args = parser.parse_args(argv)

    args.dry_run = not args.create
    args.selected_only = not args.all_variants

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")

    return args


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_delimited(value: str, sep: str = "_") -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", sep, normalized)
    normalized = re.sub(rf"{re.escape(sep)}+", sep, normalized).strip(sep)
    return normalized


def slugify(value: str, max_len: int = 96) -> str:
    text = normalize_delimited(value, sep="-")
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"[-._]{2,}", "-", text)
    text = text.strip("-._")
    if not text:
        text = "unknown"
    if len(text) > max_len:
        text = text[:max_len].rstrip("-._") or text[:max_len]
    return text


def canonicalize_quant(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    quant = normalize_delimited(raw, sep="_")
    quant_map = {
        "fp16": "f16",
        "fp32": "f32",
    }
    return quant_map.get(quant, quant)


def dedupe_tokens(tokens: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result


def parse_preferred_quants(raw: str) -> List[str]:
    items = [canonicalize_quant(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]
    result = [item for item in items if item]
    result = dedupe_tokens(result)
    return result or DEFAULT_PREFERRED_QUANTS.copy()


def find_gguf_files(root: Path) -> List[Path]:
    if not root.exists() or not root.is_dir():
        return []
    paths: List[Path] = []
    for candidate in root.rglob("*"):
        if candidate.is_file() and candidate.suffix.lower() == ".gguf":
            paths.append(candidate)
    return sorted(paths)


def detect_tags(*parts: str) -> List[str]:
    bag = " ".join(parts).lower()
    normalized = normalize_delimited(bag, sep="_")
    tokens = set(normalized.split("_"))
    tags: List[str] = []
    for tag in SEMANTIC_TAGS:
        if tag == "embed":
            if "embed" in tokens or "embedding" in tokens:
                tags.append(tag)
            continue
        if tag in tokens:
            tags.append(tag)
    return tags


_QUANT_PATTERNS = [
    r"(iq\d+_[msl])",
    r"(iq\d+)",
    r"(q\d+_k_[msl])",
    r"(q\d+_k)",
    r"(q\d+_[01])",
    r"(q\d+)",
    r"(fp16|f16|bf16|fp32|f32)",
]


def guess_quant_from_text(*texts: str) -> Optional[str]:
    for text in texts:
        normalized = normalize_delimited(text, sep="_")
        for pattern in _QUANT_PATTERNS:
            match = re.search(rf"(?:^|_){pattern}(?:_|$)", normalized)
            if match:
                return canonicalize_quant(match.group(1))
    return None


def parse_variant(path: Path, lmstudio_root: Path) -> VariantRecord:
    warnings: List[str] = []
    source_abs = path.expanduser().resolve(strict=False)
    exists = source_abs.exists()

    publisher: Optional[str] = None
    model_family: Optional[str] = None

    try:
        relative = source_abs.relative_to(lmstudio_root)
        if len(relative.parts) >= 3:
            publisher = relative.parts[0]
            model_family = relative.parts[1]
        elif len(relative.parts) == 2:
            publisher = relative.parts[0]
            warnings.append("Ruta parcial bajo LM Studio: falta carpeta de familia de modelo.")
        else:
            warnings.append("Ruta no canónica bajo LM Studio; publisher/model_family inciertos.")
    except ValueError:
        warnings.append("Ruta fuera de --lmstudio-dir; metadatos de publisher/model_family inciertos.")

    quant_guess = guess_quant_from_text(path.stem, path.name, str(path.parent))
    tags = detect_tags(str(path.parent), path.name)

    if not publisher:
        warnings.append("No se pudo inferir publisher de forma fiable.")
    if not model_family:
        warnings.append("No se pudo inferir model_family de forma fiable.")
    if not quant_guess:
        warnings.append("No se pudo inferir quant_guess; se marca como incierto.")

    return VariantRecord(
        source_path=path,
        source_path_abs=source_abs,
        exists=exists,
        publisher=publisher,
        model_family=model_family,
        filename=path.name,
        stem=path.stem,
        quant_guess=quant_guess,
        tags_detected=tags,
        warnings=warnings,
    )


def remove_quant_fragments(value: str) -> str:
    normalized = normalize_delimited(value, sep="_")
    patterns = [
        r"(?:^|_)(?:iq\d+_[msl]|iq\d+)(?:_|$)",
        r"(?:^|_)(?:q\d+_k_[msl]|q\d+_k|q\d+_[01]|q\d+)(?:_|$)",
        r"(?:^|_)(?:fp16|f16|bf16|fp32|f32)(?:_|$)",
    ]
    for pattern in patterns:
        normalized = re.sub(pattern, "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def derive_group_key(variant: VariantRecord) -> Tuple[str, bool]:
    base = variant.model_family or variant.stem
    cleaned = remove_quant_fragments(base)

    tokens = [t for t in cleaned.split("_") if t and t not in NOISE_TOKENS]
    tokens = dedupe_tokens(tokens)

    low_confidence = False
    if len(tokens) < 2:
        fallback = normalize_delimited(variant.stem, sep="_")
        fallback_tokens = [t for t in fallback.split("_") if t and t not in NOISE_TOKENS]
        tokens = dedupe_tokens(fallback_tokens)
        low_confidence = True

    if not tokens:
        tokens = ["unknown"]
        low_confidence = True

    publisher_token = slugify(variant.publisher or "unknown-publisher", max_len=30).replace("-", "_")
    core = "_".join(tokens[:12])
    return f"{publisher_token}__{core}", low_confidence


def apply_filters(
    variants: List[VariantRecord],
    only: Optional[str],
    exclude: Optional[str],
    limit: Optional[int],
) -> Tuple[List[VariantRecord], int]:
    only_norm = only.lower() if only else None
    exclude_norm = exclude.lower() if exclude else None

    filtered: List[VariantRecord] = []
    omitted = 0

    for variant in variants:
        haystack = " ".join(
            [
                str(variant.source_path_abs),
                variant.publisher or "",
                variant.model_family or "",
                variant.filename,
                variant.stem,
            ]
        ).lower()

        if only_norm and only_norm not in haystack:
            omitted += 1
            continue
        if exclude_norm and exclude_norm in haystack:
            omitted += 1
            continue
        filtered.append(variant)

    if limit is not None and len(filtered) > limit:
        omitted += len(filtered) - limit
        filtered = filtered[:limit]

    return filtered, omitted


def group_variants(variants: List[VariantRecord]) -> Dict[str, List[VariantRecord]]:
    grouped: Dict[str, List[VariantRecord]] = {}
    for variant in variants:
        key, low_conf = derive_group_key(variant)
        variant.group_key = key
        if low_conf:
            variant.warnings.append("Agrupación de baja confianza; clave derivada de información limitada.")
        grouped.setdefault(key, []).append(variant)

    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda v: str(v.source_path_abs))
    return grouped


def quant_size_bucket(quant: Optional[str]) -> int:
    if not quant:
        return 99
    q = canonicalize_quant(quant)
    if q in {"f32"}:
        return 32
    if q in {"f16", "bf16"}:
        return 16
    match = re.match(r"[iq]?(\d+)", q or "")
    if match:
        return int(match.group(1))
    return 99


def quant_quality_score(quant: Optional[str]) -> int:
    if not quant:
        return 0
    q = canonicalize_quant(quant)
    if q == "f32":
        return 100
    if q in {"f16", "bf16"}:
        return 90
    if q == "q8_0":
        return 80
    if q == "q6_k":
        return 70
    if q and q.startswith("q5"):
        return 62
    if q and q.startswith("q4"):
        return 52
    if q and q.startswith("q3"):
        return 42
    if q and q.startswith("q2"):
        return 32
    if q and q.startswith("iq"):
        return 48
    return 10


def path_cleanliness_score(variant: VariantRecord) -> int:
    score = 0
    name = variant.filename
    if name == name.lower():
        score += 1
    if " " not in name:
        score += 1
    if re.search(r"[^a-zA-Z0-9._-]", name) is None:
        score += 1
    if variant.model_family:
        score += 1
    if variant.publisher:
        score += 1
    score += max(0, 12 - min(12, len(name) // 8))
    return score


def candidate_sort_key(
    variant: VariantRecord,
    preferred_map: Dict[str, int],
    prefer_small: bool,
    prefer_quality: bool,
) -> Tuple[Any, ...]:
    quant = canonicalize_quant(variant.quant_guess)
    pref_rank = preferred_map.get(quant or "", 10_000)
    cleanliness = path_cleanliness_score(variant)
    path_len = len(str(variant.source_path_abs))

    if prefer_small:
        return (
            quant_size_bucket(quant),
            pref_rank,
            -cleanliness,
            path_len,
            variant.filename,
        )

    if prefer_quality:
        return (
            -quant_quality_score(quant),
            pref_rank,
            -cleanliness,
            path_len,
            variant.filename,
        )

    return (
        pref_rank,
        -cleanliness,
        path_len,
        variant.filename,
    )


def selection_profile_name(prefer_small: bool, prefer_quality: bool) -> str:
    if prefer_small:
        return "prefer-small"
    if prefer_quality:
        return "prefer-quality"
    return "default"


def select_variants(
    groups: Dict[str, List[VariantRecord]],
    preferred_quants: List[str],
    prefer_small: bool,
    prefer_quality: bool,
) -> Dict[str, VariantRecord]:
    preferred_map = {canonicalize_quant(quant) or quant: idx for idx, quant in enumerate(preferred_quants)}
    selected: Dict[str, VariantRecord] = {}
    profile = selection_profile_name(prefer_small, prefer_quality)

    for key, variants in groups.items():
        ordered = sorted(
            variants,
            key=lambda item: candidate_sort_key(item, preferred_map, prefer_small, prefer_quality),
        )
        chosen = ordered[0]
        selected[key] = chosen

        chosen_quant = canonicalize_quant(chosen.quant_guess) or "unknown"
        pref_index = preferred_map.get(chosen_quant)
        if pref_index is not None:
            reason = (
                f"Seleccionada por heurística operativa (perfil={profile}): quant '{chosen_quant}' "
                f"aparece en prioridad #{pref_index + 1} de {preferred_quants}; desempate por limpieza de ruta."
            )
        else:
            reason = (
                f"Seleccionada por heurística operativa (perfil={profile}): quant '{chosen_quant}' "
                f"no listada en prioridad explícita; desempate por limpieza de ruta."
            )

        for variant in variants:
            if variant is chosen:
                variant.selected_in_group = True
                variant.selection_reason = reason
            else:
                variant.selected_in_group = False
                variant.selection_reason = (
                    "No seleccionada por heurística operativa; "
                    f"se prefirió '{chosen.filename}' para el grupo '{key}'."
                )

    return selected


def safe_join_parts(parts: Iterable[str]) -> str:
    clean: List[str] = []
    for part in parts:
        if not part:
            continue
        normalized = slugify(part)
        if normalized == "unknown":
            continue
        clean.append(normalized)
    if not clean:
        return "unknown"
    text = "-".join(clean)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "unknown"


def model_name_component(variant: VariantRecord) -> str:
    base = variant.model_family or remove_quant_fragments(variant.stem) or variant.stem
    model_slug = slugify(base)
    if variant.publisher:
        publisher_slug = slugify(variant.publisher)
        if model_slug.startswith(f"{publisher_slug}-"):
            model_slug = model_slug[len(publisher_slug) + 1 :]
    return model_slug or "unknown-model"


def build_ollama_name(
    variant: VariantRecord,
    mode: str,
    prefix: str,
    suffix: str,
) -> str:
    publisher = slugify(variant.publisher or "unknown-publisher", max_len=30)
    model_core = model_name_component(variant)
    quant = slugify(canonicalize_quant(variant.quant_guess) or "unknown", max_len=30)

    pieces: List[str] = []
    if prefix:
        pieces.append(prefix)
    pieces.append(publisher)
    pieces.append(model_core)

    if mode == "variant":
        pieces.append(quant)
    else:
        pieces.append("best")

    if suffix:
        pieces.append(suffix)

    raw = safe_join_parts(pieces)
    return slugify(raw, max_len=96)


def render_modelfile_content(
    gguf_abs: Path,
    num_ctx: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    system: Optional[str] = None,
) -> str:
    lines = [f"FROM {gguf_abs}"]
    if num_ctx is not None:
        lines.append(f"PARAMETER num_ctx {num_ctx}")
    if temperature is not None:
        lines.append(f"PARAMETER temperature {temperature}")
    if top_p is not None:
        lines.append(f"PARAMETER top_p {top_p}")
    if system is not None:
        lines.append(f"SYSTEM {json.dumps(system)}")
    return "\n".join(lines) + "\n"


def render_variant_readme(variant: VariantRecord, modelfile_content: str) -> str:
    status = "Seleccionada en su grupo" if variant.selected_in_group else "No seleccionada en su grupo"
    user_status = "Seleccionada por usuario" if variant.selected_by_user else "No seleccionada por usuario"
    manual_create = (
        f"ollama create {variant.proposed_ollama_name} -f "
        f"{variant.modelfile_path if variant.modelfile_path else '<Modelfile>'}"
    )
    run_cmd = f"ollama run {variant.proposed_ollama_name}"

    warnings = variant.warnings or ["Sin advertencias registradas."]
    warning_lines = "\n".join(f"- {item}" for item in warnings)
    effective = json.dumps(variant.effective_parameters, ensure_ascii=False, indent=2)
    overrides = json.dumps(variant.overrides_applied, ensure_ascii=False, indent=2)

    return f"""# {variant.proposed_ollama_name}

## Source GGUF
- Ruta original: `{variant.source_path}`
- Ruta absoluta: `{variant.source_path_abs}`
- Existe en disco: `{variant.exists}`

## Grouping
- Group key: `{variant.group_key}`
- Motivo de agrupación: tokens normalizados y eliminación prudente de sufijos de quantización.
- Estado heurístico: **{status}**
- Estado sesión: **{user_status}**
- Razón de selección heurística: {variant.selection_reason}

## Ollama
- Nombre propuesto: `{variant.proposed_ollama_name}`

## Effective Settings
```json
{effective}
```

## Overrides Applied
```json
{overrides}
```

### Modelfile
```text
{modelfile_content.rstrip()}
```

### Comandos
- Crear manualmente:
  ```bash
  {manual_create}
  ```
- Ejecutar:
  ```bash
  {run_cmd}
  ```

## Notas y advertencias
{warning_lines}
"""


def render_batch_script(selected_only: bool) -> str:
    header = "selected_models.json" if selected_only else "index.json"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated by lmstudio_to_ollama.py",
        "# Safety gate: set EXECUTE=1 to run ollama create commands.",
        "# Optional filter: set NAME_FILTER='qwen' to run only matching model names.",
        'EXECUTE="${EXECUTE:-0}"',
        'NAME_FILTER="${NAME_FILTER:-}"',
        f'INDEX_FILE="${{INDEX_FILE:-$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)/{header}}}"',
        "",
        'if [[ ! -f "$INDEX_FILE" ]]; then',
        '  echo "Index file not found: $INDEX_FILE" >&2',
        "  exit 1",
        "fi",
        "",
        'echo "Using index: $INDEX_FILE"',
        'echo "EXECUTE=$EXECUTE"',
        '[[ -n "$NAME_FILTER" ]] && echo "NAME_FILTER=$NAME_FILTER"',
        "",
        "while IFS=$'\\t' read -r name modelfile; do",
        '  [[ -z "$name" || -z "$modelfile" ]] && continue',
        '  if [[ -n "$NAME_FILTER" && "$name" != *"$NAME_FILTER"* ]]; then',
        "    continue",
        "  fi",
        "",
        '  cmd=(ollama create "$name" -f "$modelfile")',
        '  echo "> ${cmd[*]}"',
        '  if [[ "$EXECUTE" == "1" ]]; then',
        '    "${cmd[@]}"',
        "  fi",
        "done < <(python3 - \"$INDEX_FILE\" <<'PY'",
        "import json",
        "import sys",
        "from pathlib import Path",
        "",
        "idx = Path(sys.argv[1])",
        "data = json.loads(idx.read_text(encoding='utf-8'))",
        "if isinstance(data, dict):",
        "    items = data.get('models', [])",
        "elif isinstance(data, list):",
        "    items = data",
        "else:",
        "    items = []",
        "for item in items:",
        "    name = item.get('proposed_ollama_name')",
        "    modelfile = item.get('modelfile_path')",
        "    if not name or not modelfile:",
        "        continue",
        "    print(f\"{name}\\t{modelfile}\")",
        "PY",
        ")",
        "",
        'if [[ "$EXECUTE" != "1" ]]; then',
        '  echo "Dry mode active. Re-run with EXECUTE=1 to execute imports."',
        "fi",
    ]
    return "\n".join(lines) + "\n"


def write_text_file(path: Path, content: str, force: bool) -> bool:
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def should_skip_variant_dir(variant_dir: Path, force: bool) -> bool:
    if force:
        return False
    if not variant_dir.exists():
        return False
    expected = [variant_dir / "Modelfile", variant_dir / "metadata.json", variant_dir / "README.md"]
    return any(path.exists() for path in expected)


def execute_create(variant: VariantRecord, modelfile_path: Path, logger: Console) -> None:
    variant.create_attempted = True
    cmd = ["ollama", "create", variant.proposed_ollama_name, "-f", str(modelfile_path)]
    variant.create_command = " ".join(cmd)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        variant.stdout = (result.stdout or "").strip()
        variant.stderr = (result.stderr or "").strip()
        variant.create_success = result.returncode == 0
        if not variant.create_success:
            variant.warnings.append(f"ollama create devolvió código {result.returncode}.")
            logger.debug(
                f"[create:fail] {variant.proposed_ollama_name}: returncode={result.returncode}"
            )
        else:
            logger.debug(f"[create:ok] {variant.proposed_ollama_name}")
    except Exception as exc:  # pragma: no cover
        variant.create_success = False
        variant.stderr = str(exc)
        variant.warnings.append("Excepción al ejecutar ollama create.")
        logger.debug(f"[create:error] {variant.proposed_ollama_name}: {exc}")


def clone_summary(summary: Summary) -> Summary:
    return Summary(
        detected_models=summary.detected_models,
        grouped_models=summary.grouped_models,
        groups_detected=summary.groups_detected,
        variants_processed=summary.variants_processed,
        selected_variants=summary.selected_variants,
        modelfiles_generated=summary.modelfiles_generated,
        created_ok=summary.created_ok,
        created_failed=summary.created_failed,
        omitted=summary.omitted,
    )


def discover_models(
    lmstudio_root: Path,
    only: Optional[str],
    exclude: Optional[str],
    limit: Optional[int],
    preferred_quants: List[str],
    prefer_small: bool,
    prefer_quality: bool,
) -> DiscoveryResult:
    summary = Summary()
    discovered_paths = find_gguf_files(lmstudio_root)
    summary.detected_models = len(discovered_paths)

    variants = [parse_variant(path, lmstudio_root) for path in discovered_paths]
    variants, omitted_by_filter = apply_filters(variants, only=only, exclude=exclude, limit=limit)
    summary.omitted += omitted_by_filter

    grouped = group_variants(variants)
    summary.groups_detected = len(grouped)
    summary.grouped_models = sum(len(items) for items in grouped.values())

    selected_map = select_variants(
        grouped,
        preferred_quants=preferred_quants,
        prefer_small=prefer_small,
        prefer_quality=prefer_quality,
    )
    summary.selected_variants = len(selected_map)

    return DiscoveryResult(
        variants=variants,
        groups=grouped,
        selected_map=selected_map,
        preferred_quants=preferred_quants,
        summary=summary,
    )


def assign_unique_names_to_jobs(
    jobs: List[VariantJob],
    mode: str,
    prefix: str,
    suffix: str,
) -> None:
    used: Dict[str, int] = {}

    for job in jobs:
        if job.requested_name:
            base = slugify(job.requested_name, max_len=96)
        else:
            base = build_ollama_name(job.variant, mode=mode, prefix=prefix, suffix=suffix)

        candidate = base
        counter = 2
        while candidate in used:
            candidate = slugify(f"{base}-{counter}", max_len=96)
            counter += 1
        used[candidate] = 1
        job.final_name = candidate


def build_effective_parameters(
    global_num_ctx: Optional[int],
    global_temperature: Optional[float],
    global_top_p: Optional[float],
    global_system: Optional[str],
    override: Optional[ModelOverride] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    overrides_applied: Dict[str, Any] = {}

    num_ctx = global_num_ctx
    temperature = global_temperature
    top_p = global_top_p
    system = global_system

    if override is not None:
        if override.num_ctx is not None:
            num_ctx = override.num_ctx
            overrides_applied["num_ctx"] = override.num_ctx
        if override.temperature is not None:
            temperature = override.temperature
            overrides_applied["temperature"] = override.temperature
        if override.top_p is not None:
            top_p = override.top_p
            overrides_applied["top_p"] = override.top_p
        if override.system is not None:
            system = override.system
            overrides_applied["system"] = override.system
        if override.name_override:
            overrides_applied["name_override"] = override.name_override

    effective = {
        "num_ctx": num_ctx,
        "temperature": temperature,
        "top_p": top_p,
        "system": system,
    }

    return effective, overrides_applied


def process_jobs(
    jobs: List[VariantJob],
    output_dir: Path,
    force: bool,
    create: bool,
    logger: Console,
    selected_only: bool,
    summary_seed: Summary,
    index_context: Dict[str, Any],
) -> ArtifactResult:
    summary = clone_summary(summary_seed)
    generated_at = utc_now_iso()

    output_dir.mkdir(parents=True, exist_ok=True)

    processed_metadata: List[Dict[str, Any]] = []
    selected_metadata: List[Dict[str, Any]] = []

    for job in jobs:
        variant = job.variant
        variant.generated_at = generated_at
        variant.proposed_ollama_name = job.final_name
        variant.selected_by_user = job.selected_by_user
        variant.overrides_applied = job.overrides_applied
        variant.effective_parameters = job.effective_parameters

        variant_dir = output_dir / variant.proposed_ollama_name
        if should_skip_variant_dir(variant_dir, force=force):
            variant.skipped = True
            variant.skip_reason = "Directorio de salida ya existe; use --force para sobrescribir."
            variant.warnings.append(variant.skip_reason)
            summary.omitted += 1
            processed_metadata.append(variant.to_metadata())
            if variant.selected_in_group or variant.selected_by_user:
                selected_metadata.append(variant.to_metadata())
            continue

        variant_dir.mkdir(parents=True, exist_ok=True)

        modelfile_path = variant_dir / "Modelfile"
        metadata_path = variant_dir / "metadata.json"
        readme_path = variant_dir / "README.md"

        modelfile_content = render_modelfile_content(
            gguf_abs=variant.source_path_abs,
            num_ctx=job.effective_parameters.get("num_ctx"),
            temperature=job.effective_parameters.get("temperature"),
            top_p=job.effective_parameters.get("top_p"),
            system=job.effective_parameters.get("system"),
        )

        modelfile_written = write_text_file(modelfile_path, modelfile_content, force=force)
        if modelfile_written:
            summary.modelfiles_generated += 1

        variant.modelfile_path = modelfile_path
        variant.readme_path = readme_path

        readme_content = render_variant_readme(variant, modelfile_content)
        _ = write_text_file(readme_path, readme_content, force=force)

        if not variant.exists:
            variant.warnings.append("GGUF no existe al momento de generar artefactos.")

        if create and variant.exists:
            execute_create(variant, modelfile_path, logger)
        else:
            variant.create_attempted = False
            variant.create_success = None

        metadata_content = json.dumps(variant.to_metadata(), ensure_ascii=False, indent=2)
        _ = write_text_file(metadata_path, metadata_content + "\n", force=force)

        processed_metadata.append(variant.to_metadata())
        if variant.selected_in_group or variant.selected_by_user:
            selected_metadata.append(variant.to_metadata())

        summary.variants_processed += 1
        if variant.create_attempted:
            if variant.create_success:
                summary.created_ok += 1
            elif variant.create_success is False:
                summary.created_failed += 1

    index_payload = {
        "generated_at": generated_at,
        "models": processed_metadata,
        "summary": summary.to_dict(),
    }
    index_payload.update(index_context)

    selected_payload = {
        "generated_at": generated_at,
        "models": selected_metadata,
        "lmstudio_dir": index_context.get("lmstudio_dir"),
        "output_dir": index_context.get("output_dir"),
    }

    index_path = output_dir / "index.json"
    selected_path = output_dir / "selected_models.json"
    batch_path = output_dir / "import_all.sh"

    if not write_text_file(index_path, json.dumps(index_payload, ensure_ascii=False, indent=2) + "\n", force=force):
        summary.omitted += 1
    if not write_text_file(selected_path, json.dumps(selected_payload, ensure_ascii=False, indent=2) + "\n", force=force):
        summary.omitted += 1

    batch_content = render_batch_script(selected_only=selected_only)
    if not write_text_file(batch_path, batch_content, force=force):
        summary.omitted += 1

    try:
        batch_path.chmod(0o755)
    except OSError:
        pass

    return ArtifactResult(
        summary=summary,
        index_path=index_path,
        selected_path=selected_path,
        batch_path=batch_path,
        processed_metadata=processed_metadata,
        selected_metadata=selected_metadata,
    )


def summarize_to_console(summary: Summary, logger: Console) -> None:
    logger.info("")
    logger.info("Summary")
    logger.info(f"- modelos detectados: {summary.detected_models}")
    logger.info(f"- grupos detectados: {summary.groups_detected}")
    logger.info(f"- variantes agrupadas: {summary.grouped_models}")
    logger.info(f"- variantes procesadas: {summary.variants_processed}")
    logger.info(f"- seleccionados: {summary.selected_variants}")
    logger.info(f"- modelfiles generados: {summary.modelfiles_generated}")
    logger.info(f"- creados en Ollama: {summary.created_ok}")
    logger.info(f"- fallidos: {summary.created_failed}")
    logger.info(f"- omitidos: {summary.omitted}")


def run_cli(args: argparse.Namespace, logger: Console) -> int:
    lmstudio_root = Path(args.lmstudio_dir).expanduser().resolve(strict=False)
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)

    logger.info(f"LM Studio dir: {lmstudio_root}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Mode: {'dry-run' if args.dry_run else 'create'}")

    preferred_quants = parse_preferred_quants(args.preferred_quants)
    discovery = discover_models(
        lmstudio_root=lmstudio_root,
        only=args.only,
        exclude=args.exclude,
        limit=args.limit,
        preferred_quants=preferred_quants,
        prefer_small=args.prefer_small,
        prefer_quality=args.prefer_quality,
    )

    if discovery.summary.detected_models == 0:
        logger.info("No se detectaron archivos GGUF.")

    variants_to_process: List[VariantRecord] = []
    if args.selected_only:
        for key in sorted(discovery.selected_map.keys()):
            variants_to_process.append(discovery.selected_map[key])
    else:
        for key in sorted(discovery.groups.keys()):
            variants_to_process.extend(discovery.groups[key])

    jobs: List[VariantJob] = []
    for variant in variants_to_process:
        effective, overrides = build_effective_parameters(
            global_num_ctx=args.num_ctx,
            global_temperature=args.temperature,
            global_top_p=args.top_p,
            global_system=args.system,
            override=None,
        )
        jobs.append(
            VariantJob(
                variant=copy.deepcopy(variant),
                requested_name=None,
                selected_by_user=variant.selected_in_group,
                overrides_applied=overrides,
                effective_parameters=effective,
            )
        )

    assign_unique_names_to_jobs(jobs, mode=args.name_mode, prefix=args.prefix, suffix=args.suffix)

    create_enabled = args.create
    if create_enabled and shutil.which("ollama") is None:
        logger.info("Error: 'ollama' no está disponible en PATH. Se omite --create.")
        create_enabled = False
        for job in jobs:
            job.variant.warnings.append("--create solicitado, pero ollama no está instalado/accesible.")

    summary_seed = clone_summary(discovery.summary)
    summary_seed.selected_variants = len(discovery.selected_map)

    index_context = {
        "lmstudio_dir": str(lmstudio_root),
        "output_dir": str(output_dir),
        "dry_run": not create_enabled,
        "selected_only": args.selected_only,
        "name_mode": args.name_mode,
        "preferred_quants": preferred_quants,
        "selection_profile": selection_profile_name(args.prefer_small, args.prefer_quality),
        "interactive": False,
    }

    result = process_jobs(
        jobs=jobs,
        output_dir=output_dir,
        force=args.force,
        create=create_enabled,
        logger=logger,
        selected_only=args.selected_only,
        summary_seed=summary_seed,
        index_context=index_context,
    )

    if args.json:
        print(json.dumps(result.summary.to_dict(), ensure_ascii=False, indent=2))
    else:
        summarize_to_console(result.summary, logger)
        logger.info("")
        logger.info(f"Artifacts index: {result.index_path}")
        logger.info(f"Selected index: {result.selected_path}")
        logger.info(f"Batch script: {result.batch_path}")

    return 0


def _parse_optional_int(value: str) -> Optional[int]:
    stripped = value.strip()
    if stripped == "":
        return None
    return int(stripped)


def _parse_optional_float(value: str) -> Optional[float]:
    stripped = value.strip()
    if stripped == "":
        return None
    return float(stripped)


def _profile_from_flags(prefer_small: bool, prefer_quality: bool) -> str:
    if prefer_small:
        return "small"
    if prefer_quality:
        return "quality"
    return "default"


def _flags_from_profile(profile: str) -> Tuple[bool, bool]:
    if profile == "small":
        return True, False
    if profile == "quality":
        return False, True
    return False, False


class TerminalImporterApp:
    def __init__(self, stdscr: Any, args: argparse.Namespace):
        self.stdscr = stdscr
        self.args = args
        self.logger = Console(verbose=False, json_mode=False)
        self.attr_box = curses.A_BOLD
        self.attr_text = curses.A_BOLD
        self.attr_title = curses.A_BOLD
        self.attr_list_normal = curses.A_BOLD
        self.attr_list_selected = curses.A_REVERSE | curses.A_BOLD
        self.attr_dim = curses.A_DIM
        self._init_theme()

        self.state = InteractiveSelection(
            global_config=GlobalConfig(
                preferred_quants=parse_preferred_quants(args.preferred_quants),
                prefer_small=args.prefer_small,
                prefer_quality=args.prefer_quality,
                name_mode=args.name_mode,
                prefix=args.prefix,
                suffix=args.suffix,
                num_ctx=args.num_ctx,
                temperature=args.temperature,
                top_p=args.top_p,
                system=args.system,
                only=args.only,
                exclude=args.exclude,
                limit=args.limit,
                output_dir=Path(args.output_dir).expanduser().resolve(strict=False),
                force=args.force,
            ),
            lmstudio_root=Path(args.lmstudio_dir).expanduser().resolve(strict=False),
        )

        self.cursor = 0
        self.scroll = 0
        self.quit_requested = False
        self.ollama_available = shutil.which("ollama") is not None

        self.refresh_discovery(initial=True)

    def _init_theme(self) -> None:
        # Monochrome-safe defaults first. This avoids terminal/theme combinations
        # where color pairs become unreadable or invisible.
        self.attr_box = curses.A_BOLD
        self.attr_text = curses.A_NORMAL
        self.attr_title = curses.A_BOLD
        self.attr_list_normal = curses.A_NORMAL
        self.attr_list_selected = curses.A_REVERSE | curses.A_BOLD
        self.attr_dim = curses.A_DIM

        # Optional color mode is opt-in via env var.
        if os.environ.get("LMSTUDIO_TUI_COLOR", "0") != "1":
            return
        if curses is None or not hasattr(curses, "has_colors") or not curses.has_colors():
            return
        try:
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_CYAN, -1)
            curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_CYAN)
            curses.init_pair(3, curses.COLOR_WHITE, -1)
            curses.init_pair(4, curses.COLOR_YELLOW, -1)
            curses.init_pair(5, curses.COLOR_WHITE, -1)
            self.attr_box = curses.color_pair(5) | curses.A_BOLD
            self.attr_text = curses.color_pair(3)
            self.attr_title = curses.color_pair(1) | curses.A_BOLD
            self.attr_list_normal = self.attr_text
            self.attr_list_selected = curses.color_pair(2) | curses.A_BOLD
            self.attr_dim = curses.color_pair(4) | curses.A_DIM
        except Exception:
            pass

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        if max_len <= 0:
            return ""
        if len(text) <= max_len:
            return text
        if max_len <= 3:
            return text[:max_len]
        return text[: max_len - 3] + "..."

    def _draw_box(self, y: int, x: int, h: int, w: int, title: str) -> None:
        if h < 2 or w < 2:
            return
        try:
            # ASCII borders are more predictable across terminals than ACS chars.
            self.stdscr.addnstr(y, x, "+" + "-" * max(0, w - 2) + "+", w, self.attr_box)
            for row in range(1, h - 1):
                self.stdscr.addnstr(y + row, x, "|", 1, self.attr_box)
                self.stdscr.addnstr(y + row, x + w - 1, "|", 1, self.attr_box)
            self.stdscr.addnstr(y + h - 1, x, "+" + "-" * max(0, w - 2) + "+", w, self.attr_box)
            if title and w > 6:
                self.stdscr.addnstr(y, x + 2, f" {title} ", max(0, w - 4), self.attr_title)
        except curses.error:
            pass

    def set_status(self, msg: str) -> None:
        self.state.status = msg

    def refresh_discovery(self, initial: bool = False) -> None:
        old = self.state.overrides
        cfg = self.state.global_config

        discovery = discover_models(
            lmstudio_root=self.state.lmstudio_root,
            only=cfg.only,
            exclude=cfg.exclude,
            limit=cfg.limit,
            preferred_quants=cfg.preferred_quants,
            prefer_small=cfg.prefer_small,
            prefer_quality=cfg.prefer_quality,
        )

        self.state.groups = discovery.groups
        self.state.group_order = sorted(discovery.groups.keys())
        self.state.discovery_summary = discovery.summary

        new_overrides: Dict[str, ModelOverride] = {}
        for key in self.state.group_order:
            variants = self.state.groups[key]
            selected_variant = discovery.selected_map.get(key)
            default_idx = 0
            if selected_variant is not None:
                for idx, item in enumerate(variants):
                    if item is selected_variant:
                        default_idx = idx
                        break

            previous = old.get(key)
            if previous is None:
                chosen_variant = variants[default_idx]
                new_overrides[key] = ModelOverride(
                    enabled=True,
                    variant_index=default_idx,
                    selected_source_path=str(chosen_variant.source_path_abs),
                )
                continue

            idx = previous.variant_index
            if previous.selected_source_path:
                for i, candidate in enumerate(variants):
                    if str(candidate.source_path_abs) == previous.selected_source_path:
                        idx = i
                        break

            idx = max(0, min(idx, len(variants) - 1))
            previous.variant_index = idx
            previous.selected_source_path = str(variants[idx].source_path_abs)
            new_overrides[key] = previous

        self.state.overrides = new_overrides

        if self.cursor >= len(self.state.group_order):
            self.cursor = max(0, len(self.state.group_order) - 1)
        if self.cursor < 0:
            self.cursor = 0

        self.set_status(
            f"Detectados={discovery.summary.detected_models} | Grupos={discovery.summary.groups_detected} | "
            f"Filtrados/Omitidos={discovery.summary.omitted}"
        )

        if not initial and len(self.state.group_order) == 0:
            self.set_status("No hay grupos disponibles con los filtros actuales.")

    def current_group_key(self) -> Optional[str]:
        if not self.state.group_order:
            return None
        return self.state.group_order[self.cursor]

    def current_variant(self) -> Optional[VariantRecord]:
        key = self.current_group_key()
        if not key:
            return None
        override = self.state.overrides[key]
        variants = self.state.groups[key]
        return variants[override.variant_index]

    def cycle_variant(self, step: int) -> None:
        key = self.current_group_key()
        if not key:
            return
        override = self.state.overrides[key]
        variants = self.state.groups[key]
        if not variants:
            return
        override.variant_index = (override.variant_index + step) % len(variants)
        override.selected_source_path = str(variants[override.variant_index].source_path_abs)

    def toggle_enabled(self) -> None:
        key = self.current_group_key()
        if not key:
            return
        override = self.state.overrides[key]
        override.enabled = not override.enabled

    def prompt_input(self, prompt: str) -> Optional[str]:
        h, w = self.stdscr.getmaxyx()
        curses.echo()
        try:
            curses.curs_set(1)
        except Exception:
            pass

        self.stdscr.move(h - 1, 0)
        self.stdscr.clrtoeol()
        self.stdscr.addnstr(h - 1, 0, prompt, max(0, w - 1))
        self.stdscr.refresh()

        try:
            raw = self.stdscr.getstr(h - 1, min(len(prompt), max(0, w - 1)), max(1, w - len(prompt) - 1))
        except Exception:
            raw = None

        curses.noecho()
        try:
            curses.curs_set(0)
        except Exception:
            pass

        if raw is None:
            return None
        return raw.decode("utf-8", errors="ignore").strip()

    def confirm(self, message: str) -> bool:
        answer = self.prompt_input(f"{message} [y/N]: ")
        return (answer or "").lower() in {"y", "yes"}

    def edit_model_override(self, field_name: str) -> None:
        key = self.current_group_key()
        if not key:
            self.set_status("No hay grupo activo.")
            return

        override = self.state.overrides[key]

        if field_name == "name_override":
            value = self.prompt_input("Name override (vacío para limpiar): ")
            if value is None:
                return
            override.name_override = value or None
            self.set_status("name_override actualizado.")
            return

        if field_name == "num_ctx":
            value = self.prompt_input("num_ctx override (entero, vacío=heredar): ")
            if value is None:
                return
            try:
                override.num_ctx = _parse_optional_int(value)
                self.set_status("num_ctx override actualizado.")
            except ValueError:
                self.set_status("num_ctx inválido.")
            return

        if field_name == "temperature":
            value = self.prompt_input("temperature override (float, vacío=heredar): ")
            if value is None:
                return
            try:
                override.temperature = _parse_optional_float(value)
                self.set_status("temperature override actualizado.")
            except ValueError:
                self.set_status("temperature inválido.")
            return

        if field_name == "top_p":
            value = self.prompt_input("top_p override (float, vacío=heredar): ")
            if value is None:
                return
            try:
                override.top_p = _parse_optional_float(value)
                self.set_status("top_p override actualizado.")
            except ValueError:
                self.set_status("top_p inválido.")
            return

        if field_name == "system":
            value = self.prompt_input("system override (vacío=heredar): ")
            if value is None:
                return
            override.system = value or None
            self.set_status("system override actualizado.")

    def edit_global(self) -> None:
        choice = self.prompt_input(
            "Global [m name-mode, b perfil, f prefix, u suffix, q quants, c num_ctx, t temp, p top_p, s system, o output, l limit]: "
        )
        if not choice:
            return

        key = choice[0].lower()
        cfg = self.state.global_config

        if key == "m":
            cfg.name_mode = "best" if cfg.name_mode == "variant" else "variant"
            self.set_status(f"name-mode={cfg.name_mode}")
            return

        if key == "b":
            profile = _profile_from_flags(cfg.prefer_small, cfg.prefer_quality)
            next_profile = "small" if profile == "default" else "quality" if profile == "small" else "default"
            cfg.prefer_small, cfg.prefer_quality = _flags_from_profile(next_profile)
            self.refresh_discovery()
            self.set_status(f"perfil de selección={next_profile}")
            return

        if key == "f":
            value = self.prompt_input("prefix global (vacío para limpiar): ")
            if value is not None:
                cfg.prefix = value or ""
                self.set_status("prefix actualizado.")
            return

        if key == "u":
            value = self.prompt_input("suffix global (vacío para limpiar): ")
            if value is not None:
                cfg.suffix = value or ""
                self.set_status("suffix actualizado.")
            return

        if key == "q":
            value = self.prompt_input("preferred quants csv: ")
            if value:
                cfg.preferred_quants = parse_preferred_quants(value)
                self.refresh_discovery()
                self.set_status("preferred quants actualizados.")
            return

        if key == "c":
            value = self.prompt_input("global num_ctx (entero, vacío=none): ")
            if value is None:
                return
            try:
                cfg.num_ctx = _parse_optional_int(value)
                self.set_status("global num_ctx actualizado.")
            except ValueError:
                self.set_status("num_ctx inválido.")
            return

        if key == "t":
            value = self.prompt_input("global temperature (float, vacío=none): ")
            if value is None:
                return
            try:
                cfg.temperature = _parse_optional_float(value)
                self.set_status("global temperature actualizado.")
            except ValueError:
                self.set_status("temperature inválido.")
            return

        if key == "p":
            value = self.prompt_input("global top_p (float, vacío=none): ")
            if value is None:
                return
            try:
                cfg.top_p = _parse_optional_float(value)
                self.set_status("global top_p actualizado.")
            except ValueError:
                self.set_status("top_p inválido.")
            return

        if key == "s":
            value = self.prompt_input("global system (vacío=none): ")
            if value is not None:
                cfg.system = value or None
                self.set_status("global system actualizado.")
            return

        if key == "o":
            value = self.prompt_input("output-dir: ")
            if value:
                cfg.output_dir = Path(value).expanduser().resolve(strict=False)
                self.set_status(f"output-dir={cfg.output_dir}")
            return

        if key == "l":
            value = self.prompt_input("limit (entero, vacío=none): ")
            if value is None:
                return
            try:
                cfg.limit = _parse_optional_int(value)
                self.refresh_discovery()
                self.set_status("limit actualizado.")
            except ValueError:
                self.set_status("limit inválido.")
            return

        self.set_status("Comando global no reconocido.")

    def set_filters(self, only: bool) -> None:
        cfg = self.state.global_config
        prompt = "Filtro ONLY (vacío=none): " if only else "Filtro EXCLUDE (vacío=none): "
        value = self.prompt_input(prompt)
        if value is None:
            return
        if only:
            cfg.only = value or None
        else:
            cfg.exclude = value or None
        self.refresh_discovery()
        self.set_status("Filtros actualizados.")

    def collect_jobs(self) -> Tuple[List[VariantJob], Dict[str, VariantJob]]:
        cfg = self.state.global_config
        jobs: List[VariantJob] = []
        by_group: Dict[str, VariantJob] = {}

        for key in self.state.group_order:
            override = self.state.overrides[key]
            if not override.enabled:
                continue

            variants = self.state.groups[key]
            variant = copy.deepcopy(variants[override.variant_index])
            variant.selected_by_user = True

            effective, overrides_applied = build_effective_parameters(
                global_num_ctx=cfg.num_ctx,
                global_temperature=cfg.temperature,
                global_top_p=cfg.top_p,
                global_system=cfg.system,
                override=override,
            )

            job = VariantJob(
                variant=variant,
                requested_name=override.name_override,
                selected_by_user=True,
                overrides_applied=overrides_applied,
                effective_parameters=effective,
            )
            jobs.append(job)
            by_group[key] = job

        assign_unique_names_to_jobs(jobs, mode=cfg.name_mode, prefix=cfg.prefix, suffix=cfg.suffix)
        return jobs, by_group

    def preview_active(self) -> None:
        key = self.current_group_key()
        if not key:
            self.popup("Preview", ["No hay grupo activo."])
            return

        jobs, by_group = self.collect_jobs()
        if key not in by_group:
            self.popup("Preview", ["El grupo activo está deshabilitado."])
            return

        job = by_group[key]
        modelfile = render_modelfile_content(
            gguf_abs=job.variant.source_path_abs,
            num_ctx=job.effective_parameters.get("num_ctx"),
            temperature=job.effective_parameters.get("temperature"),
            top_p=job.effective_parameters.get("top_p"),
            system=job.effective_parameters.get("system"),
        )

        lines = [
            f"Nombre Ollama: {job.final_name}",
            f"GGUF: {job.variant.source_path_abs}",
            "",
            "Modelfile:",
        ] + modelfile.strip().splitlines()
        self.popup("Preview", lines)

    def build_summary_seed(self, selected_count: int) -> Summary:
        seed = clone_summary(self.state.discovery_summary)
        seed.selected_variants = selected_count
        return seed

    def run_generation(self, create: bool) -> None:
        jobs, _ = self.collect_jobs()
        if not jobs:
            self.set_status("No hay modelos seleccionados para procesar.")
            return

        if create:
            if not self.ollama_available:
                self.set_status("'ollama' no está en PATH. Create bloqueado.")
                self.popup("Create blocked", ["No se encontró 'ollama' en PATH."])
                return
            if not self.confirm(
                "Esto ejecutará ollama create con los parámetros actuales (heredados/overrides). ¿Continuar?"
            ):
                self.set_status("Create cancelado por el usuario.")
                return

        cfg = self.state.global_config
        summary_seed = self.build_summary_seed(len(jobs))
        index_context = {
            "lmstudio_dir": str(self.state.lmstudio_root),
            "output_dir": str(cfg.output_dir),
            "dry_run": not create,
            "selected_only": True,
            "name_mode": cfg.name_mode,
            "preferred_quants": cfg.preferred_quants,
            "selection_profile": selection_profile_name(cfg.prefer_small, cfg.prefer_quality),
            "interactive": True,
            "filters": {
                "only": cfg.only,
                "exclude": cfg.exclude,
                "limit": cfg.limit,
            },
        }

        result = process_jobs(
            jobs=jobs,
            output_dir=cfg.output_dir,
            force=cfg.force,
            create=create,
            logger=self.logger,
            selected_only=True,
            summary_seed=summary_seed,
            index_context=index_context,
        )

        self.state.last_result = result
        self.set_status(
            f"Procesadas={result.summary.variants_processed} | "
            f"CreadasOK={result.summary.created_ok} | Fallidas={result.summary.created_failed}"
        )

        summary_lines = [
            f"Detected: {result.summary.detected_models}",
            f"Groups: {result.summary.groups_detected}",
            f"Processed: {result.summary.variants_processed}",
            f"Generated Modelfiles: {result.summary.modelfiles_generated}",
            f"Created OK: {result.summary.created_ok}",
            f"Created Failed: {result.summary.created_failed}",
            f"Omitted: {result.summary.omitted}",
            "",
            f"index.json: {result.index_path}",
            f"selected_models.json: {result.selected_path}",
            f"import_all.sh: {result.batch_path}",
        ]
        self.popup("Run Summary", summary_lines)

    def popup(self, title: str, lines: List[str]) -> None:
        h, w = self.stdscr.getmaxyx()
        max_line = max([len(line) for line in lines] + [len(title) + 4])
        pw = min(w - 4, max(50, max_line + 4))
        ph = min(h - 4, max(8, min(len(lines) + 4, h - 4)))
        y = max(1, (h - ph) // 2)
        x = max(1, (w - pw) // 2)

        win = curses.newwin(ph, pw, y, x)
        win.box()
        win.addnstr(0, 2, f" {title} ", pw - 4, curses.A_BOLD)

        usable = ph - 2
        for idx in range(min(usable - 1, len(lines))):
            win.addnstr(idx + 1, 1, lines[idx], pw - 2)

        if len(lines) > usable - 1:
            win.addnstr(ph - 2, 1, "...", pw - 2)

        win.addnstr(ph - 1, 2, "Press any key", pw - 4, curses.A_DIM)
        win.refresh()
        win.getch()

    def draw(self) -> None:
        self.stdscr.erase()
        h, w = self.stdscr.getmaxyx()

        if h < 18 or w < 96:
            self.stdscr.addnstr(0, 0, "Terminal too small. Min size ~96x18.", max(0, w - 1), curses.A_BOLD)
            self.stdscr.addnstr(1, 0, "Resize and press any key.", max(0, w - 1))
            self.stdscr.refresh()
            return

        left_w = max(28, w // 3)
        mid_w = max(32, w // 3)
        right_w = w - left_w - mid_w

        top_h = h - 3
        left_x = 0
        mid_x = left_w
        right_x = left_w + mid_w

        self._draw_box(0, left_x, top_h, left_w, "Groups/Variants")
        self._draw_box(0, mid_x, top_h, mid_w, "Active Model Detail")
        self._draw_box(0, right_x, top_h, right_w, "Config (Global + Override)")

        visible_rows = top_h - 2
        if self.cursor < self.scroll:
            self.scroll = self.cursor
        if self.cursor >= self.scroll + visible_rows:
            self.scroll = self.cursor - visible_rows + 1

        groups_slice = self.state.group_order[self.scroll : self.scroll + visible_rows]
        if not groups_slice:
            self.stdscr.addnstr(1, left_x + 1, "No groups detected. Press r to refresh.", left_w - 2, self.attr_dim)

        for row, key in enumerate(groups_slice):
            idx = self.scroll + row
            override = self.state.overrides[key]
            variant = self.state.groups[key][override.variant_index]
            flag = "x" if override.enabled else "-"
            label_core = key.split("__", 1)[-1].replace("_", "-")
            quant = variant.quant_guess or "?"
            total_variants = len(self.state.groups[key])
            prefix = f"{idx + 1:>3} [{flag}] "
            suffix = f" ({override.variant_index + 1}/{total_variants}) {quant}"
            budget = max(5, (left_w - 2) - len(prefix) - len(suffix))
            core = self._truncate(label_core, budget)
            line = f"{prefix}{core}{suffix}"

            if idx == self.cursor:
                attr = self.attr_list_selected
            else:
                attr = self.attr_list_normal if override.enabled else self.attr_dim
            self.stdscr.addnstr(row + 1, left_x + 1, line, left_w - 2, attr)

        active = self.current_variant()
        detail_lines: List[str] = []
        if active is None:
            detail_lines.append("No active model")
        else:
            detail_lines.extend(
                [
                    f"publisher: {active.publisher or 'unknown'}",
                    f"model_family: {active.model_family or 'unknown'}",
                    f"filename: {active.filename}",
                    f"quant_guess: {active.quant_guess or 'unknown'}",
                    f"tags: {', '.join(active.tags_detected) if active.tags_detected else '-'}",
                    f"source: {active.source_path_abs}",
                    "",
                    "Warnings:",
                ]
            )
            warnings = active.warnings or ["none"]
            for warning in warnings[:6]:
                detail_lines.append(f"- {warning}")

        for row, line in enumerate(detail_lines[: top_h - 2]):
            self.stdscr.addnstr(row + 1, mid_x + 1, line, mid_w - 2, self.attr_text)

        cfg = self.state.global_config
        profile = _profile_from_flags(cfg.prefer_small, cfg.prefer_quality)

        key = self.current_group_key()
        override = self.state.overrides.get(key) if key else None
        override_lines = [
            f"enabled: {override.enabled if override else '-'}",
            f"name_override: {override.name_override if override and override.name_override else '-'}",
            f"num_ctx: {override.num_ctx if override and override.num_ctx is not None else 'inherit'}",
            f"temperature: {override.temperature if override and override.temperature is not None else 'inherit'}",
            f"top_p: {override.top_p if override and override.top_p is not None else 'inherit'}",
            f"system: {'set' if override and override.system else 'inherit'}",
        ]

        right_lines = [
            f"lmstudio: {self.state.lmstudio_root}",
            f"output: {cfg.output_dir}",
            f"force: {cfg.force}",
            f"name_mode: {cfg.name_mode}",
            f"prefix/suffix: {cfg.prefix or '-'} / {cfg.suffix or '-'}",
            f"profile: {profile}",
            f"preferred_quants: {','.join(cfg.preferred_quants)}",
            f"global num_ctx/temp/top_p: {cfg.num_ctx}/{cfg.temperature}/{cfg.top_p}",
            f"global system: {'set' if cfg.system else '-'}",
            f"only/exclude/limit: {cfg.only or '-'} / {cfg.exclude or '-'} / {cfg.limit or '-'}",
            f"ollama: {'ok' if self.ollama_available else 'missing'}",
            "",
            "Override (active group):",
            *override_lines,
        ]

        if self.state.last_result:
            right_lines.extend(
                [
                    "",
                    f"Last run processed: {self.state.last_result.summary.variants_processed}",
                    f"Last run created_ok: {self.state.last_result.summary.created_ok}",
                ]
            )

        for row, line in enumerate(right_lines[: top_h - 2]):
            self.stdscr.addnstr(row + 1, right_x + 1, line, right_w - 2, self.attr_text)

        status = self.state.status or "Ready"
        help_line = (
            "Arrows: move/cycle | Space: enable | n/k/t/p/s override | g global | / only | x exclude | "
            "r refresh | v preview | G generate | c/C create | ? help | q quit"
        )

        self.stdscr.addnstr(h - 2, 0, status, w - 1, self.attr_title)
        self.stdscr.addnstr(h - 1, 0, help_line, w - 1, self.attr_dim)
        self.stdscr.refresh()

    def show_help(self) -> None:
        lines = [
            "Navigation:",
            "  Up/Down: seleccionar grupo",
            "  Left/Right: cambiar variante del grupo",
            "  Space: activar/desactivar grupo",
            "",
            "Overrides (grupo activo):",
            "  n: name override",
            "  k: num_ctx override",
            "  t: temperature override",
            "  p: top_p override",
            "  s: system override",
            "",
            "Global:",
            "  g: menú global",
            "  /: filtro only",
            "  x: filtro exclude",
            "  r: refrescar discovery",
            "",
            "Actions:",
            "  v: preview Modelfile/nombre",
            "  G: generar artefactos (sin create)",
            "  c/C o Enter: generar + ejecutar create (confirmado)",
            "  q: salir",
        ]
        self.popup("Help", lines)

    def run(self) -> int:
        curses.curs_set(0)
        self.stdscr.keypad(True)

        while not self.quit_requested:
            self.draw()
            ch = self.stdscr.getch()

            if ch in (ord("q"), ord("Q")):
                self.quit_requested = True
                continue

            # Keep "k" free for num_ctx override.
            if ch in (curses.KEY_UP,):
                if self.cursor > 0:
                    self.cursor -= 1
                continue

            if ch in (curses.KEY_DOWN, ord("j")):
                if self.cursor < max(0, len(self.state.group_order) - 1):
                    self.cursor += 1
                continue

            if ch in (curses.KEY_LEFT, ord("h")):
                self.cycle_variant(-1)
                continue

            if ch in (curses.KEY_RIGHT, ord("l")):
                self.cycle_variant(1)
                continue

            if ch == ord(" "):
                self.toggle_enabled()
                continue

            if ch == ord("n"):
                self.edit_model_override("name_override")
                continue

            if ch in (ord("k"), ord("K")):
                self.edit_model_override("num_ctx")
                continue

            if ch == ord("t"):
                self.edit_model_override("temperature")
                continue

            if ch == ord("p"):
                self.edit_model_override("top_p")
                continue

            if ch == ord("s"):
                self.edit_model_override("system")
                continue

            if ch == ord("g"):
                self.edit_global()
                continue

            if ch == ord("/"):
                self.set_filters(only=True)
                continue

            if ch == ord("x"):
                self.set_filters(only=False)
                continue

            if ch == ord("r"):
                self.refresh_discovery()
                continue

            if ch == ord("v"):
                self.preview_active()
                continue

            if ch == ord("G"):
                self.run_generation(create=False)
                continue

            if ch in (ord("c"), ord("C"), 10, 13, curses.KEY_ENTER):
                self.run_generation(create=True)
                continue

            if ch in (ord("?"),):
                self.show_help()
                continue

        return 0


def run_tui(args: argparse.Namespace) -> int:
    if curses is None:
        print("Error: curses no está disponible en este entorno.", file=sys.stderr)
        return 2

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("Error: --tui requiere una terminal interactiva (TTY).", file=sys.stderr)
        return 2

    exit_code = 0

    def _wrapped(stdscr: Any) -> int:
        app = TerminalImporterApp(stdscr, args)
        return app.run()

    try:
        exit_code = curses.wrapper(_wrapped)
    except KeyboardInterrupt:
        exit_code = 130
    except Exception as exc:
        print(f"Error en TUI: {exc}", file=sys.stderr)
        exit_code = 1

    return int(exit_code)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.tui:
        return run_tui(args)

    logger = Console(verbose=args.verbose, json_mode=args.json)
    return run_cli(args, logger)


if __name__ == "__main__":
    sys.exit(main())

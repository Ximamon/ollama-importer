#!/usr/bin/env python3
"""LM Studio GGUF -> Ollama import assistant.

This utility discovers local GGUF files inside an LM Studio models directory,
groups likely variants of the same base model, selects preferred candidates via
an operational heuristic, and generates Ollama-ready artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
            "selection_reason": self.selection_reason,
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
        help="Comma-separated quant priority list (default: q5_k_m,q4_k_m,q4_k_s,q8_0,q6_k,f16,fp16)",
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


def dedupe_tokens(tokens: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result


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


def assign_unique_names(
    variants: List[VariantRecord],
    mode: str,
    prefix: str,
    suffix: str,
) -> None:
    used: Dict[str, int] = {}
    for variant in variants:
        base = build_ollama_name(variant, mode=mode, prefix=prefix, suffix=suffix)
        candidate = base
        counter = 2
        while candidate in used:
            candidate = slugify(f"{base}-{counter}", max_len=96)
            counter += 1
        used[candidate] = 1
        variant.proposed_ollama_name = candidate


def render_modelfile_content(args: argparse.Namespace, gguf_abs: Path) -> str:
    lines = [f"FROM {gguf_abs}"]
    if args.num_ctx is not None:
        lines.append(f"PARAMETER num_ctx {args.num_ctx}")
    if args.temperature is not None:
        lines.append(f"PARAMETER temperature {args.temperature}")
    if args.top_p is not None:
        lines.append(f"PARAMETER top_p {args.top_p}")
    if args.system is not None:
        lines.append(f"SYSTEM {json.dumps(args.system)}")
    return "\n".join(lines) + "\n"


def render_variant_readme(variant: VariantRecord, modelfile_content: str) -> str:
    status = "Seleccionada en su grupo" if variant.selected_in_group else "No seleccionada en su grupo"
    manual_create = (
        f"ollama create {variant.proposed_ollama_name} -f "
        f"{variant.modelfile_path if variant.modelfile_path else '<Modelfile>'}"
    )
    run_cmd = f"ollama run {variant.proposed_ollama_name}"

    warnings = variant.warnings or ["Sin advertencias registradas."]
    warning_lines = "\n".join(f"- {item}" for item in warnings)

    return f"""# {variant.proposed_ollama_name}

## Source GGUF
- Ruta original: `{variant.source_path}`
- Ruta absoluta: `{variant.source_path_abs}`
- Existe en disco: `{variant.exists}`

## Grouping
- Group key: `{variant.group_key}`
- Motivo de agrupación: tokens normalizados y eliminación prudente de sufijos de quantización.
- Estado de selección: **{status}**
- Razón de selección: {variant.selection_reason}

## Ollama
- Nombre propuesto: `{variant.proposed_ollama_name}`

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


def render_batch_script(entries: List[VariantRecord], selected_only: bool) -> str:
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = Console(verbose=args.verbose, json_mode=args.json)

    lmstudio_root = Path(args.lmstudio_dir).expanduser().resolve(strict=False)
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)

    logger.info(f"LM Studio dir: {lmstudio_root}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Mode: {'dry-run' if args.dry_run else 'create'}")

    summary = Summary()

    discovered_paths = find_gguf_files(lmstudio_root)
    summary.detected_models = len(discovered_paths)
    if not discovered_paths:
        logger.info("No se detectaron archivos GGUF.")

    variants = [parse_variant(path, lmstudio_root) for path in discovered_paths]

    variants, omitted_by_filter = apply_filters(
        variants,
        only=args.only,
        exclude=args.exclude,
        limit=args.limit,
    )
    summary.omitted += omitted_by_filter

    grouped = group_variants(variants)
    summary.groups_detected = len(grouped)
    summary.grouped_models = sum(len(items) for items in grouped.values())

    preferred_quants = parse_preferred_quants(args.preferred_quants)

    selected_map = select_variants(
        grouped,
        preferred_quants=preferred_quants,
        prefer_small=args.prefer_small,
        prefer_quality=args.prefer_quality,
    )

    selected_count = len(selected_map)
    summary.selected_variants = selected_count

    to_process: List[VariantRecord] = []
    if args.selected_only:
        for key in sorted(selected_map):
            to_process.append(selected_map[key])
    else:
        for key in sorted(grouped):
            to_process.extend(sorted(grouped[key], key=lambda item: str(item.source_path_abs)))

    assign_unique_names(to_process, mode=args.name_mode, prefix=args.prefix, suffix=args.suffix)

    if args.create:
        if shutil.which("ollama") is None:
            logger.info("Error: 'ollama' no está disponible en PATH. Se omite --create.")
            args.create = False
            args.dry_run = True
            for variant in to_process:
                variant.warnings.append("--create solicitado, pero ollama no está instalado/accesible.")

    output_dir.mkdir(parents=True, exist_ok=True)

    generated_at = utc_now_iso()
    processed_metadata: List[Dict[str, Any]] = []
    selected_metadata: List[Dict[str, Any]] = []

    for variant in to_process:
        variant.generated_at = generated_at
        variant_dir = output_dir / variant.proposed_ollama_name

        if should_skip_variant_dir(variant_dir, force=args.force):
            variant.skipped = True
            variant.skip_reason = "Directorio de salida ya existe; use --force para sobrescribir."
            variant.warnings.append(variant.skip_reason)
            summary.omitted += 1
            processed_metadata.append(variant.to_metadata())
            if variant.selected_in_group:
                selected_metadata.append(variant.to_metadata())
            continue

        variant_dir.mkdir(parents=True, exist_ok=True)

        modelfile_path = variant_dir / "Modelfile"
        metadata_path = variant_dir / "metadata.json"
        readme_path = variant_dir / "README.md"

        modelfile_content = render_modelfile_content(args, variant.source_path_abs)
        modelfile_written = write_text_file(modelfile_path, modelfile_content, force=args.force)

        if modelfile_written:
            summary.modelfiles_generated += 1

        variant.modelfile_path = modelfile_path
        variant.readme_path = readme_path

        readme_content = render_variant_readme(variant, modelfile_content)
        _ = write_text_file(readme_path, readme_content, force=args.force)

        if not variant.exists:
            variant.warnings.append("GGUF no existe al momento de generar artefactos.")

        if args.create and variant.exists:
            execute_create(variant, modelfile_path, logger)
        else:
            variant.create_attempted = False
            variant.create_success = None

        metadata_content = json.dumps(variant.to_metadata(), ensure_ascii=False, indent=2)
        _ = write_text_file(metadata_path, metadata_content + "\n", force=args.force)

        processed_metadata.append(variant.to_metadata())
        if variant.selected_in_group:
            selected_metadata.append(variant.to_metadata())

        summary.variants_processed += 1
        if variant.create_attempted:
            if variant.create_success:
                summary.created_ok += 1
            elif variant.create_success is False:
                summary.created_failed += 1

    index_payload = {
        "generated_at": generated_at,
        "lmstudio_dir": str(lmstudio_root),
        "output_dir": str(output_dir),
        "dry_run": args.dry_run,
        "selected_only": args.selected_only,
        "name_mode": args.name_mode,
        "preferred_quants": preferred_quants,
        "selection_profile": selection_profile_name(args.prefer_small, args.prefer_quality),
        "models": processed_metadata,
        "summary": summary.to_dict(),
    }

    selected_payload = {
        "generated_at": generated_at,
        "lmstudio_dir": str(lmstudio_root),
        "output_dir": str(output_dir),
        "models": selected_metadata,
    }

    index_path = output_dir / "index.json"
    selected_path = output_dir / "selected_models.json"
    batch_path = output_dir / "import_all.sh"

    if not write_text_file(index_path, json.dumps(index_payload, ensure_ascii=False, indent=2) + "\n", force=args.force):
        summary.omitted += 1
    if not write_text_file(selected_path, json.dumps(selected_payload, ensure_ascii=False, indent=2) + "\n", force=args.force):
        summary.omitted += 1

    batch_content = render_batch_script(to_process, selected_only=args.selected_only)
    if not write_text_file(batch_path, batch_content, force=args.force):
        summary.omitted += 1

    try:
        batch_path.chmod(0o755)
    except OSError:
        pass

    if args.json:
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    else:
        summarize_to_console(summary, logger)
        logger.info("")
        logger.info(f"Artifacts index: {index_path}")
        logger.info(f"Selected index: {selected_path}")
        logger.info(f"Batch script: {batch_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

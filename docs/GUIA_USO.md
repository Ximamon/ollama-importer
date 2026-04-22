# Guía de Uso: `lmstudio_to_ollama.py`

Esta guía explica en detalle cómo funciona el script, cómo usar el modo CLI y cómo controlar la TUI (`--tui`).

## 1) Objetivo de la herramienta

`lmstudio_to_ollama.py` ayuda a migrar/gestionar modelos GGUF que tienes en LM Studio para prepararlos en Ollama sin mover ni copiar los archivos de origen.

Hace esto:
- Descubre `*.gguf` en `~/.lmstudio/models` (o ruta configurada).
- Intenta inferir metadatos (`publisher`, `model_family`, `quant_guess`, tags).
- Agrupa variantes del mismo modelo base.
- Elige una variante preferida por heurística configurable.
- Genera artefactos para Ollama (`Modelfile`, `metadata.json`, `README.md`, `index.json`, `selected_models.json`, `import_all.sh`).
- Opcionalmente ejecuta `ollama create`.

## 2) Requisitos

- Python 3.11+
- macOS/Linux
- `ollama` solo si vas a crear modelos (`--create` o acción `c/C` en TUI)

## 3) Estructura de salida

Por defecto genera en `./ollama_imports`:

- Por modelo/variante:
  - `./ollama_imports/<ollama-name>/Modelfile`
  - `./ollama_imports/<ollama-name>/metadata.json`
  - `./ollama_imports/<ollama-name>/README.md`
- Global:
  - `./ollama_imports/index.json`
  - `./ollama_imports/selected_models.json`
  - `./ollama_imports/import_all.sh`

## 4) Modo CLI (no interactivo)

### Comando mínimo

```bash
python3 lmstudio_to_ollama.py
```

### Patrones frecuentes

```bash
# Solo preparar artefactos (default = dry-run)
python3 lmstudio_to_ollama.py --only qwen

# Procesar todas las variantes
python3 lmstudio_to_ollama.py --all-variants

# Ejecutar creación real en Ollama
python3 lmstudio_to_ollama.py --create --selected-only

# Cambiar orden de quants preferidas
python3 lmstudio_to_ollama.py --preferred-quants q4_k_m,q5_k_m,q8_0

# Aplicar parámetros al Modelfile
python3 lmstudio_to_ollama.py --num-ctx 8192 --temperature 0.6 --top-p 0.95
```

### Flags más importantes

- Descubrimiento/filtro:
  - `--lmstudio-dir`
  - `--only`
  - `--exclude`
  - `--limit`
- Selección:
  - `--selected-only` / `--all-variants`
  - `--preferred-quants`
  - `--prefer-small` / `--prefer-quality`
- Nombre Ollama:
  - `--name-mode variant|best`
  - `--prefix`
  - `--suffix`
- Modelfile:
  - `--num-ctx`
  - `--temperature`
  - `--top-p`
  - `--system`
- Ejecución/salida:
  - `--dry-run` (default)
  - `--create`
  - `--output-dir`
  - `--force`
  - `--verbose`
  - `--json`

## 5) Modo TUI (`--tui`)

### Arranque

```bash
python3 lmstudio_to_ollama.py --tui
```

### Flujo recomendado

1. Navega por grupos con `↑/↓`.
2. Cambia variante del grupo activo con `←/→`.
3. Activa/desactiva grupos con `Space`.
4. Ajusta overrides por modelo (`n`, `c`, `t`, `p`, `s`).
5. Ajusta configuración global con `g`.
6. Previsualiza con `v`.
7. Genera artefactos con `G`.
8. Cuando esté validado, crea en Ollama con `c`/`C` o `Enter` (pide confirmación).

### Atajos de teclado

- Navegación:
  - `↑ / ↓`: mover cursor por grupos
  - `← / →`: variante anterior/siguiente del grupo activo
  - `Space`: activar/desactivar grupo
- Overrides del grupo activo:
  - `n`: `name_override`
  - `k`: `num_ctx`
  - `t`: `temperature`
  - `p`: `top_p`
  - `s`: `system`
- Config global y filtros:
  - `g`: menú global
  - `/`: set filtro `only`
  - `x`: set filtro `exclude`
  - `r`: refrescar discovery
- Acciones:
  - `v`: preview de nombre + Modelfile
  - `G`: generar artefactos (sin create)
  - `c` / `C` / `Enter`: generar y ejecutar `ollama create` (confirmado)
  - `?`: ayuda
  - `q`: salir

## 6) Qué se guarda en `metadata.json`

Además de los metadatos base, la sesión interactiva añade:

- `selected_by_user`: si el usuario marcó ese grupo/modelo en TUI
- `overrides_applied`: solo overrides explícitos aplicados
- `effective_parameters`: parámetros finales efectivos usados para generar Modelfile

## 7) Importación por lote con `import_all.sh`

Desde raíz del repo:

```bash
./import_all.sh ./ollama_imports
EXECUTE=1 ./import_all.sh ./ollama_imports
```

Variables útiles:
- `EXECUTE=0|1` (default `0`)
- `SELECTED_ONLY=1|0` (default `1`)
- `NAME_FILTER=<texto>`
- `INDEX_FILE=<ruta_json>`

## 8) Solución de problemas (TUI)

### Se ve negro o “vacío” el panel de modelos

Pasos:
1. Asegúrate de tener tamaño suficiente (`>= 96x18`).
2. Pulsa `r` para refrescar discovery.
3. Verifica que realmente hay modelos detectados en la barra inferior.
4. Si usas tmux/screen, prueba fuera de multiplexer.

Opcional: activar color explícito (por defecto la TUI usa modo monocromo seguro):

```bash
LMSTUDIO_TUI_COLOR=1 python3 lmstudio_to_ollama.py --tui
```

### No aparece ningún modelo

- Revisa ruta con `--lmstudio-dir`.
- Revisa filtros `only/exclude/limit`.
- Comprueba que existan archivos `*.gguf`.

### `create` falla

- Verifica instalación de `ollama` y `PATH`.
- Revisa `stderr` por modelo en `metadata.json`.
- Si un modelo falla, el proceso sigue con los demás.

## 9) Seguridad y límites

- No copia, no mueve y no borra GGUF.
- No migra presets/configuración interna de LM Studio.
- No asume compatibilidad total LM Studio↔Ollama más allá del archivo GGUF.
- No genera `TEMPLATE` complejos automáticamente.
- La selección de quant es una heurística operativa (no verdad universal).

# LM Studio -> Ollama GGUF Importer

Herramienta para detectar modelos GGUF descargados por LM Studio, agrupar variantes y generar artefactos listos para importar en Ollama.

## Documentación ampliada
- Guía completa de funcionamiento (CLI + TUI): [docs/GUIA_USO.md](docs/GUIA_USO.md)

## Qué hace
- Escanea `~/.lmstudio/models` (o `--lmstudio-dir`) buscando `*.gguf`.
- Infiere metadatos de ruta/nombre: publisher, familia, quant y tags semánticos.
- Agrupa variantes de forma prudente para evitar mezclar modelos distintos.
- Selecciona variante preferida por grupo con heurística operativa configurable.
- Genera por variante:
  - `Modelfile`
  - `metadata.json`
  - `README.md`
- Genera global:
  - `index.json`
  - `selected_models.json`
  - `import_all.sh`
- Soporta dos modos:
  - CLI no interactivo (flags)
  - TUI interactivo en terminal (`--tui`, curses)

## Requisitos
- Python 3.11+
- `ollama` instalado solo si vas a ejecutar `--create` o `Create` en TUI
- macOS/Linux con terminal compatible

## Modo CLI (no interactivo)
Ejemplo rápido:
```bash
python3 lmstudio_to_ollama.py
```

Flags principales:
- `--lmstudio-dir <ruta>`
- `--output-dir <ruta>`
- `--force`
- `--dry-run` / `--create`
- `--selected-only` / `--all-variants`
- `--only <texto>`
- `--exclude <texto>`
- `--limit <N>`
- `--preferred-quants q5_k_m,q4_k_m,...`
- `--prefer-small`
- `--prefer-quality`
- `--name-mode variant|best`
- `--prefix <texto>`
- `--suffix <texto>`
- `--num-ctx 8192`
- `--temperature 0.7`
- `--top-p 0.95`
- `--system "You are a helpful assistant."`
- `--verbose`
- `--json`

## Modo TUI (`--tui`)
Lanzar:
```bash
python3 lmstudio_to_ollama.py --tui
```

La TUI está centrada en:
- seleccionar/deseleccionar grupos
- cambiar variante por grupo
- editar overrides por modelo
- editar configuración global
- previsualizar `Modelfile` y nombre final
- generar artefactos y luego confirmar create

### Atajos de teclado
- `↑` / `↓`: moverse por grupos
- `←` / `→`: cambiar variante del grupo activo
- `Space`: activar/desactivar grupo
- `n`: override de nombre (modelo activo)
- `k`: override `num_ctx` (modelo activo)
- `t`: override `temperature` (modelo activo)
- `p`: override `top_p` (modelo activo)
- `s`: override `system` (modelo activo)
- `g`: menú de configuración global
- `/`: set filtro `only`
- `x`: set filtro `exclude`
- `r`: refrescar discovery
- `v`: preview de nombre + `Modelfile`
- `G`: generar artefactos (sin `ollama create`)
- `c` / `C` / `Enter`: generar y ejecutar `ollama create` con confirmación explícita
- `?`: ayuda
- `q`: salir

## Flujo recomendado (TUI)
1. Abrir `--tui`.
2. Seleccionar grupos y variantes.
3. Ajustar overrides por modelo/global.
4. Previsualizar (`v`).
5. Generar artefactos (`G`).
6. Revisar `index.json` / `selected_models.json`.
7. Ejecutar create (`c`/`C` o `Enter`) cuando esté validado.

## Salida generada
Por variante:
- `./ollama_imports/<ollama-name>/Modelfile`
- `./ollama_imports/<ollama-name>/metadata.json`
- `./ollama_imports/<ollama-name>/README.md`

Global:
- `./ollama_imports/index.json`
- `./ollama_imports/selected_models.json`
- `./ollama_imports/import_all.sh`

`metadata.json` incluye trazabilidad interactiva adicional:
- `selected_by_user`
- `overrides_applied`
- `effective_parameters`

## Importación batch
Script genérico de raíz:
```bash
./import_all.sh ./ollama_imports
EXECUTE=1 ./import_all.sh ./ollama_imports
```

Variables útiles:
- `EXECUTE=0|1` (default `0`)
- `SELECTED_ONLY=1|0` (default `1`)
- `NAME_FILTER=qwen`
- `INDEX_FILE=/ruta/personalizada/index.json`

## Ejemplos CLI
```bash
python3 lmstudio_to_ollama.py
python3 lmstudio_to_ollama.py --dry-run
python3 lmstudio_to_ollama.py --selected-only
python3 lmstudio_to_ollama.py --all-variants
python3 lmstudio_to_ollama.py --only qwen
python3 lmstudio_to_ollama.py --exclude vision
python3 lmstudio_to_ollama.py --create --only coder
python3 lmstudio_to_ollama.py --create --preferred-quants q4_k_m,q5_k_m
python3 lmstudio_to_ollama.py --create --num-ctx 8192 --temperature 0.6
python3 lmstudio_to_ollama.py --output-dir ./salida --force --verbose
```

## Limitaciones importantes
- Solo asume compatibilidad LM Studio <-> Ollama a nivel GGUF.
- No migra presets/configuración de LM Studio.
- No genera `TEMPLATE` complejos automáticamente.
- No copia, mueve ni borra GGUF.
- Si algo no es inferible con seguridad, se marca en `warnings`/metadata.
- La heurística de selección es operativa; no afirma una quant universalmente mejor.

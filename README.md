# LM Studio -> Ollama GGUF Importer

Herramienta para detectar modelos GGUF descargados por LM Studio, agrupar variantes parecidas y generar artefactos listos para importarlos en Ollama de forma segura y auditable.

## Qué hace
- Descubre archivos `*.gguf` dentro de `~/.lmstudio/models` (o la ruta indicada).
- Intenta inferir metadatos de ruta y nombre (publisher, familia, quant, tags).
- Agrupa variantes del mismo modelo con una heurística prudente.
- Selecciona una variante preferida por grupo con una heurística operativa configurable.
- Genera por variante:
  - `Modelfile`
  - `metadata.json`
  - `README.md`
- Genera a nivel global:
  - `index.json`
  - `selected_models.json`
  - `import_all.sh`
- Opcionalmente ejecuta `ollama create`.

## Uso rápido
```bash
python3 lmstudio_to_ollama.py
python3 lmstudio_to_ollama.py --all-variants --output-dir ./salida
python3 lmstudio_to_ollama.py --create --only coder --preferred-quants q4_k_m,q5_k_m
```

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

## Revisión antes de crear en Ollama
1. Inspecciona `selected_models.json` e `index.json`.
2. Revisa `README.md` y `metadata.json` dentro de cada carpeta generada.
3. Verifica el `FROM` absoluto de cada `Modelfile`.
4. Ejecuta importación en lote con seguridad:
   ```bash
   ./ollama_imports/import_all.sh
   EXECUTE=1 ./ollama_imports/import_all.sh
   ```

## Limitaciones importantes
- No asume compatibilidad total LM Studio <-> Ollama más allá del GGUF.
- No migra presets/configuración interna de LM Studio.
- No genera `TEMPLATE` complejos por defecto.
- No mueve, copia ni borra archivos GGUF.
- Si algo no se puede inferir con seguridad, queda marcado en `warnings`/metadata.

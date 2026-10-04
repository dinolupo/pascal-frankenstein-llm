# 🔬 Qwen Pascal Lab (Dual GPU Frankenstein Benchmark & Stress Test)

Laboratorio isolato per il testing, il benchmarking e la risoluzione dei problemi di stabilità di **Qwen 35B MoE Frankenstein** su architettura NVIDIA Pascal (GTX 1080 Ti 11GB + GTX 1070 8GB).

---

## 🎯 Obiettivo
1. **Risolvere il blocco/deadlock della chat in Open WebUI** dopo poche iterazioni sul modello Vision (`Qwen36 her vis`).
2. **Misurare le prestazioni reali in tokens/secondo** (verificare i 40–50 t/s osservati con MoE cache dual-GPU + MTP).
3. **Stabilizzare un'unica configurazione eccellente per Open WebUI** (con supporto Vision) senza compromessi di memoria.

---

## 📂 Contenuto della cartella

- **`benchmark_stress.py`**: Script di benchmark multi-turno.
  - Simula conversazioni sequenziali fino a 10+ turni.
  - Monitora in tempo reale TTFT (Time To First Token), velocità di generazione effettiva (t/s) e VRAM GPU 0/1.
  - Rileva stalli o timeout.

---

## 🚀 Come Eseguire i Test

### 1. Avviare il Server LLM
In un terminale (o tmux):
```bash
genai-llamacpp.sh
```

### 2. Eseguire lo Stress Test Multi-Turno
In un altro terminale, spostati in questa cartella ed esegui:

```bash
cd ~/proj/genAI/qwen-pascal-lab

# Test sul profilo Vision (quello che si bloccava in Open WebUI)
./benchmark_stress.py --model "Qwen36 her vis" --turns 10 --max-tokens 384

# Test comparativo sul profilo testo
./benchmark_stress.py --model "Qwen36 her jin" --turns 10 --max-tokens 384
```

---

## 🔍 Parametri Chiave Individuati per il Fix
Nel file `~/.local/share/pascal-frankenstein-llm/config/llama-models.ini`:

Nel profilo `[Qwen3.6-35B-vis]` mancavano due parametri critici:
```ini
no-context-shift = true
n = 32768
```
Senza `no-context-shift = true`, l'inferenza speculativa MTP (`draft-mtp`) entra in deadlock durante il riarrangiamento della KV-cache nei dialoghi lunghi.

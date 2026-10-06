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
cd /path/to/pascal-frankenstein-llm/scripts

# Test sul profilo Vision (quello che si bloccava in Open WebUI)
./benchmark_stress.py --model "Qwen36 her vis" --turns 10 --max-tokens 384

# Test comparativo sul profilo testo
./benchmark_stress.py --model "Qwen36 her jin" --turns 10 --max-tokens 384
```

### Gate per aggiornamenti del fork

Il benchmark usa `usage.completion_tokens` e i timings nativi del server:
i chunk SSE non sono token. Mantiene separati reasoning e risposta, e termina
con codice nonzero su errori HTTP/SSE, timeout, stream incompleti, metriche
mancanti o risposte senza contenuto finale. Il timeout misura inattivita dello
stream, non la durata totale della richiesta. La telemetria VRAM e obbligatoria
ed e campionata ai confini dei turni; per i picchi usare un monitor esterno.

La chiave viene letta da `LLAMA_API_KEY` senza essere salvata nel report.
I report contengono testo generato: conservarli localmente, non pubblicarli
automaticamente. Usare una directory di risultati esterna a Git.

```bash
python3 -m unittest discover -s scripts -p 'test_benchmark_stress.py'
python3 scripts/benchmark_stress.py \
  --url http://127.0.0.1:18081 --model "Qwen36 her jin" \
  --turns 10 --max-tokens 384 --reasoning off \
  --temperature 0.6 --seed 123 --repeats 3 --timeout 180 \
  --output /path/to/results/baseline-text.json
# Stessi parametri sul binario candidato:
python3 scripts/benchmark_stress.py \
  --url http://127.0.0.1:18081 --model "Qwen36 her jin" \
  --turns 10 --max-tokens 384 --reasoning off \
  --temperature 0.6 --seed 123 --repeats 3 --timeout 180 \
  --baseline /path/to/results/baseline-text.json \
  --output /path/to/results/candidate-text.json
```

Ripetere separatamente per `Qwen36 her vis`: una conversazione testuale con
projector caricato non sostituisce un test funzionale con immagini.
Il gate richiede tre repliche complete e workload identico: blocca perdite
oltre 5% in generation, oltre 10% in prefill o aumenti oltre 10% in TTFT.
Le statistiche includono la dispersione dei turni; non sono intervalli di
confidenza. Verificare i casi borderline e la correttezza separatamente.
Reasoning disabilitato nel controllo evita di esaurire 384 token prima della
risposta finale; aggiungere prove funzionali con reasoning attivo.
Ogni checkpoint deve usare gli stessi preset, template, pesi e warm-up.
Non inferire prestazioni a 120k token da questo breve workload multi-turno.

La modalita predefinita resta stretta. Per questa integrazione e stata approvata
l'accettazione delle sole regressioni TTFT spiegate, senza cambiare i preset.
Usare esplicitamente `--accept-latency-regression "Diagnosi e motivazione"`
nel benchmark o nel runner: generation/prefill, errori funzionali e misure
incomplete restano bloccanti. I report conservano violazioni, delta percentuali,
costo TTFT in secondi e motivazione. `passed` significa accettato secondo la
politica selezionata, non privo di regressioni: leggere anche
`performance_thresholds_passed` nel gate e `performance_assessment` nei report.
La modalita greedy resta un controllo separato, non sostituisce T=0,6.

Per eseguire automaticamente entrambi i profili sul binario candidato:

```bash
python3 scripts/run-integration-gate.py \
  --server llama.cpp/build-pascal-cuda/bin/llama-server \
  --preset "$HOME/.local/share/pascal-frankenstein-llm/config/llama-models.ini" \
  --results /path/to/results/baseline
# Aggiungere --baseline /path/to/results/baseline per ogni checkpoint successivo.
```

Il runner rifiuta altri processi llama attivi e directory di risultati esistenti,
avvia un server loopback sulla porta 18081, esegue warm-up e controlli aritmetici
multi-turno/EOG e un'immagine rossa sul profilo vision, poi tre stress replications
per profilo. Campiona VRAM e memoria host ogni due secondi e arresta il proprio
server anche in caso di fallimento. Non modifica l'installazione. Usa i due alias
operativi indicati sopra; un preset con alias differenti richiede adattamento
esplicito. Questi smoke test non sostituiscono backend/logits o long-context.

### Regressione padding MMQ

Il catalogo di test-backend-ops contiene i due casi dimensionali segnalati in
ggml-org/llama.cpp#29941: Q4_0, 512 esperti, 10 selezionati, m=640, n=508,
k=2560, input con/senza broadcast. Gli ID randomizzati coprono tutti gli
esperti e sono una vista strided, come nel caso originale.
Eseguire senza server modello attivo:

```bash
llama.cpp/build-pascal-cuda/bin/test-backend-ops test -b CUDA0 \
  -o MUL_MAT_ID -p 'n_mats=512.*m=640.*n=508'
llama.cpp/build-pascal-cuda/bin/test-backend-ops test -b CUDA1 \
  -o MUL_MAT_ID -p 'n_mats=512.*m=640.*n=508'
```

I casi sono relativamente grandi (circa 450 MiB di pesi ciascuno), quindi
usarli con un singolo worker e senza processi modello concorrenti.
Il catalogo copre separatamente gli ID hot/cold `-1`.

---

## 🔍 Parametri Chiave Individuati per il Fix
Nel file `~/.local/share/pascal-frankenstein-llm/config/llama-models.ini`:

Nel profilo `[Qwen3.6-35B-vis]` mancavano due parametri critici:
```ini
no-context-shift = true
n = 32768
```
Senza `no-context-shift = true`, l'inferenza speculativa MTP (`draft-mtp`) entra in deadlock durante il riarrangiamento della KV-cache nei dialoghi lunghi.

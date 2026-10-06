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

Per la certificazione finale aggiungere `--require-max-performance`. Il runner
verifica la preferenza PowerMizer tramite NV-CONTROL prima e durante le prove,
confrontando gli UUID con tutte le GPU CUDA. Registra i modi nella telemetria
e fallisce se una GPU non e su `1` o se manca copertura. Non cambia la politica.
Richiede il display X corrente e le librerie X11/XNVCtrl gia installate; nessuna
lettura mancante viene trattata come successo. Senza il flag il comportamento
del gate resta invariato. Verifica indipendente:

```bash
python3 scripts/nvidia_power.py
python3 -m unittest discover -s scripts -p 'test_nvidia_power.py'
```

Nel sistema provato, client `nvidia-settings` 615.71.09 e driver 580.178.04
mostrano letture discordanti: CLI `0`, NV-CONTROL `1` persistente anche
da una nuova connessione. Il gate usa la preferenza del driver X, non la
sola conferma testuale di assegnazione del client. Non confondere questa
preferenza con lo stato CUDA P2 o con una garanzia di clock massimo.
I confronti con le vecchie misure senza verifica della politica non
certificano uno speedup: servono controlli ravvicinati con la stessa politica.
Per controlli compilati in precedenza usare anche `--expected-source SHA`.
Il runner legge il commit incorporato in `--version` e lo risolve nella storia
Git; non attribuisce un vecchio binario all'HEAD corrente del checkout.
Conserva separatamente `checkout_sha` e `source_diff_applies_to_binary`.
Un'identita assente, non risolvibile o diversa da quella richiesta fallisce.
Un load con registrazione host massiva e polling GPU e andato in timeout.
Le repliche senza polling, con polling differito e con polling precoce
ripristinato sono passate: la causa resta irrisolta, non e dimostrato un
deadlock del monitor o un limite di registrazione del driver.
Con `GGML_CUDA_REGISTER_HOST` presente il runner campiona RAM durante il load
e differisce esplicitamente i probe GPU fino al warm-up completato, con
sincronizzazione per non lasciare query in volo al cambio modello. Il campo
`gpu_sampling_deferred` segnala i campioni senza dati GPU: non sono letture
riuscite o copertura energetica durante quel load. La politica e verificata
prima e dopo il load; durante tutte le misure torna la telemetria completa.
Con la registrazione spenta il comportamento di campionamento resta invariato.
Il monitor registra anche durata del probe GPU e timestamp di completamento:
due secondi sono la pausa tra campioni, non una garanzia di cadenza esatta.
I probe CLI conservano il timeout predefinito di 10 s. Per lo screening lungo
il chiamante puo specificare `gpu_probe_timeout=60`; il primo screening a
58.698 token aveva risposte corrette ma un probe in timeout, dunque il gate
resta registrato come fallito. La replica con timeout esplicito non elimina
quel fallimento; assenze/errori di telemetria continuano a fallire.

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

### Scheduler hot/cold e prefetch multi-GPU

Il target `test-moe-scheduler` verifica risultati numerici su grafi assegnati
CUDA0/CUDA1/CUDA0, tre esecuzioni dello stesso grafo, input tutti mascherati
e upload opzionale di pesi host. Eseguire senza altri processi modello:

```bash
env -u GGML_CUDA_P2P -u GGML_SCHED_PREFETCH_EXPERTS \
  llama.cpp/build-pascal-v060/bin/test-moe-scheduler masked
env -u GGML_CUDA_P2P -u GGML_SCHED_PREFETCH_EXPERTS \
  llama.cpp/build-pascal-v060/bin/test-moe-scheduler prefetch
env -u GGML_CUDA_P2P GGML_SCHED_PREFETCH_EXPERTS=1 \
  llama.cpp/build-pascal-v060/bin/test-moe-scheduler prefetch
```

Il percorso prefetch ricrea gli slot/eventi al cambio di GPU dopo aver
sincronizzato gli utenti precedenti: nessun P2P. Questo e un controllo numerico
sintetico, non una prova di speedup del prefetch sul modello.

### Codec opzionali CPU/CUDA Pascal

I preset del gate restano Q8 K/V. Per controllare separatamente TurboQuant e
il Q2_0 upstream senza cambiare l'installazione:

```bash
ctest --test-dir llama.cpp/build-pascal-v060 \
  -R '^test-turbo-quant$' --output-on-failure
env -u GGML_CUDA_P2P llama.cpp/build-pascal-v060/bin/test-backend-ops \
  -b CUDA0 -o TURBO_WHT,SET_ROWS,FLASH_ATTN_EXT,MUL_MAT \
  -p 'turbo|width=|type_a=(q2_0|tq3_1s|tq4_1s),type_b=f32,m=32,n=(1|32),k=256,'
```

Ripetere il secondo comando con `-b CUDA1`. Devono essere eseguiti 128 casi
per GPU, senza `not supported` o errori; il solo exit code non certifica
copertura, perche il runner upstream considera validi anche i casi saltati.
Il test CPU verifica 18 round trip con soglie numeriche esplicite.
Le cache Turbo sono lossy: queste verifiche non sostituiscono perplexity,
quality benchmark o contesti lunghi e non dimostrano speedup.
I matmul TQ CUDA usano dequantizzazione/cuBLAS, non kernel MMVQ/MMQ dedicati;
MUL_MAT_ID TQ conserva il fallback CPU.

---

## 🔍 Parametri Chiave Individuati per il Fix
Nel file `~/.local/share/pascal-frankenstein-llm/config/llama-models.ini`:

Nel profilo `[Qwen3.6-35B-vis]` mancavano due parametri critici:
```ini
no-context-shift = true
n = 32768
```
Senza `no-context-shift = true`, l'inferenza speculativa MTP (`draft-mtp`) entra in deadlock durante il riarrangiamento della KV-cache nei dialoghi lunghi.

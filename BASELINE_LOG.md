# Baseline e obiettivi

Registro iniziale delle misure effettuate in LM Studio su Windows. I valori
sono il riferimento da riprodurre e superare con i test nativi in WSL2.

## Hardware disponibile

- CPU: Intel Core i7-4790K (AVX2), 32 GB RAM DDR3.
- GPU 0: NVIDIA GeForce GTX 1080 Ti, 11 GB VRAM, Pascal (compute capability 6.1).
- GPU 1: NVIDIA GeForce GTX 1070, 8 GB VRAM, Pascal (compute capability 6.1).
- VRAM combinata rilevata da LM Studio: 19.54 GB.

## Ambiente WSL2 verificato

- Ubuntu 24.04.1 su WSL2; GPU passthrough operativo dal 30 agosto 2026.
- Driver NVIDIA Windows 581.80, CUDA massima esposta dal driver: 13.0.
- CUDA Toolkit 12.9 (`nvcc` 12.9.86), con target Pascal `compute_61` disponibile.
- Toolchain: GCC 13.3, GNU Make 4.3, CMake 3.28 e Ninja 1.11.
- Topologia GPU: GPU0↔GPU1 = `SYS`; CUDA P2P in lettura e scrittura = `NS`
  (non supportato). Non usare `GGML_CUDA_P2P` nella configurazione normale.

Il toolkit resta deliberatamente nella serie CUDA 12.x: Pascal (compute
capability 6.1) non è un target di compilazione della serie CUDA 13.

## Misure LM Studio

| Modello | Configurazione nota | Risultato osservato |
| --- | --- | --- |
| Qwen3.8-27B dense | Contesto 8k, GPU offload 52; altri parametri da registrare se necessari | 4–6 token/s; 15.93 GB di VRAM occupati su 19.54 GB |
| Qwen3-8B | Modello interamente residente nella GTX 1080 Ti | fino a 30 token/s |
| Qwen3.8-27B standard | GGUF `Q4_K_M` da `/mnt/e`; `llama-bench`, 52 layer GPU, split layer `10/7`, 3 ripetizioni | **131.44 ± 9.75 t/s** in prefill (`pp512`); **3.43 ± 0.05 t/s** in generation (`tg128`) |
| Qwen3.8-27B standard | GGUF `Q4_K_M` da `/mnt/e`; `llama-bench`, 52 layer GPU, split automatico, 3 ripetizioni | **151.91 ± 1.96 t/s** in prefill (`pp512`); **3.73 ± 0.16 t/s** in generation (`tg128`) |
| Qwen3.8-27B standard | GGUF `Q4_K_M` da `/mnt/e`; `llama-bench`, tutti i layer GPU (`-ngl 999`), split `10/7`, una ripetizione | **202.90 t/s** in prefill (`pp512`); **11.14 t/s** in generation (`tg128`) |
| Qwen3.8-27B standard | Chat reale con contesto allocato a 8192, tutti i layer GPU e split `10,7` | Caricamento riuscito; primo prompt: **89.2 t/s** e **11.9 t/s** in generation. |

La seconda misura conferma che una Pascal può generare velocemente quando il
modello e il percorso di inferenza restano locali a una GPU. Per il 27B il
vincolo principale da investigare è quindi la distribuzione tra GPU e RAM,
inclusi trasferimenti PCIe e sincronizzazioni per token.

## Obiettivo di lavoro

1. Riprodurre in WSL2 una baseline Qwen3.8-27B comparabile a LM Studio.
2. Registrare separatamente velocità di prefill e generazione, quantizzazione,
   contesto, split GPU e memoria usata.
3. Migliorare il dense 27B senza compromettere qualità o stabilità.
4. Valutare anche Qwen3.6-35B-A3B MoE con il fork `llama.cpp`, in
   particolare cache degli expert e prefetch CPU/GPU.

Il valore di 4–6 token/s è la baseline iniziale da battere; 30 token/s su 8B è
un riferimento del potenziale quando non intervengono offload e comunicazioni
tra dispositivi.

## Campagna certificata Qwen3.8 Flash Next (11 settembre 2026)

La prova del modello `unsloth/Qwen3.8-Flash-Next-GGUF`, quantizzazione
`UD-IQ3_XXS`, è stata chiusa con `llama-server` e richieste HTTP bounded sul
sistema Linux Mint nativo. Il modello principale è di circa 82 GB; sono stati
usati `--load-mode mmap`, `-fit off`, contesto 32k, una richiesta alla volta e
nessun processo concorrente. Sono state provate sia la configurazione
single-GPU stile Codacus sia la distribuzione dual-GPU asimmetrica del progetto.

La migliore configurazione API a 32k è risultata:

```text
-ts 10,7 -ncmoe 44 --moe-cache-slots 80,40
-ctk q8_0 -ctv q8_0 -b 512 -ub 256 -t 4
MTP disabilitato
prefill 12.60 tok/s
generation 1.95 tok/s
```

Con contesto 16k, la variante migliore ha raggiunto 11.10 tok/s in prefill e
2.15 tok/s in generation. Dieci ulteriori prove hanno variato:

- `-t 3/4/6` e `--no-sched-async-cpu`;
- cache `64,32`, `80,40`, `96,32` e `72,32`;
- `-ncmoe 43`, `44` e `99`;
- KV `q4_0/q8_0`, `q8_0/q4_0` e `q8_0/q8_0`;
- MTP attivo/disattivato;
- contesto 16k/32k;
- single-GPU e dual-GPU;
- batch 512 e 1024.

Il range completo è rimasto tra 1.71 e 2.15 tok/s in generation. Con MTP
l'accettazione draft era alta, ma il costo del draft non migliorava il
throughput end-to-end. Il risultato certifica che il modello funziona, ma che
su i7-4790K/DDR3 e due GPU Pascal il percorso mmap/MoE è limitato da CPU,
memoria, traffico page-cache/SSD e kernel CUDA legacy; non è stato trovato un
flag runtime capace di portarlo a 10 tok/s in generation.

Il piano completo, i comandi e i risultati riproducibili sono in
`QWEN38_FLASH_SERVER_TEST_PLAN.md`; il JSONL della campagna è
`qwen38_flash_sweep_results.jsonl`.

## Campagna finale WSL del 1 settembre 2026

Questa sezione contiene le misure conclusive ottenute dopo aver corretto la
policy energetica NVIDIA in Windows. Prima della correzione, durante la
generation la GTX 1080 Ti cadeva in P5 a 734–759 MHz core e 810 MHz memoria;
il driver riportava soltanto il clock-event `Idle`, senza power cap o thermal
slowdown. Impostando nel Pannello di controllo NVIDIA la modalita' di gestione
dell'alimentazione su **Preferisci prestazioni massime**, entrambe le GPU
mantengono i clock sotto carico:

- GTX 1080 Ti: P2, 1556 MHz core, 5005 MHz memoria;
- GTX 1070: P2, fino a 1949 MHz core, 3802 MHz memoria.

Come controllo indipendente, Qwen3-8B Q8_0 `tg128` e' passato da 4,42 a
21,55 t/s (+387,6%). I numeri raccolti prima della correzione restano solo
diagnostici e non vengono usati come baseline finale.

### Build usata

- Fork `llama.cpp`, branch `perf`, commit `d927e7dc1` (build
  `10125`).
- Build `Release`, CUDA Toolkit 12.9, target Pascal `sm_61`, CUDA Graphs
  disabilitata come previsto, NCCL non disponibile e P2P non usato.
- Configurazione riproducibile:

```bash
cd /home/dino/pascal-frankenstein-llm/llama.cpp
cmake -S . -B build-pascal-cuda -G Ninja \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=61 \
  -DGGML_NATIVE=ON \
  -DLLAMA_BUILD_TESTS=OFF
cmake --build build-pascal-cuda -j 4
```

Una seconda build con `GGML_CUDA_FORCE_MMQ=ON` e' stata creata in
`build-pascal-cuda-mmq` senza modificare i sorgenti, ma non e' il candidato
finale: a clock corretti e' risultata piu' lenta sul dense.

### Dense Qwen3.8-27B: benchmark finale

Modello:

```text
/mnt/e/lmstudio-models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf
```

Q4_K_M, 15,65 GiB, 27,32 B parametri. `llama-bench` usa batch 2048,
ubatch 512, 4 thread CPU, KV f16 e Flash Attention `auto` (default). I test
`pp512` e `tg128` sono sintetici e non allocano un server da 16k.

| Split layer | Ripetizioni | pp512 | tg128 | VRAM durante tg |
| --- | ---: | ---: | ---: | --- |
| Manuale `10/7` | 3 | 162,94 ± 1,70 t/s | 10,13 ± 0,40 t/s | circa 9,4/7,1 GiB |
| Automatico, candidato finale | 3 | **176,15 ± 4,52 t/s** | **10,31 ± 0,60 t/s** | 9.520/7.105 MiB |

Comando finale del benchmark:

```bash
cd /home/dino/pascal-frankenstein-llm
./llama.cpp/build-pascal-cuda/bin/llama-bench \
  -m /mnt/e/lmstudio-models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf \
  -ngl 999 -p 512 -n 128 -r 3
```

Durante la misura finale entrambe le GPU erano P2; restavano circa
1.587/963 MiB liberi. Il processo aveva circa 343 MiB RSS dopo l'offload, ma
WSL usava circa 14 GiB come page cache e circa 1 GiB di swap. WSL era limitato
a 15 GiB RAM + 4 GiB swap; questo rende lungo ogni reload da `/mnt/e`, ma non
entra nei t/s del benchmark.

Rispetto alla vecchia misura singola full-offload 202,90/11,14 t/s, il nuovo
risultato `r=3` e' -13,2% in prefill e -7,5% in generation. La vecchia misura
resta valida come osservazione, ma non aveva deviazione standard.

#### Varianti dense esplorative scartate

Misure `r=1`, usate solo per selezionare il candidato:

| Unica variabile rispetto al candidato | pp512 | tg128 | Esito |
| --- | ---: | ---: | --- |
| Flash Attention forzata `on`, split `10/7` | 123,03 | 10,44 | forte perdita prefill, guadagno tg non significativo |
| Split `11/6`, FA auto | 171,81 | 9,98 | peggiore in generation |
| Build FORCE_MMQ, split automatico | 170,33 | 9,63 | peggiore a clock corretti |

Lo split `tensor` non e' stato candidato a una prova prestazionale: e'
sperimentale, richiede Flash Attention e beneficia di NCCL, mentre questa
build non ha NCCL e la topologia GPU e' `SYS` senza P2P. Anche un draft model
separato per speculative decoding dense non e' stato caricato: al server 16k
restano soltanto 969/431 MiB liberi, quindi richiederebbe sacrificare il full
offload che produce il principale guadagno del dense.

### Dense: server 16k consigliato

Configurazione verificata: un solo slot, full offload, split automatico,
KV f16, Flash Attention `on`, batch/ubatch default 2048/512, thread server
automatici e bind esclusivamente loopback.

```bash
cd /home/dino/pascal-frankenstein-llm
./llama.cpp/build-pascal-cuda/bin/llama-server \
  -m /mnt/e/lmstudio-models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf \
  -c 16384 --parallel 1 -ngl 999 \
  -ctk f16 -ctv f16 -fa on \
  --host 127.0.0.1 --port 18080 --jinja --no-webui
```

Risultato verificato tramite `/health` e `/v1/chat/completions`:

- 74 token prompt: **89,02 t/s**;
- 188 token generation: **10,37 t/s**;
- VRAM post-richiesta: 10.138 MiB GPU0 e 7.637 MiB GPU1, con 969/431
  MiB liberi;
- RAM WSL: circa 1,3 GiB usati dal sistema/processi e 14 GiB di page cache;
- output: implementazione Python di binary search corretta, terminata
  normalmente, type hints e quattro assert validi.

Il prompt processing e' sostanzialmente uguale alla vecchia chat 8k
(89,02 contro 89,2 t/s, -0,2%); la generation e' 12,9% sotto 11,9 t/s, ma la
nuova misura usa context 16k e parametri completamente registrati.

Flash Attention e' necessaria per far entrare il server 16k con ubatch 512:
con `-fa off` l'allocazione dei compute buffer su GPU1 fallisce; `ubatch=128`
consente il load ma il primo decode fallisce creando il handle cuBLAS. Non e'
quindi un profilo stabile.

#### Limite di contesto dense

- **16k:** stabile e veloce; e' il contesto massimo pratico consigliato.
- **24k f16/FA off:** OOM sui compute buffer; ridurre batch a 512 non cambia
  l'allocazione. Ridurre ubatch a 128 e 64 porta il buffer a 315 e 157 MiB,
  ma GPU1 resta comunque senza margine.
- **32k q8/q8, FA on, auto split:** OOM sui compute buffer di GPU1.
- **32k q8/q8, FA on, split `11,6`:** il server carica con 11.061/6.883 MiB
  VRAM, ma su un prompt breve il prefill scende a 4,39 t/s e la generation a
  circa 0,03 t/s (14 token dopo diversi minuti). Il test e' stato cancellato
  pulitamente; fit verificato, stabilita' completa non rivendicata e profilo
  scartato come impraticabile.

### MoE Qwen3.6-35B-A3B: modelli e profili

Modello standard usato per baseline e cache:

```text
/home/dino/llm-models/pascal-tests/Qwen3.6-35B-A3B-Q4_K_M.gguf
```

- 19,70 GiB, 34,66 B parametri;
- SHA-256:
  `439fcb8266f37a035d2192d0fa773e59b177f379bd2bf90976419cea8c7dbb58`.

Profilo routing standard unificato, 320 token di decode coding+chat:

```text
/home/dino/pascal-frankenstein-llm/moe-traces/qwen36-35b-merged.csv
```

SHA-256:
`04f25d0f4ea68eab1e125062d963e5b4145a90bdb3c273dc0499cbe3243114ec`.

Il modello MTP separato e il profilo MTP specifico sono stati verificati ma
non sono il candidato finale:

- GGUF MTP SHA-256:
  `0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b`;
- trace MTP unificato, 256 token, SHA-256:
  `1ad2a9636abd70291a177f25fed859edb4d00365599db593bdea151f6f5e5e0f`.

### MoE: baseline e cache finale

Parametri comuni: `-ngl 99 -ncmoe 99`, split `10/7`, FA on, batch 2048,
ubatch 512, 4 thread, KV f16, `pp512`, `tg128`, tre ripetizioni. I layer
non-expert sono GPU; tutti gli expert partono in CPU/RAM.

| Configurazione | pp512 | tg128 | VRAM indicativa |
| --- | ---: | ---: | --- |
| Nessuna cache | **16,48 ± 3,69 t/s** | **0,66 ± 0,02 t/s** | 1.513/1.005 MiB |
| Cache statica, 112 slot/layer | **19,41 ± 6,10 t/s** | **1,51 ± 0,20 t/s** | 9.829/1.005 MiB |

La cache migliora il prefill del 17,8% e la generation del **128,8%**. E' un
incremento verificato del fork `perf`, ma il dense 27B resta circa 6,8 volte
piu' veloce in generation sintetica.

Comandi riproducibili:

```bash
# Baseline senza cache
./llama.cpp/build-pascal-cuda/bin/llama-bench \
  -m /home/dino/llm-models/pascal-tests/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  -ngl 99 -ncmoe 99 -ts 10/7 -fa on \
  -p 512 -n 128 -r 3

# Stessa configurazione, unica variabile: cache 112
GGML_MOE_CACHE_PROFILE=/home/dino/pascal-frankenstein-llm/moe-traces/qwen36-35b-merged.csv \
GGML_MOE_CACHE_SLOTS=112 \
./llama.cpp/build-pascal-cuda/bin/llama-bench \
  -m /home/dino/llm-models/pascal-tests/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  -ngl 99 -ncmoe 99 -ts 10/7 -fa on \
  -p 512 -n 128 -r 3
```

Con cache112 il processo aveva circa 5,2 GiB RSS nel campione e WSL usava
circa 14 GiB di page cache e 1,1 GiB swap. L'host registration non e' stata
provata: pinning degli expert da circa 18--20 GiB non e' compatibile con gli
attuali 15 GiB assegnati a WSL.

Il prefetch expert, provato a clock corretti come unica variabile con cache112,
ha dato 18,64 pp512 e 1,10 tg128 (`r=1`): -27,2% in generation rispetto alla
media cache-only. E' stato scartato e non confermato `r=3`.

### MoE: replica dopo aumento RAM WSL a 24 GB

Il 2 settembre 2026 il limite di memoria WSL2 e' stato portato da 15 a 24 GB
(23 GiB visibili, swap invariato a 4 GiB). Modello, build e parametri sono
rimasti invariati: GGUF standard SHA-256 `439fcb8266f37a035d2192d0fa773e59b177f379bd2bf90976419cea8c7dbb58`,
fork `d927e7dc1`, `-ngl 99 -ncmoe 99 -ts 10/7 -fa on`, batch 2048,
ubatch 512, 4 thread, KV f16, `pp512`, `tg128`, tre ripetizioni.

| Configurazione | pp512 | tg128 | Variazione rispetto a cache off |
| --- | ---: | ---: | ---: |
| Nessuna cache | **126,84 +/- 1,40 t/s** | **9,08 +/- 0,20 t/s** | riferimento |
| Cache statica distribuita, 112 slot/layer | **155,87 +/- 4,06 t/s** | **11,28 +/- 1,92 t/s** | +22,9% pp; +24,2% tg |
| Ibrido `-ncmoe 30`, cache distribuita 112 | **192,45 +/- 28,29 t/s** | **18,71 +/- 1,03 t/s** | +51,7% pp; +106,1% tg |

Rispetto alle misure con 15 GiB WSL, la configurazione senza cache sale da
16,48 a 126,84 t/s in prefill (7,7x) e da 0,66 a 9,08 t/s in generation
(13,8x). La cache112 passa da 1,54 a 11,28 t/s in generation. Questo dimostra
che i vecchi valori assoluti erano dominati dal page-cache thrashing; non erano
una misura valida del potenziale del fork o delle GPU. Il guadagno relativo
della cache, ora circa 24%, e' dello stesso ordine del +21% cache-only riportato
dall'autore su RTX 3060. Le prestazioni finali restano da replicare su Linux
nativo, che e' l'ambiente target.

L'ibrido cambia una sola variabile rispetto alla cache112 (`-ncmoe 99` a 30):
gli expert degli ultimi 10 layer diventano interamente residenti in GPU e la
cache resta sui 30 layer CPU-residenti. Durante il benchmark sono stati
campionati 7.164 MiB su CUDA0 e 7.215 MiB su CUDA1, con 3.943/853 MiB liberi:
lo split `10/7` e' quindi sbilanciato per l'ibrido, perche' concentra gli expert
completi degli ultimi layer sulla GTX 1070. Il pp512 ha variabilita' elevata e
va replicato; il tg128 e' sufficientemente stabile per promuovere lo split GPU
a prossima variabile di test.

Lo split `13/4`, provato come unica variabile sullo stesso ibrido, ha dato
204,18 +/- 10,60 pp512 e 16,27 +/- 3,20 tg128. Il prefill migliora, ma il decode
peggiora del 13,0% ed e' molto piu' variabile: configurazione scartata.

Con split `10/7`, `-ncmoe 29` (11 layer expert completi) ha dato
207,41 +/- 3,83 pp512 e 16,64 +/- 1,81 tg128: peggiore di `-ncmoe 30` in
decode. `-ncmoe 28` (12 layer completi) carica i pesi ma non riesce a creare il
contesto per esaurimento VRAM. Il massimo numero di expert residenti non e'
quindi il massimo prestazionale: l'undicesimo layer cade sulla GTX 1070 e il
suo calcolo GPU sostituisce un percorso CPU/GPU che il fork riesce a
sovrapporre.

### MoE: controllo della diversa quantizzazione Unsloth UD/MTP

Il GGUF locale `Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf` e' il file ufficiale
Unsloth MTP (SHA-256 gia' registrato sopra), 21,10 GiB e 35,51 B parametri.
Mantiene la stessa architettura `qwen35moe`; aggiunge la testa NextN e usa una
politica di quantizzazione dinamica diversa dal GGUF standard da 19,70 GiB.

Con speculazione disattivata, split `10/7` e gli stessi parametri del benchmark:

| Configurazione UD/MTP | pp512 | tg128 | Stato |
| --- | ---: | ---: | --- |
| `-ncmoe 99`, cache off | 70,92 +/- 49,17 t/s | **12,18 +/- 0,95 t/s** | pp non affidabile per pressione RAM; tg completo |
| `-ncmoe 32`, cache88 distribuita | 159,04 +/- 30,52 t/s | **15,42 +/- 1,23 t/s** | entra, ma piu' lento dell'ibrido standard |
| `-ncmoe 30`, cache112 | n/a | n/a | fallisce creando il contesto |
| `-ncmoe 30`, cache88 | n/a | n/a | OOM nel pool CUDA al primo calcolo |

La quantizzazione UD migliora il decode senza cache rispetto allo standard
(12,18 contro 9,08 t/s), ma il file piu' grande consuma abbastanza VRAM da
impedire il migliore ibrido. Non spiega il vecchio divario di ordini di
grandezza e non e' il candidato Pascal piu' veloce tra quelli provati.

### MoE: server pratico e speculative decoding (misure storiche pre-aumento RAM)

Il server pratico usa cache88 per lasciare margine a context e compute buffer:

```bash
cd /home/dino/pascal-frankenstein-llm
./llama.cpp/build-pascal-cuda/bin/llama-server \
  -m /home/dino/llm-models/pascal-tests/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  -c 16384 --parallel 1 \
  -ngl 99 -ncmoe 99 -ts 10,7 -fa on \
  --moe-cache-profile /home/dino/pascal-frankenstein-llm/moe-traces/qwen36-35b-merged.csv \
  --moe-cache-slots 88 \
  --host 127.0.0.1 --port 18081 --jinja --no-webui
```

Test reale sul modello standard, 74 token prompt + 188 output:

- prompt **3,50 t/s**;
- generation **0,605 t/s**;
- VRAM post-richiesta 8.359/1.083 MiB, liberi 2.748/6.985 MiB;
- risposta binary-search completa, corretta e terminata normalmente.

Il GGUF MTP e' stato confrontato a una variabile con cache88, profilo MTP
specifico e lo stesso prompt:

| MTP draft | Prompt | Generation | Acceptance | Qualita' |
| --- | ---: | ---: | ---: | --- |
| off | 1,695 t/s | 0,560 t/s | - | output corretto |
| `draft-n-max=2` | 1,471 t/s | 0,424 t/s | 125/126 = 99,2% | output identico e corretto |

In questa vecchia configurazione, con RAM WSL insufficiente e prima della cache
per-device, speculative decoding peggiorava prompt e generation. La conclusione
non va generalizzata: le misure aggiornate a 23 GiB RAM, `-ncmoe 32` e quote
cache `160/88` sono registrate piu' avanti e superano nettamente 20 t/s.

### Audit separato: pipeline MoE del fork, prove svolte e lacune

Il README del fork definisce una pipeline in due fasi: **(1) catturare un
profilo di routing** con due workload diversi e unirlo; **(2) avviare il server
con cache, poi verificare la combinazione cache/prefetch/speculative decoding**.
Le prove svolte e i loro output sono i seguenti.

| Fase | Prova eseguita | Output verificato | Stato |
| --- | --- | --- | --- |
| 1. Capture profile standard | `llama-moe-trace` su prompt coding e chat, CSV poi concatenati | `qwen36-35b-code.csv` (8.800 righe), `...-chat.csv` (6.280), `...-merged.csv` (15.080); SHA-256 del merged registrato sopra | eseguita |
| 1. Capture profile MTP | Stessa procedura sul GGUF MTP | CSV merged da 13.280 righe; SHA-256 registrato sopra | eseguita |
| 2. Cache | Benchmark `r=3`, cache off/on, unica variabile 112 slot/layer | 0,66 ± 0,02 a 1,51 ± 0,20 t/s generation; +128,8%; output token-identico per progetto e risposta server corretta | eseguita |
| 2. Server cache | Server 16k, cache88, un solo slot | 3,50 t/s prompt, 0,605 t/s generation, 188 token corretti | eseguita |
| Prefetch | `GGML_SCHED_PREFETCH_EXPERTS=1` con cache112 | 1,10 t/s generation (`r=1`), peggiore della media cache-only 1,51 t/s | eseguita, scartata provvisoriamente |
| MTP | prima prova cache88 tutto-CPU; poi prova aggiornata ibrida e cache per-device | storica: 0,424 contro 0,560 t/s; aggiornata: 29,03 +/- 1,10 t/s a 16k | verificata; raccomandata solo nella configurazione aggiornata |
| CPU/GPU async | Lasciata l'impostazione predefinita `--sched-async-cpu` attiva | il fork esegue gli split CPU indipendenti in un worker in sovrapposizione alla catena GPU | attiva, ma non isolata con A/B `--no-sched-async-cpu` |
| Host registration | diagnostica con 23 GiB WSL | `cudaHostRegister` su 14.763,15 MiB: `operation not supported` | non disponibile in WSL; da provare su Linux nativo |

La pipeline cache + MTP e' ora validata in `llama-cli` a 4k e 16k. La parte
host registration + prefetch resta invece non validabile in WSL per una
limitazione esplicita del passthrough CUDA, non per insufficienza della RAM.

### Memoria: dense, MoE e cache

Le tre misure non vanno confuse:

| Voce | Dense Qwen3.8-27B | MoE Qwen3.6-35B-A3B | Differenza / conseguenza |
| --- | ---: | ---: | --- |
| File GGUF | 15,65 GiB | 19,70 GiB | MoE +4,05 GiB (+25,9%) |
| VRAM server 16k misurata | 10.138 + 7.637 MiB = 17,36 GiB | 8.359 + 1.083 MiB = 9,22 GiB con cache88 | il dense sta interamente nelle due GPU; gli expert MoE freddi restano in RAM |
| VRAM benchmark senza cache | n/a | 1.513 + 1.005 MiB = 2,46 GiB | non-expert GPU, expert CPU |
| VRAM benchmark cache112 | n/a | 9.829 + 1.005 MiB = 10,58 GiB | la cache aggiunge **8.316 MiB = 8,12 GiB** soprattutto su GPU0 |

Per interpolazione lineare, cache88 richiede circa 6,4 GiB della sola VRAM per
gli expert caldi; la misura server include anche KV e compute buffer, percio'
non e' direttamente sottraibile dalla riga di benchmark. Il fork raccomanda
inoltre circa 900 MiB liberi oltre il pack: cache88 e 16k sono stati scelti per
questo margine, non per massimizzare artificialmente il numero di slot.

La WSL attuale dispone di 23 GiB visibili e 4 GiB swap dopo l'aumento a 24 GB.
Questo ha eliminato il thrashing che falsava le prime misure MoE. Non ha pero'
reso possibile host registration: il driver WSL rifiuta esplicitamente il
pinning CUDA del buffer expert da 14.763,15 MiB con `operation not supported`.

Configurazione WSL applicata in `%UserProfile%\\.wslconfig`:

```ini
[wsl2]
memory=24GB
```

Linux nativo evita l'overhead e la contesa di memoria con Windows, ma non
elimina il limite fondamentale DDR3/PCIe/Pascal. E' pero' necessario per
verificare host registration; WSL non puo' rappresentare quel percorso.

## Esperimento MoE: cache distribuita per device (2 settembre 2026)

Modifica sorgente mirata nel fork `perf` commit `d927e7dc1`: in
`src/llama-model.cpp`, `init_moe_expert_cache()` alloca ora un pack hot per
ogni device presente in `dev_layer[il]`, anziche' copiare tutti i layer nel
primo device GPU. Build Pascal esistente ricompilata correttamente; nessuna
modifica a CMake, CUDA o ai flag di compilazione.

| Modello | Configurazione | Risultato |
| --- | --- | --- |
| Qwen3.6-35B-A3B Q4_K_M | `-ngl 99 -ncmoe 99 -ts 10/7 -fa on`, profilo merged, cache112, `pp512/tg128`, `r=3` | **21,82 +/- 5,77 t/s** prefill; **1,54 +/- 0,24 t/s** generation |

Controlli completati:

- log del loader: layer MoE 0--24 su CUDA0, con 5.042,62 MiB di pack hot;
  layer 25--39 su CUDA1, con 3.094,88 MiB. Dopo il load: CUDA0
  10.456/11.264 MiB, CUDA1 6.393/8.192 MiB;
- prompt deterministico a temperatura 0, cache on e off: entrambe le esecuzioni
  hanno prodotto `ponte`;
- screening r=1: `pp128` 5,66 t/s, `tg32` 3,10 t/s. Non e' confrontabile come
  baseline con `tg128`, quindi non e' usato per stimare il guadagno.

Confrontata con la precedente cache112 mono-GPU (`19,41 +/- 6,10` prefill,
`1,51 +/- 0,20` generation), la Fase 1 corregge la residenza ma non dimostra
un miglioramento significativo di generation. Nessuna configurazione di cache
distribuita viene quindi ancora raccomandata come piu' veloce: servono server
16k e un confronto lungo di correttezza prima di una conclusione operativa.

### Quote cache per GPU: screening controllato (2 settembre 2026)

La cache distribuita e' stata estesa con una quota di slot distinta per GPU.
Sintassi CLI: `--moe-cache-slots N0,N1`; per `llama-bench`, che usa la virgola
per separare configurazioni, sintassi env `GGML_MOE_CACHE_SLOTS=N0/N1`.
I primi benchmark di screening non sono registrati: un percorso profilo errato
ha prodotto il warning `cannot open profile` e ha disabilitato la cache. La
causa e' stata identificata prima di usare quei numeri come baseline; le prove
sono state ripetute sotto con `moe-traces/qwen36-35b-merged.csv`.

Ripetizione corretta con modello standard, profilo
`moe-traces/qwen36-35b-merged.csv`,
`-ngl 99 -ncmoe 30 -ts 10/7 -fa on -p 512 -n 128 -r 3`:

| Slot CUDA0/CUDA1 | pp512 | tg128 | Esito |
| ---: | ---: | ---: | --- |
| 112/112 (controllo) | 196,56 +/- 4,03 t/s | 15,57 +/- 1,79 t/s | controllo valido |
| 160/112 | 225,64 +/- 12,91 t/s | 16,50 +/- 1,78 t/s | +14,8% prefill; +6,0% generation |
| 192/112 | **235,15 +/- 4,30 t/s** | **17,10 +/- 1,67 t/s** | +19,6% prefill; +9,8% generation |

Con `192/112` la quota asimmetrica produce quindi un vantaggio reale, ma non
raggiunge 20 t/s in generation. `208/112` entra in memoria, ma durante `pp512`
lascia soltanto 155 MiB liberi su CUDA0 e non completa dopo oltre due minuti;
test interrotto e configurazione esclusa per cliff di memoria. Lo smoke verbose
di `160/112` conferma 25 layer x 160 slot su CUDA0 (7.143,00 MiB) e 5 layer x
112 slot su CUDA1 (1.002,75 MiB).

### Host registration e prefetch dopo aumento RAM WSL (2 settembre 2026)

Controllo diagnostico con 23 GiB RAM visibili a WSL e modello standard:

```text
ggml_backend_cuda_register_host_buffer: failed to register 14763.15 MiB of pinned memory: operation not supported
```

Il pinning CUDA dell'intero buffer expert e' quindi **non supportato dal driver
WSL corrente**. L'aumento RAM elimina il precedente thrashing, ma non rende
disponibile il percorso `cudaHostRegister` del fork. Questo e' un limite WSL
verificato; la prova va ripetuta su Linux nativo, dove il target operativo e'
comunque previsto.

I benchmark lanciati insieme alla cache non vengono registrati perche' usavano
il percorso profilo errato descritto sopra. Il fallimento diagnostico del
pinning resta valido e indipendente dalla cache; il prefetch completo andra'
rimisurato insieme al pinning su Linux nativo.

Il comando prefill del README del fork (`pp2048`, `b=ub=2048`, `-ncmoe 26`)
non crea il contesto sullo split dual-GPU per insufficiente VRAM runtime della
GTX 1070. Lo smoke test sulla sola GTX 1080 Ti entra (circa 10,8 GiB VRAM,
GPU al 100%, 1936 MHz, 62 C), ma non completa una singola ripetizione dopo
oltre sette minuti ed e' stato interrotto. Non e' una baseline e non e'
confrontabile con i 1143/1880 t/s misurati dal creator su RTX 3060 Ampere.

### MTP Unsloth + cache asimmetrica: obiettivo 20 t/s superato (2 settembre 2026)

Inferenza reale `llama-cli`, non `llama-bench`, per esercitare il decoder MTP:

```text
modello: Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf
profilo: moe-traces/qwen36-35b-mtp-merged.csv
-ngl 99 -ncmoe 32 -ts 10,7 -fa on -b 512 -ub 512
cache: 88/88 oppure 160/88
MTP: --spec-type draft-mtp --spec-draft-n-max 2
sampling: --reasoning off --temp 0 --seed 123
output: 128 token, stesso prompt italiano lungo
```

La prima esecuzione senza MTP a filesystem freddo (14,1 t/s generation) e'
esclusa dalle medie. Le tre repliche calde sono:

| Contesto | Cache | MTP | Prompt t/s | Generation t/s |
| ---: | ---: | --- | ---: | ---: |
| 4k | 88/88 | off | 28,5 / 20,3 / 30,8 | 15,8 / 15,7 / 16,9 |
| 4k | 88/88 | n-max 2 | 28,7 / 30,0 / 21,9 | 23,4 / 23,3 / 22,5 |
| 4k | 160/88 | n-max 2 | 43,2 / 39,6 / 39,1 | 29,7 / 28,1 / 28,4 |
| 16k | 160/88 | n-max 2 | 40,5 / 38,5 / 39,7 | 27,8 / 29,9 / 29,4 |

Sintesi delle medie:

- baseline MTP model, spec off, cache88: **16,13 +/- 0,67 t/s** generation;
- MTP n-max2, cache88: **23,07 +/- 0,49 t/s** (+43,0%);
- MTP n-max2, cache160/88, 4k: **28,73 +/- 0,85 t/s** (+78,1% sulla baseline,
  +24,6% sulla cache uniforme);
- MTP n-max2, cache160/88, 16k: **39,57 +/- 1,01 t/s prompt** e
  **29,03 +/- 1,10 t/s generation**.

Il campione VRAM a 16k mostra CUDA0 8.831 MiB usati / 2.276 MiB liberi e
CUDA1 6.505 MiB usati / 1.563 MiB liberi. Le tre risposte 16k sono uguali fra
loro e semanticamente corrette; terminano al limite esplicito di 128 token.
Questo risultato spiega anche Unsloth: la quantizzazione dinamica da sola
portava il decode a circa 16 t/s; il superamento dei 20 arriva dalla
composizione MTP + cache, e la quota asimmetrica sfrutta la VRAM aggiuntiva
della 1080 Ti senza saturare la 1070.

Verifica `llama-server` separata con gli stessi parametri, context 16k e un solo
slot: richiesta `/completion` da 24 token prompt + 128 output, **35,05 t/s
prompt** e **25,70 t/s generation**; acceptance MTP 72/108 = 66,7%. Il server
e' stato arrestato normalmente dopo il test. Questa e' una singola richiesta
di convalida, mentre la media `llama-cli` sopra deriva da tre repliche.

### Disposizione expert e contesto 64k (2 settembre 2026)

Semantica verificata dei parametri, importante per leggere i risultati:
`-ngl 99` mantiene in GPU i pesi non-expert di tutti i 40 layer;
`-ncmoe N` lascia in CPU/RAM gli expert originali dei primi N layer, sui quali
la cache crea una copia GPU del subset hot. Gli ultimi `40-N` layer conservano
invece tutti i loro expert in GPU. Percio' `-ncmoe 32` non significa che gli
ultimi otto layer vadano in RAM: significa l'opposto, cioe' otto layer expert
completi in VRAM e 32 layer con expert cold in RAM + cache hot in VRAM.

Screening a una replica, contesto allocato 65.536, KV F16, stesso prompt da 128
token, MTP n-max2 e tutti gli altri parametri invariati. Le quote sulla GTX 1070
sono state aumentate al crescere di `-ncmoe` per reinvestire lo spazio liberato
dai layer expert completi e mantenere l'occupazione VRAM approssimativamente
comparabile:

| `-ncmoe` | Layer con cache / completi | Cache CUDA0/CUDA1 | Generation | Esito |
| ---: | ---: | ---: | ---: | --- |
| 31 | 31 / 9 | 160/64 | 28,3 t/s | 80 e 88 slot su CUDA1: OOM |
| 32 | 32 / 8 | 160/88 | **30,6 t/s** | miglior screening |
| 33 | 33 / 7 | 160/108 | 29,4 t/s | finalista |
| 36 | 36 / 4 | 160/144 | 29,0 t/s | valido |
| 40 (`99`) | 40 / 0 | 160/160 | 25,1 t/s | cache parziale su tutti i layer |
| 40 (`99`) | 40 / 0 | 160/168 | 24,2 t/s | valido, nessun guadagno |
| 40 (`99`) | 40 / 0 | 176/176 | n/a | OOM al primo decode |
| 40 (`99`) | 40 / 0 | 192/192 | n/a | OOM allocazione compute buffer |

La configurazione con un subset hot su tutti i 40 layer e' quindi stata provata
esplicitamente e supera 20 t/s, ma e' piu' lenta del compromesso che lascia gli
ultimi layer completi in GPU. Sul CSV di training, la copertura nominale degli
expert selezionati e' 98,45% per 40 layer/cache160-160 contro 97,60% per
33 layer/cache160-108; le righe layer-token con tutti gli otto routed expert hot
sono rispettivamente 90,96% e 86,40%. La maggiore copertura nominale non si
traduce dunque in maggiore throughput.

Il sorgente conferma una differenza strutturale: per ogni layer servito dalla
cache, `build_moe_ffn()` costruisce sempre due catene FFN complete, cold e hot,
piu' l'add finale; gli ID `-1` azzerano le righe non applicabili. Un layer expert
interamente GPU usa invece il percorso MoE ordinario. E' quindi verificato che
40 layer cache hanno piu' struttura ibrida di 33 layer cache + 7 completi.
Che questo overhead sia da solo la causa della differenza di throughput resta
un'inferenza, non ancora una misura diretta. Inoltre la copertura sopra e' sullo
stesso CSV usato per scegliere gli expert, non su un workload held-out. Servono
contatori runtime per layer e il nuovo profilo multi-dominio prima di dichiarare
dimostrato il meccanismo causale.

Repliche dei due finalisti, sempre con capacita' 65.536 ma prompt breve:

| Configurazione | Prompt t/s | Generation t/s | Media generation |
| --- | --- | --- | ---: |
| `-ncmoe 32`, cache160/88 | 45,4 / 46,8 / 46,7 | 26,8 / 29,5 / 29,8 | **28,70 +/- 1,65** |
| `-ncmoe 33`, cache160/108 | 48,5 / 41,3 / 48,8 | 30,7 / 29,9 / 29,8 | **30,13 +/- 0,49** |

Il test di contesto realmente popolato ha usato i primi 245.000 byte di
`tests/test-chat.cpp`, pari a 60.132 token misurati con `llama-tokenize`, in un
contesto 65.536. Configurazione: `-ncmoe 33`, cache160/108, split10,7, FA on,
batch/ubatch512, MTP n-max2, output128, temperatura0 e seed123:

| KV | Prompt popolato | Prompt t/s | Generation t/s | VRAM durante prefill |
| --- | ---: | ---: | ---: | --- |
| F16/F16 | 60.132 token | 132,8 | **25,5** | 10.164 MiB + 7.571 MiB |
| Q8_0/Q8_0 | 60.132 token | 133,4 | **23,0** | 9.799 MiB + 7.411 MiB |

Sono singole prove lunghe, non medie. Entrambe hanno completato senza OOM e con
output coerente col sorgente. Q8 risparmia circa 525 MiB complessivi, ma sulle
due Pascal perde il 9,8% in generation rispetto a F16 a circa 60k token. Il
risultato F16 dimostra che 64k e' praticabile e resta sopra l'obiettivo di 20
t/s anche con KV quasi piena. Non dimostra ancora la qualita' a lungo contesto,
ne' sostituisce le suite RULER/Aider/HumanEval+/BFCL pianificate.

### Profilo routing multi-dominio v2 (3 settembre 2026)

Fase 1 completata senza modificare il fork e senza sovrascrivere il profilo v1.
I dati raw, prompt e script del v2 sono conservati in un archivio locale e non
sono versionati nel repository pubblico minimale. Hash, configurazione e
conclusioni misurate restano registrati in questa sezione.

Acquisizione con GGUF MTP verificato, `llama-moe-trace`, `-ngl 99 -ncmoe 99
-ts 10,7 -fa on -c 4096 -b 512 -ub 512 -n 512`, greedy, senza cache e senza
speculative MTP. Sei domini training e sei held-out: coding, tool agent, RAG,
reasoning, filosofia/role-play e multilingue. Ognuna delle dodici trace contiene
esattamente 512 posizioni decode x 40 layer x 8 expert, senza EOG precoce,
duplicati, buchi o ID invalidi. Il profilo v1 aveva 256 token decode per layer;
v2 ne usa 3.072 training piu' 3.072 held-out.

- train merged SHA-256:
  `06a9c01db379d3fdc43307e89b06fab0e4348ce8a748fc1fa4d4223e90cc2f78`;
- held-out merged SHA-256:
  `b8ba0c2a2c413c0691f4e8914bb38a15600c2ed97ee1426402e3e106a9bcb450`.

Con il placement corrente `-ncmoe 33`, cache160/108, il v2 scelto solo sul
training copre sul held-out il 91,87% degli expert routed e il 63,62% delle
righe layer-token interamente hot. Il v1 ottiene 82,43% e 39,72%. Il guadagno
v2 e' presente in tutti i domini (+4,07 fino a +11,90 punti percentuali), mentre
il Jaccard medio fra set hot v1/v2 e' soltanto 0,611.

La simulazione a memoria costante sulla GTX 1070 trova come sette layer completi
`29,32,33,35,36,37,39`, ma il vantaggio held-out rispetto al suffix attuale
33--39 e' solo +0,06 punti di hit e +0,25 punti di righe all-hot: insufficiente
per giustificare una modifica. Su CUDA0 emerge invece un candidato futuro:
layer2 completo e cache156 sugli altri 24 layer usa gli stessi 4.000 slab di
25x160 e porta la copertura complessiva simulata da 91,87%/63,62% a
92,02%/64,23%. Non e' ancora un benchmark.

A/B contemporaneo a capacita'64k, prompt128, KV F16, cache160/108:

| Profilo | MTP | Generation t/s | Media |
| --- | --- | --- | ---: |
| v1 | n-max2 | 33,4 / 34,1 / 30,8 | **32,77 +/- 1,74** |
| v2 | n-max2 | 29,1 / 25,3 / 27,8 | **27,40 +/- 1,93** |
| v1 | off | 19,5 / 17,3 / 17,1 | **17,97 +/- 1,33** |
| v2 | off | 19,2 / 20,7 / 20,7 | **20,20 +/- 0,87** |

Il v2 e' quindi +12,4% senza MTP, coerentemente con la maggiore hit-rate, ma
-16,4% con MTP. Due richieste server diagnostiche hanno misurato acceptance
70,48% (74/105) e 29,66 t/s per v1, contro 63,39% (71/112) e 28,56 t/s per v2.
Anche l'output a temperatura zero cambia: spostare expert fra kernel CPU e CUDA
introduce differenze numeriche sufficienti a cambiare alcuni token target e
l'acceptance speculativa.

Conclusione verificata della fase: v2 e' il profilo statico general-purpose
migliore quando MTP e' spento; v1 resta il profilo operativo piu' veloce per il
benchmark MTP corrente. La hit-rate non puo' essere ottimizzata isolatamente:
servono sempre throughput, acceptance e correttezza/qualita'.

### Smoke test modello alternativo: Heretic Q4_K_M (3 settembre 2026)

Scaricato in `/home/dino/llm-models/pascal-tests/` il GGUF
`llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-GGUF`, file
`Qwen3.6-35B-A3B-uncensored-heretic-Q4_K_M.gguf` (20 GiB). Un tentativo con
MTP `n-max 2` si e' fermato correttamente in caricamento: il GGUF non espone
una MTP context compatibile (`failed to create MTP context`).

Retry unico senza speculative decoding, con le altre variabili invariate:
`-ngl 99 -ncmoe 33 -ts 10,7 -fa on -c 65536 -ctk f16 -ctv f16 -b 512 -ub 512`,
profilo cache v1 `qwen36-35b-mtp-merged.csv`, slot `160,108`, `-n 128`,
`--reasoning off --temp 0 --seed 123`. Misura completa singola:
**44,2 t/s prompt** e **19,1 t/s generation**.

E' solo una verifica di funzionamento e prestazioni preliminare: non ha MTP,
il profilo hot/cold e' stato addestrato sull'altro GGUF, e non e' quindi un
confronto diretto con i 30,13 t/s MTP del modello Unsloth ufficiale ne' un nuovo
profilo da adottare.

### Passaggio a Linux Mint nativo: candidato operativo (6 settembre 2026)

L'ambiente di prova e' ora Linux Mint nativo. L'utente riferisce di avere
installato localmente i binari della release
`v0.1.0-pascal-cuda12-sm61`; il comando funzionante corrente usa il binario:

```text
/home/dino/proj/pascal-frankenstein-llm-v0.1.0-linux-x86_64-cuda12-sm61/bin/llama-server
```

L'ultimo modello scaricato e funzionante e':

```text
/home/dino/proj/genAI/models/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-Q4_K_M.gguf
```

SHA-256 verificato:

```text
fc89d92377b27fe0f80eb683a5105d0921234c24f7c1d70ecc2356ddf994d781
```

Il profilo routing resta quello v1 gia' versionato; il suo percorso assoluto nel
comando seguente presuppone il checkout `/home/dino/pascal-frankenstein-llm`.

Comando funzionante riferito dall'utente:

```bash
~/proj/pascal-frankenstein-llm-v0.1.0-linux-x86_64-cuda12-sm61/bin/llama-server \
  -m /home/dino/proj/genAI/models/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-Q4_K_M.gguf \
  -c 65536 -ngl 99 -ncmoe 33 -ts 10,7 -fa on \
  -ctk f16 -ctv f16 -b 512 -ub 512 \
  --moe-cache-profile /home/dino/pascal-frankenstein-llm/moe-traces/qwen36-35b-mtp-merged.csv \
  --moe-cache-slots 160,108 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --reasoning on --temp 0.7 --seed 123
```

Questa e' una configurazione operativa iniziale, non ancora una nuova baseline:
non sono stati ancora registrati throughput, acceptance MTP, VRAM/RAM o
correttezza dell'output. Il modello corrisponde all'hash atteso per la variante
`Native-MTP-Preserved`; la creazione del contesto MTP e' da verificare
nuovamente su Linux nativo.

Stato hardware al 6 settembre 2026, prima del test:

- Linux nativo, driver NVIDIA `580.173.02`, CUDA esposta dal driver `13.0`;
- GTX 1080 Ti: P2, 2.007 MiB occupati su 11.264 MiB;
- GTX 1070: P8, 9 MiB occupati su 8.192 MiB;
- sulla GTX 1080 Ti restano processi grafici desktop/Xorg, Cinnamon e Firefox;
- il binario della release resta quello verificato:
  `/home/dino/proj/pascal-frankenstein-llm-v0.1.0-linux-x86_64-cuda12-sm61/bin/llama-server`.

### Linux Mint nativo: prova reale a 128k (6 settembre 2026)

La release richiede di esportare la directory `bin` nel
`LD_LIBRARY_PATH`; senza questo export `llama-server` termina con
`libllama-server-impl.so: cannot open shared object file`. La release inoltre
non accetta il JSON Reddit `{"preserve_thinking": true}` nella forma provata:
il test usa l'opzione equivalente disponibile nel binario,
`--reasoning-preserve`.

Il primo avvio a 128k ha caricato il modello ma ha disabilitato la cache per un
percorso profilo inesistente e ha aperto quattro slot. E' stato arrestato senza
benchmark. Il test valido ha usato il profilo presente nel worktree corrente,
un solo slot e la configurazione seguente:

```bash
export LD_LIBRARY_PATH=/home/dino/proj/pascal-frankenstein-llm-v0.1.0-linux-x86_64-cuda12-sm61/bin

/home/dino/proj/pascal-frankenstein-llm-v0.1.0-linux-x86_64-cuda12-sm61/bin/llama-server \
  --model /home/dino/proj/genAI/models/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-Q4_K_M.gguf \
  --port 8001 --alias qwen36-35b-a3b --parallel 1 \
  -c 131072 -n 32768 --no-context-shift \
  -ngl 99 -ncmoe 33 -ts 10,7 -fa on \
  -ctk q8_0 -ctv q8_0 -b 512 -ub 512 \
  --moe-cache-profile /home/dino/proj/pascal-frankenstein-llm.worktrees/progetto-situazione-attuale/moe-traces/qwen36-35b-mtp-merged.csv \
  --moe-cache-slots 160,108 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --reasoning on --reasoning-preserve \
  --temp 0.6 --top-p 0.95 --top-k 20 \
  --repeat-penalty 1.0 --presence-penalty 0.0 \
  --host 127.0.0.1 --jinja --no-webui
```

Il server ha allocato `n_ctx_slot=131072`, un solo slot, e ha risposto
`/health` con `status=ok`. Il prompt deterministico di prova conteneva
120.000 token misurati dall'endpoint `/tokenize`; con il template la richiesta
ha elaborato 120.021 token. Risultati HTTP:

| Misura | Risultato |
| --- | ---: |
| Codice HTTP | 200 |
| Prompt processing | 780,52 s; **153,77 t/s** |
| Generation | 128 token; **23,62 t/s** |
| Tempo totale | 785,94 s |
| MTP acceptance | **75/103 = 72,8%**; lunghezza media 2,44 |
| Context shift / truncation | disabilitato / `truncated=0` |
| Fine risposta | `length`, durante il reasoning |

Il reasoning restituito è coerente con l'obiettivo del test; il campo `content`
è rimasto vuoto perché il limite di 128 token è stato consumato dal reasoning.
Questa prova dimostra la capacità operativa del contesto quasi pieno, non la
qualità finale di una risposta agent completa.

Durante l'esecuzione la VRAM è rimasta quasi piena ma stabile: circa
10.968--11.045 MiB sulla GTX 1080 Ti e 7.919 MiB sulla GTX 1070. Il picco
osservato è stato circa 74--75 °C sulla 1080 Ti e 55--60 °C sulla 1070; la
1080 Ti ha raggiunto circa 99% di utilizzo GPU e 227 W. Il processo ha usato
circa 16,8 GiB RSS; il sistema ha mantenuto circa 1,4 GiB di swap occupata.

I log completi sono conservati localmente in
`/home/dino/pascal-test-logs/`, incluso il log server
`20260906-163525-128k-server.log`, la richiesta/risposta HTTP e il campione
GPU del test. Il test è riuscito come prova di capacità 128k; la prossima
prova deve usare un `max_tokens` maggiore per separare reasoning e risposta
finale e misurare correttamente il comportamento dell'agente.

### Confronto rapido Heretic 64k: F16 contro Q8 K/V (6 settembre 2026)

Per separare l'effetto del contesto lungo da quello del modello e della KV,
sono state eseguite tre richieste identiche sul modello Heretic MTP-preserved,
con MTP sempre attivo, cache `160,108`, `-ncmoe 33`, split `10,7`, un solo
slot, reasoning disattivato e 128 token forzati (`ignore_eos=true`). Il prompt
di 1.841 token era identico in tutte le prove.

| KV | Generation | Acceptance MTP | Stato |
| --- | ---: | ---: | --- |
| F16/F16 | **45,91 / 46,56 / 45,91 t/s** | 78/97, poi 79/95, 79/95 | tre repliche valide |
| Q8_0/Q8_0 | **45,34 / 45,49 / 44,90 t/s** | 79/94 in tutte | tre repliche valide |

Medie: F16 **46,13 t/s**, Q8 **45,24 t/s**. La differenza di circa 1,9%
è piccola sul prompt breve; la KV Q8 non spiega il calo da 30 a 23,6 t/s
osservato sul contesto realmente popolato a 120k.

La prova ha prodotto risposte corrette nelle richieste brevi (`17 times 19 is
323.`). Il valore di circa 46 t/s non è direttamente confrontabile con il
precedente `30,13 t/s` se il workload, il client e la lunghezza del prompt
differiscono; dimostra però che il checkpoint Heretic conserva un percorso MTP
veloce su contesto corto.

La documentazione pubblica di Unsloth descrive il checkpoint
`Qwen3.6-35B-A3B-MTP-GGUF` come modello con MTP nativo addestrato multi-step,
35B totali/3B attivi e contesto nativo 262.144. Le quantizzazioni `UD` sono
artefatti Unsloth separati (dynamic quantization), mentre il modello locale
Heretic `Native-MTP-Preserved-Q4_K_M` è una conversione Heretic che preserva i
20 blocchi MTP, non una quantizzazione UD Unsloth. Non è stata trovata una
release pubblica che combini esplicitamente Heretic + Native MTP Preserved +
UD-Q4_K_XL; non va quindi presunto che esista o che sia intercambiabile con il
file locale.

### Linux Mint: confronto preliminare mobile/Tailscale contro Firefox locale
(6 settembre 2026)

Con lo stesso server Heretic a 64k, cache `160,108`, MTP `n-max=2`, KV Q8,
split `10,7`, un solo slot e `--host 0.0.0.0`, sono state osservate due
richieste provenienti da client diversi:

| Client | Prompt | Generation | Acceptance MTP | Note |
| --- | ---: | ---: | ---: | --- |
| Mobile via Tailscale | 338 token | **48,61 t/s** | 348/476 = **73,1%** | 585 token generati |
| Firefox sul server | 18 token | **32,30 t/s** | 223/336 = **66,4%** | 390 token generati; LCP `sim_best=0,962` |

Il server ha riportato anche un valore transitorio di 49,54 t/s durante la
prima richiesta. Il risultato non è ancora un A/B scientifico: i prompt,
template, lunghezze di risposta e acceptance MTP differiscono; la seconda
richiesta inoltre ha riutilizzato un prefisso tramite LCP cache. Il dato
importante è però che il percorso remoto via Tailscale non mostra un overhead
apprezzabile e ha raggiunto circa 48,6 t/s sul workload più favorevole.

Il log contiene tre richieste `unauthorized: Invalid API Key`, probabilmente
generate da controlli del client/browser prima della richiesta valida. La API
key usata nel comando è stata esposta nella conversazione e va ruotata prima
di lasciare il server accessibile ad altri dispositivi. Per i prossimi test
usare una chiave nuova, lunga e non pubblicarla nei log condivisi.

## Raccomandazione conclusiva

1. **Chat/coding con massima prudenza qualitativa:** Qwen3.8-27B Q4_K_M dense,
   server 16k, full offload sulle due GPU, split automatico, KV f16 e FA on;
   ~10,37 t/s reali gia' verificati.
2. **Massime prestazioni MoE verificate a capacita' 64k:** GGUF ufficiale
   Unsloth Qwen3.6-35B-A3B-MTP-UD-Q4_K_M, `-ncmoe 33`, split `10,7`, cache
   `160,108`, KV F16 e MTP n-max2: **30,13 +/- 0,49 t/s generation** con prompt
   breve. Con 60.132 token realmente presenti nella KV ha ottenuto **25,5
   t/s** in una prova completa.
3. **Contesto:** 64k e' verificato stabile e praticabile, anche se la qualita'
   long-context deve ancora essere misurata. 128k non e' ancora validato e
   richiedera' probabilmente KV quantizzata e/o una diversa quota expert.
4. **Vincoli operativi:** mantenere NVIDIA su Preferisci prestazioni massime;
   lasciare P2P disabilitato; non passare a CUDA 13. In WSL lasciare spenti
   host registration e prefetch; riprovare la coppia soltanto su Linux nativo.

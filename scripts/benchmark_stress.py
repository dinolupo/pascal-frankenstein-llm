#!/usr/bin/env python3
"""
benchmark_stress.py - Qwen Pascal Frankenstein Multi-Turn Stress & Performance Benchmark

Simula una conversazione reale multi-turno su llama-server per:
1. Misurare la velocità reale di generazione (token/s) e latenza TTFT.
2. Monitorare l'occupazione VRAM reale su GPU 0 (1080 Ti) e GPU 1 (1070).
3. Testare la stabilità del contesto all'aumentare dei turni e individuare blocchi/deadlock.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import requests

def get_vram_stats():
    """Restituisce l'occupazione VRAM di ciascuna GPU NVIDIA."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ]
        out = subprocess.check_output(cmd, encoding="utf-8").strip()
        gpus = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                idx, name, used, total = parts
                gpus.append({
                    "id": idx,
                    "name": name,
                    "used_mb": int(used),
                    "total_mb": int(total),
                    "pct": (int(used) / int(total)) * 100
                })
        return gpus
    except Exception as e:
        return []

def format_vram(gpus):
    if not gpus:
        return "VRAM: [N/A]"
    items = [f"GPU{g['id']} ({g['name']}): {g['used_mb']}/{g['total_mb']}MB ({g['pct']:.1f}%)" for g in gpus]
    return " | ".join(items)

# Domande sequenziali per simulare un dialogo reale e far crescere progressivamente il contesto
TURNS_PROMPTS = [
    "Ciao! Spiegami brevemente quali sono le differenze architetturali chiave tra una GPU Pascal (GTX 1080 Ti) e una moderna architettura con Tensor Core.",
    "Molto chiaro. Ora approfondisci come funziona l'operazione di Matrix Multiplication a 8-bit (INT8) rispetto a FP16 su una scheda senza Tensor Core nativi.",
    "Tenendo a mente quanto hai appena detto, cosa succede se applichiamo il Multi-Token Prediction (MTP) a un modello Mixture of Experts su questo hardware?",
    "Come influisce la cache degli esperti MoE (hot/cold caching) sul traffico PCIe tra RAM di sistema e VRAM?",
    "Fammi un riassunto dei punti chiave toccati finora in questa conversazione, evidenziando le criticità di VRAM.",
    "Se dovessimo aggiungere una seconda scheda asimmetrica (come una GTX 1070 da 8GB), come andrebbe ripartito il carico dei tensori?",
    "Scrivi un breve frammento in pseudocodice o C per spiegare la logica di smistamento dei token tra i diversi esperti.",
    "Analizza il codice appena scritto e dimmi quali sono i potenziali colli di bottiglia per quanto riguarda la latenza di memoria.",
    "Supponiamo ora che la finestra di contesto raggiunga decine di migliaia di token: come deve gestire la KV-cache l'algoritmo Flash Attention per non esaurire la memoria?",
    "Ottimo lavoro fin qui. Concludi sintetizzando l'intera discussione in una tabella comparativa finale."
]

def main():
    parser = argparse.ArgumentParser(description="Multi-Turn Stress & Benchmark for Pascal Frankenstein LLM")
    parser.add_argument("--url", default="http://127.0.0.1:8001", help="Base URL di llama-server (default: http://127.0.0.1:8001)")
    parser.add_argument("--api-key", default=os.getenv("LLAMA_API_KEY", "changethisverylongboringkey"), help="API Key")
    parser.add_argument("--model", default="Qwen36 her vis", help="Alias o nome del modello (es. 'Qwen36 her vis' o 'Qwen36 her jin')")
    parser.add_argument("--turns", type=int, default=10, help="Numero di turni di chat da eseguire (default: 10)")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per risposta (default: 2048)")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in secondi per richiesta (default: 60)")
    parser.add_argument("--stream", action="store_true", default=True, help="Usa streaming per calcolare con precisione i token/s")
    args = parser.parse_args()

    print("=" * 80)
    print("🚀 PASCAL FRANKENSTEIN MULTI-TURN STRESS & BENCHMARK")
    print(f"Target URL   : {args.url}")
    print(f"Model Alias  : {args.model}")
    print(f"Max Turns    : {args.turns}")
    print(f"Max Tokens/T : {args.max_tokens}")
    print(f"Initial VRAM : {format_vram(get_vram_stats())}")
    print("=" * 80)

    # 1. Healthcheck
    headers = {"Authorization": f"Bearer {args.api_key}"}
    try:
        r = requests.get(f"{args.url}/v1/models", headers=headers, timeout=5)
        r.raise_for_status()
        models_data = r.json()
        print(f"✅ Connessione stabilita con successo a llama-server.")
        print(f"   Modelli disponibili sul server: {[m.get('id') for m in models_data.get('data', [])]}")
    except Exception as e:
        print(f"❌ Impossibile contattare {args.url}: {e}")
        print("   Assicurati che llama-server sia avviato (es. con genai-llamacpp.sh).")
        sys.exit(1)

    messages = [
        {"role": "system", "content": "Sei un assistente AI specializzato in architetture hardware e sistemi distribuiti. Rispondi in modo tecnico e conciso."}
    ]

    results = []
    total_tokens_all_turns = 0

    for i in range(args.turns):
        turn_num = i + 1
        prompt = TURNS_PROMPTS[i % len(TURNS_PROMPTS)]
        messages.append({"role": "user", "content": prompt})

        print(f"\n--- [Turno {turn_num}/{args.turns}] ----------------------------------------------------")
        print(f"👤 User: {prompt[:80]}...")

        payload = {
            "model": args.model,
            "messages": messages,
            "max_tokens": args.max_tokens,
            "temperature": 0.6,
            "stream": True,
            "stream_options": {"include_usage": True}
        }

        t_start = time.perf_counter()
        t_first = None
        assistant_reply = []
        reasoning_tokens = 0
        content_tokens = 0
        timings = None

        try:
            resp = requests.post(
                f"{args.url}/v1/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=args.timeout
            )
            resp.raise_for_status()

            for line in resp.iter_lines():
                if not line:
                    continue
                line_text = line.decode("utf-8")
                if line_text.startswith("data: "):
                    data_str = line_text[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        if "timings" in chunk:
                            timings = chunk["timings"]
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            reasoning = delta.get("reasoning_content", "")
                            if reasoning:
                                if t_first is None:
                                    t_first = time.perf_counter()
                                reasoning_tokens += 1
                                assistant_reply.append(reasoning)
                                print("~", end="", flush=True)
                            elif content:
                                if t_first is None:
                                    t_first = time.perf_counter()
                                content_tokens += 1
                                assistant_reply.append(content)
                                print(".", end="", flush=True)
                    except json.JSONDecodeError:
                        pass

            t_end = time.perf_counter()
            print() # Nuova linea dopo i simboli

            total_tokens = reasoning_tokens + content_tokens
            full_reply = "".join(assistant_reply)
            messages.append({"role": "assistant", "content": full_reply})

            total_time = t_end - t_start
            ttft = (t_first - t_start) if t_first else total_time
            gen_time = (t_end - t_first) if t_first and (t_end > t_first) else total_time
            tok_per_sec = (total_tokens / gen_time) if gen_time > 0 else 0

            # Se llama-server fornisce i timings nativi interni, usiamo quelli per massima precisione
            prefill_tps = None
            server_gen_tps = None
            mtp_rate = None
            if timings:
                prefill_tps = timings.get("prompt_per_second")
                server_gen_tps = timings.get("predicted_per_second")
                draft_n = timings.get("draft_n", 0)
                draft_acc = timings.get("draft_n_accepted", 0)
                if draft_n > 0:
                    mtp_rate = (draft_acc / draft_n) * 100

            vram_now = get_vram_stats()
            total_tokens_all_turns += total_tokens

            turn_data = {
                "turn": turn_num,
                "reasoning_tokens": reasoning_tokens,
                "content_tokens": content_tokens,
                "total_tokens": total_tokens,
                "ttft_s": ttft,
                "tok_per_sec": server_gen_tps or tok_per_sec,
                "prefill_tps": prefill_tps,
                "mtp_rate": mtp_rate,
                "vram": vram_now
            }
            results.append(turn_data)

            print(f"🤖 Bot ({total_tokens} tok [Ragionamento: {reasoning_tokens}, Risposta: {content_tokens}]): {full_reply[:70]}...")
            
            perf_str = f"⏱️  Prefill: {ttft:.3f}s"
            if prefill_tps:
                perf_str += f" ({prefill_tps:.1f} t/s)"
            perf_str += f" | Generazione: {(server_gen_tps or tok_per_sec):.1f} tok/s"
            if mtp_rate is not None:
                perf_str += f" (MTP Hit: {mtp_rate:.1f}%)"
            perf_str += f" | Tempo: {total_time:.2f}s"
            print(perf_str)
            print(f"📊 {format_vram(vram_now)}")

        except requests.exceptions.Timeout:
            print(f"\n❌ [TIMEOUT] Il server non ha risposto entro {args.timeout}s al Turno {turn_num}!")
            print(f"   Possibile DEADLOCK o context shift bloccato.")
            print(f"   VRAM attuale: {format_vram(get_vram_stats())}")
            break
        except Exception as e:
            print(f"\n❌ [ERRORE] Eccezione al Turno {turn_num}: {e}")
            print(f"   VRAM attuale: {format_vram(get_vram_stats())}")
            break

    # Riepilogo finale
    print("\n" + "=" * 80)
    print("📈 RIEPILOGO PRESTAZIONI & STABILITÀ")
    print("=" * 80)
    if results:
        avg_speed = sum(r["tok_per_sec"] for r in results) / len(results)
        max_speed = max(r["tok_per_sec"] for r in results)
        avg_ttft = sum(r["ttft_s"] for r in results) / len(results)
        prefill_vals = [r["prefill_tps"] for r in results if r.get("prefill_tps") is not None]
        avg_prefill = (sum(prefill_vals) / len(prefill_vals)) if prefill_vals else None
        mtp_vals = [r["mtp_rate"] for r in results if r.get("mtp_rate") is not None]
        avg_mtp = (sum(mtp_vals) / len(mtp_vals)) if mtp_vals else None

        print(f"Turni completati con successo : {len(results)}/{args.turns}")
        print(f"Token totali generati         : {total_tokens_all_turns}")
        print(f"Velocità media di generazione : {avg_speed:.2f} tok/s")
        print(f"Velocità di picco             : {max_speed:.2f} tok/s")
        if avg_prefill:
            print(f"Velocità media di prefill     : {avg_prefill:.2f} tok/s")
        if avg_mtp:
            print(f"MTP Speculative Hit Rate medio: {avg_mtp:.1f}%")
        print(f"Latenza media TTFT            : {avg_ttft:.3f}s")
        if len(results) == args.turns:
            print("🏆 STATO: NESSUN BLOCCO RILEVATO! La sessione multi-turno è stabile.")
        else:
            print("⚠️  STATO: BLOCCO RILEVATO! Esaminare i parametri di contesto e context-shift.")
    else:
        print("Nessun dato raccolto.")
    print("=" * 80)

if __name__ == "__main__":
    main()

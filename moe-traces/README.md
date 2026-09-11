# Operational routing profiles

This directory intentionally contains only the consolidated v1 profiles used by
the documented experiments:

- `qwen36-35b-merged.csv`: Qwen3.6-35B-A3B standard GGUF profile, SHA-256
  `04f25d0f4ea68eab1e125062d963e5b4145a90bdb3c273dc0499cbe3243114ec`.
- `qwen36-35b-mtp-merged.csv`: Qwen3.6-35B-A3B-MTP-UD profile, SHA-256
  `1ad2a9636abd70291a177f25fed859edb4d00365599db593bdea151f6f5e5e0f`.

Pass the appropriate file to `--moe-cache-profile`. The upstream fork documents
how to regenerate a profile with `llama-moe-trace`: capture contrasting
workloads for the same model, then concatenate the resulting CSV files.

The larger multi-domain v2 study was not adopted for the MTP operating profile
and is kept outside this lean public repository. Its configuration, hashes, and
measured conclusion are recorded in `doc/BASELINE_LOG.md`.

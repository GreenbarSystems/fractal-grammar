\# fractal-grammar
https://doi.org/10.5281/zenodo.21020196


\*\*Persistent behavioral memory for local LLMs. No cloud. No fine-tuning.\*\*



fractal-grammar is a behavioral compression library and Ollama sidecar CLI that learns how you interact with your local AI — then injects that context automatically on every session.



Built on fractal grammar extraction and hyperdimensional computing (HDC, D=10,000). Runs entirely on-device.



\---



\## What's in this repo



| Directory | Description |

|---|---|

| `fractal\_grammar/` | Core library — MinHash dedup, HDC encoding, HDBSCAN clustering, grammar extraction, AssociativeMemory |

| `fg-sync/` | CLI sidecar for Ollama — HTTP proxy, cron pipeline, system prompt injection |

| `whitepaper/` | Fractal Behavioral Grammar hypothesis — full whitepaper PDF and markdown |



\---



\## The problem



Every Ollama session starts cold. You re-explain your stack. You correct the same misunderstandings. You re-establish context — every time.



Naive fixes (pasting a system prompt, RAG over chat history) either consume your context window or require external infrastructure.



\## The solution



fractal-grammar compresses your behavioral patterns into a \~12KB ruleset using fractal grammar extraction. fg-sync injects that ruleset as a system prompt prefix on every Ollama request — automatically, locally, without touching model weights.



\---



\## Measured results



| Metric | Value |

|---|---|

| Storage vs raw conversation history | \~82:1 compression |

| AssociativeMemory at n=10,000 events | 39KB flat |

| Tests passing | 70/70 |



\---



\## Quick start (fg-sync)



```bash

pip install fg-sync\[pipeline]

fg-sync init

fg-sync run

\# Point your Ollama client at http://localhost:11435

```



Full docs: \[fg-sync/README.md](fg-sync/README.md)



\---



\## Whitepaper



\[Fractal Behavioral Grammar: A Hypothesis for Behavioral Compression in Local LLMs](whitepaper/fractal\_behavioral\_grammar\_whitepaper.pdf)



\---



\## License



Apache 2.0 — see \[LICENSE](LICENSE)



Built by \[Ryan Moore](https://github.com/GreenbarSystems) | Goodyear, Arizona


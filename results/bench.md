CPU: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz. Single-threaded Python 3.12.14.

### Training on TinyStories train (19.9 MB, 12,642 unique pre-tokens; pre-tokenization 2.6 s)

| vocab | fast (s) | naive (s) | speed-up | identical merges | HF tokenizers, Rust (s) |
|---|---|---|---|---|---|
| 512 | 0.4 | 9.9 | 27× | yes | 2.5 |
| 1,024 | 0.6 | 22.8 | 41× | yes | 4.7 |
| 2,048 | 2.0 | 50.2 | 25× | yes | 3.0 |
| 4,096 | 0.6 | 110.7 | 179× | yes | 4.5 |
| 8,192 | 0.7 | 215.9 | 290× | yes | 5.1 |
| 16,384 | 1.2 | 364.5 | 305× | yes | 6.6 |
| 32,768 | 1.3 | 288.8 | 226× | yes | 5.7 |

### Encoding with GPT-2's merges (MB/s; output identical to tiktoken)

| corpus | MB | tokens | heap + cache | heap | simple | tiktoken (Rust) |
|---|---|---|---|---|---|---|
| prose (TinyStories test) | 2.27 | 564,837 | 6.91 | 1.32 | 1.85 | 18.23 |
| code | 0.62 | 285,720 | 4.23 | 1.49 | 1.85 | 10.63 |

### One long pre-token of n random letters (seconds)

| n | simple | heap | tiktoken (Rust) |
|---|---|---|---|
| 256 | 0.002 | 0.000 | 0.000 |
| 1,024 | 0.040 | 0.001 | 0.000 |
| 4,096 | 0.538 | 0.003 | 0.001 |
| 16,384 | 12.146 | 0.023 | 0.002 |
| 65,536 | 149.594 | 0.091 | 0.008 |
| 262,144 | – | 0.611 | 0.054 |

### Compression: bytes per token on held-out text (higher is better)

| tokenizer | prose | code |
|---|---|---|
| BPE-8192 on prose, gpt2 pattern | 4.06 | 1.88 |
| BPE-8192 on prose, cl100k pattern | 4.18 | 1.62 |
| BPE-8192 on code, gpt2 pattern | 2.77 | 3.69 |
| BPE-8192 on code, cl100k pattern | 2.77 | 3.67 |
| gpt2 (50,257) | 4.02 | 2.16 |
| cl100k_base (100,277) | 4.15 | 3.91 |

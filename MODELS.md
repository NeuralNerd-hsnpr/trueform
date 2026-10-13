# Models

Only models whose licence allows commercial use. Licences were read from each
model's Hub metadata and model card at the pinned revision.

| Model | Used for | Revision | Licence |
|---|---|---|---|
| [google/electra-small-discriminator](https://huggingface.co/google/electra-small-discriminator) | tagger encoder (pretrained weights); also architecture and tokenizer of the from-scratch tagger | `fa8239aadc095e9164941d05878b98afe9b953c3` | Apache-2.0 |
| [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | LoRA fine-tune, constrained decoding, zero-shot baseline; tokenizer in tests | `7ae557604adf67be50417f59c2c2f167def9a775` | Apache-2.0 |
| [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | zero-shot baseline | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | Apache-2.0 |

## Checkpoints trained here

| Checkpoint | Base | Trained on | Licence notes |
|---|---|---|---|
| `checkpoints/tagger-electra-small` | electra-small-discriminator | 128,000 WikiText-103 lines | inherits Apache-2.0 (base) and CC BY-SA 3.0 (training data) |
| `checkpoints/tagger-electra-small-scratch` | none (random init; ELECTRA tokenizer vocabulary) | same | tokenizer from Apache-2.0 model; training data CC BY-SA 3.0 |

The "from scratch" tagger has no pretrained weights in its history, but it
still uses ELECTRA's WordPiece vocabulary file.

The code in this repository is MIT. Each checkpoint additionally carries its
base model's licence and the licences of its training data (see DATA.md).

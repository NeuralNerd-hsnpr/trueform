# Data

All training and test pairs are built by `scripts/prepare_data.py` from the
sources below. The script downloads them, checks their SHA-256 against
`data/manifest.json`, and stops if a source file has changed.

Each pair is `{"input", "target"}`: the input is the target lowercased with
its leading and trailing punctuation removed from every word. Internal
characters (apostrophes, hyphens, decimal points) stay part of the word.
Marks are folded into four classes: none, `,` (also `:`), `.` (also `;` `!`)
and `?`. Quotes, brackets and dashes are dropped, and WikiText parentheticals
are removed.

## Sources

| Source | Use | Licence | Version | SHA-256 of downloaded file |
|---|---|---|---|---|
| [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) `wikitext-103-raw-v1/train-00000-of-00002.parquet` | training | CC BY-SA 3.0 / GFDL | revision `b08601e04326c79dfdd32d625aee71d232d685c3` | `74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212` |
| same, `validation-00000-of-00001.parquet` | dev | CC BY-SA 3.0 / GFDL | same | `204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c` |
| same, `test-00000-of-00001.parquet` | in-domain test | CC BY-SA 3.0 / GFDL | same | `5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91` |
| Project Gutenberg #1661, *The Adventures of Sherlock Holmes* | out-of-domain test | public domain (USA) | `pg1661.txt` from the gutenberg.pglaf.org mirror, Oct 2026 | `922e2a12ccb43a4c9544c260b2166c6ad2097aeb5957faeee113f173bb857cd0` |
| Project Gutenberg #1342, *Pride and Prejudice* | out-of-domain test | public domain (USA) | `pg1342.txt`, same | `3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75` |
| Project Gutenberg #74, *The Adventures of Tom Sawyer* | out-of-domain test | public domain (USA) | `pg74.txt`, same | `74d77384b123a6360db9ab58463cff8b38df8525fbdc6000c81ee388e1f3cf10` |
| Project Gutenberg #11, *Alice's Adventures in Wonderland* | out-of-domain test | public domain (USA) | `pg11.txt`, same | `01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e` |

The Project Gutenberg licence header and footer are stripped; the remaining
text is public domain in the USA. Check the status in your own country
before redistributing it.

The WikiText-derived files in `data/` (dev, in-domain test, CI sample) are
CC BY-SA 3.0, with attribution to the Wikipedia contributors via the WikiText
dataset (Merity et al., 2016).

## Built files

| File | Lines | SHA-256 |
|---|---|---|
| `data/train/train.jsonl` (not committed, rebuilt by the script) | 300,000 | `9c4d92395ad0c839fd178fea74cb20f1c41d646d76df27b869bae1dfdefdbf85` |
| `data/dev.jsonl` | 2,000 | `6f51210f60ba492df428455991184e65f778212656a046a96daac6d6fdf6ced2` |
| `data/test_in_domain.jsonl` | 1,000 | `52276276ae0c0e72321388b2d45fc2a17e5dae2ea9874691dc26830398765064` |
| `data/test_out_of_domain.jsonl` | 1,000 | `05d120fd641af8a3c9f9e6ebee536624fca09bc77672f55d8098e8040ea7ed2c` |
| `data/ci_sample.jsonl` (first 200 in-domain test lines) | 200 | `6023c452cdb7f24ce78b50bebe1ce74c275c53bf1ec69819168536dbfdc714dd` |

WikiText's train/validation/test splits come from different articles. Lines
are segments of 6–60 words made of whole sentences, so most hold one to
three sentences.

## Overlap check

Every dev and test line is compared with the 300,000 training lines: a line
is dropped if it equals a training line or shares any sentence of four or
more words with one (exact match after lowercasing and removing marks).
From `data/manifest.json`:

| Set | Candidates | Dropped for overlap | Kept |
|---|---|---|---|
| dev | 4,000 | 47 | 2,000 |
| in-domain test | 2,000 | 25 | 1,000 |
| out-of-domain test | 2,000 | 1 | 1,000 |

A second pass over the final files finds 0 overlapping lines in each set.

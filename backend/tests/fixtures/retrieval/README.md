# Keyword retrieval evaluation

This fixed dataset measures the deterministic KeywordRetriever contract before
semantic retrieval is introduced. It contains exactly 68 cases:

| Category | Count | Purpose |
|---|---:|---|
| canonical | 7 | One exact name for each supported test |
| alias | 26 | Every alias from the approved curated KB |
| normalized_variant | 21 | Three formatting variants per test |
| unsupported | 14 | Safe rejection of unknown or unsupported input forms |

Formatting variants exercise only case, whitespace, and supported hyphen/dash
normalization. Query whitespace must be preserved. Unsupported cases include
generic terms, partial names, extra words, misspellings, other tests, and blanks.
They describe this baseline's rejection contract, not universal semantic labels.

This is a deterministic functional baseline, not a statistically representative
sample of real-world medical language. A separate semantic/paraphrase dataset
will be created later. Expected labels must never be generated from predictions.
Version the fixture when its queries or labels change.

From backend, run `python -m agents.evaluate_keyword_retrieval` to write
`evaluation_results/keyword_baseline.json`. Generated reports are ignored by Git;
the fixture and this README remain version-controlled. Reports include per-query
outcomes, metrics, and hashes. KB hashing uses sorted filenames and length-prefixed
filename/raw-byte pairs. No source websites are contacted.

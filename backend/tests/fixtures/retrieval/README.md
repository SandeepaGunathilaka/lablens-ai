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

# Semantic ranked-search benchmark

`semantic_queries.json` is independent of the exact keyword contract above. It
contains 144 author-labelled, fixed cases, written before inspecting predictions:
21 direct semantic phrases, 21 paraphrases, 21 descriptions, 21 natural-language
questions, 40 unsupported medical queries, and 20 unrelated queries.
Each supported test has 12 cases (three per supported category).

The fixed calibration split has 56 supported and 40 negative cases (96 total).
The held-out split has 28 supported and 20 negative cases (48 total). Every
supported test/category has two calibration cases and one held-out case.
Medical negatives split 28/12 and unrelated negatives split 12/8.

Labels are explicitly authored from the approved KB, never generated from
retriever predictions. Wording families share group IDs and never cross splits.
For supported tests the calibration families emphasize function/relationships;
held-out families emphasize measurement, alternative descriptions/nicknames, or
stored-energy wording. These are small, deliberately separated author-created
families, not independent samples of real patient language. Negative families
are separated by subject. Group IDs enforce structural separation; human review
of semantic overlap remains useful. Do not tune wording or relabel failures after
viewing results. No canonical names/aliases alone serve as supported queries.

Run `python -m agents.evaluate_semantic_retrieval` from backend. The runner uses
the cached model only and queries the existing index without rebuilding. It
requests all seven candidates, enabling full MRR. Hit@3 equals Recall@3 because
there is exactly one relevant document per supported query. Supported errors
remain in denominators. Missing/error scores are counted separately and excluded
from descriptive score statistics, never converted into successful rejections.

There is no threshold or acceptance policy. Negative queries still receive
nearest candidates; their scores are descriptive statistics, not accuracy.
Correct and incorrect supported Top-1 scores are also summarized, separately by
split. Keyword comparison uses the same 84 supported queries, with no-match
counting as failure to retrieve the labelled document.

The report is `evaluation_results/semantic_baseline.json`, with query-level
rankings, metrics, UTC timestamp, dataset/KB hashes, model and index provenance.
The dataset is tracked; generated reports are already ignored. Threshold
calibration is a later step and must use only the calibration split. Held-out
results shown in this baseline must not guide threshold selection, fixture edits,
or model tuning; doing so would require a new untouched evaluation set.

# Acceptance-policy calibration

Run `python -m agents.calibrate_semantic_retrieval` from backend.

- Calibration uses the 96 calibration cases only (56 supported, 40 negative).
- The conjunctive policy evaluates both Top-1 similarity and rank-1 / rank-2 similarity margin (`top1_similarity >= similarity_threshold AND similarity_margin >= margin_threshold`).
- Predeclared selection constraints:
  - `accepted_precision >= 0.95`
  - `negative_false_accept_rate <= 0.05`
  - Maximizes `supported_correct_coverage` among qualifying candidates.
- The held-out split (48 cases) is strictly firewalled and not used for threshold candidates, selection, or reporting.
- Calibration selected a policy with `similarity_threshold = 0.0` and `margin_threshold = 0.22541916370391846`, achieving 100% precision, 0% negative false-accept rate, and 12.5% supported coverage (7/56) with 0 wrong accepts on calibration data. Full results are written to `evaluation_results/semantic_calibration.json`.


# Portable benchmark fingerprints

The semantic fixture's canonical LF SHA-256 is
`cf799511635cb44add34e8710933142a45a1c717a8f712551f7f42809b1dd0e1`.
The historical `e481c484da856d1488b2bd728cb4c0732b577d4bea23c3a52156271c2f5e890b`
fingerprint used CRLF between lines but a single LF after the final closing brace.
Converting only those line endings reproduces the historical hash exactly; query
wording, labels, splits, and ordering are unchanged. `.gitattributes` now pins both
benchmark JSON files and the KB JSON files to LF. Runtime provenance checks still
hash exact bytes, so content edits continue to invalidate frozen evaluations.

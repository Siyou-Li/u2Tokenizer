# data_synthesis.py Specification

Unified CLI for medical radiology data synthesis preprocessing. Provides four subcommands: `vqa`, `report`, `vqa_translation`, `report_translation`.

## Global Option

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-v` / `--verbose` | bool | `False` | DEBUG level logging (including third-party libraries) |

---

## Subcommand: `vqa`

Generate VQA (question / thinking / answer) data from radiology reports via LLM pipeline.

### Pipeline

For each report, sequentially:

1. **Question Generation** — LLM generates numbered question list from the report (`question.prompt`).
2. **Thinking + Answer Synthesis** — For each question, LLM produces `Thinking: ...` and `Answer: ...` (`thinking.prompt`).
3. **Filter** — LLM judges each QA pair against the report; keep only those starting with `"Yes"` (`filter.prompt`).
4. **Refine** — LLM rewrites thinking to remove report references (`edit.prompt`). Result stored as `refined_thinking`.

A QA triple is discarded if any of `question`, `thinking`, `answer` is <= 20 characters.

### Arguments

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `-d` / `--dataset` | `{ct_rate, amos_mm, abdomen_atlas}` | No | `ct_rate` | Dataset type |
| `-i` / `--input-file` | str | **Yes** | — | Input file path (CSV or JSON depending on dataset) |
| `-o` / `--output-file` | str | **Yes** | — | Output JSONL file path |
| `--data-type` | str | No | `train` | Data split key (`ct_rate`: train/valid; `amos_mm`: training/validation) |
| `--split` | str | No | `train` | Split CSV suffix for `abdomen_atlas` (loads `IID_{split}.csv`) |
| `--start-line` | int | No | `0` | Start index in input data |
| `--end-line` | int | No | `None` (all) | End index in input data |
| `--batch-size` | int | No | `2` | Batch size (number of records per processing iteration) |
| `--skip-batches` | int | No | `0` | Skip first N batches |
| `--max-concurrent` | int | No | `5` | Max concurrent LLM requests within a batch |
| `--max-concurrent-reports` | int | No | `1` | Max reports processed in parallel |
| `--retry-times` | int | No | `3` | Retries on LLM failure |
| `--retry-delay` | float | No | `1.0` | Delay between retries (seconds) |
| `--test-mode` | bool | No | `False` | Process only the first batch then stop |

### Input Format

| Dataset | File Format | Report Field | Image Path Construction |
|---------|-------------|--------------|-------------------------|
| `ct_rate` | CSV | `Findings_EN` | `CT-RATE/dataset/{data_type}/{p0}_{p1}/{p0}_{p1}_{p2}/{VolumeName}` |
| `amos_mm` | JSON (keyed by `data_type`) | `item["labels"]["report"]["findings"][loc]` for `loc` in `{chest, abdomen, pelvis}` (non-empty only) | `AMOS-MM/{image}` (strips leading `./`) |
| `abdomen_atlas` | CSV + split CSV (`IID_{split}.csv`) | `structured report` | `AbdomenAtlas3.0Report/{BDMAP ID}/ct.nii.gz` |

### Output Format (JSONL)

Each line:

```json
{
  "image":    "string — constructed image path",
  "question": "string — generated question",
  "thinking": "string — refined thinking (report references removed)",
  "answer":   "string — generated answer"
}
```

Records are grouped by `image` and flushed atomically per image. Supports resume: already-processed images (by `image` field in output) are skipped.

---

## Subcommand: `report`

Generate report-level thinking from VQA thinking data. Aggregates per-image QA thinkings, then asks LLM to paraphrase into a coherent thinking progress (`report_thinking.prompt`).

### Pipeline

1. Load input JSONL into DataFrame, group by `image`.
2. For each image, concatenate all `question + thinking + answer` as `thinking_before`.
3. LLM paraphrases `thinking_before` into `thinking_after`.
4. Randomly assign a general question (from `general_questions` list).
5. Optionally look up original report text from `--original-input`.

### Arguments

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `-i` / `--input-file` | str | **Yes** | — | Input JSONL (output of `vqa`) |
| `-o` / `--output-file` | str | No | `{input}_report_thinking.jsonl` | Output JSONL path |
| `--batch-size` | int | No | `10` | Batch size for concurrent LLM calls |
| `--max-concurrent` | int | No | `5` | Max concurrent LLM requests |
| `--retry-times` | int | No | `3` | Retries on failure |
| `--retry-delay` | float | No | `1.0` | Delay between retries (seconds) |
| `--enable-batch` | bool | No | `False` | Use batch mode (all images via `batch_query`) vs sequential mode |
| `--resume` / `--no-resume` | bool | No | `True` | Skip already-processed images |
| `--original-input` | str | No | `None` | Original dataset file for report field lookup |
| `-d` / `--dataset` | `{ct_rate, amos_mm, abdomen_atlas}` | No | `ct_rate` | Dataset type (for `--original-input`) |
| `--data-type` | str | No | `train` | Data split key (for `--original-input`) |
| `--split` | str | No | `train` | Split suffix (for `--original-input`, `abdomen_atlas` only) |

### Input Format (JSONL)

Each line must contain:

```json
{
  "image":    "string",
  "question": "string",
  "thinking": "string",
  "answer":   "string",
  "report":   "string"
}
```

### Output Format (JSONL)

Each line:

```json
{
  "image":           "string — image path",
  "report":          "string — original report text",
  "question":        "string — randomly selected general question (English)",
  "thinking_before": "string — concatenated QA thinkings for this image",
  "thinking_after":  "string — LLM-paraphrased coherent thinking progress"
}
```

Supports resume by `image` field.

---

## Subcommand: `vqa_translation`

Translate VQA thinking data (question, thinking, answer) from source to target language.

### Pipeline

1. Load input JSONL, skip already-processed images.
2. Collect all `question`, `thinking`, `answer` texts (3 per record).
3. Batch translate via LLM.
4. Reassemble: original question saved as `original_question`, translated fields overwrite `question`, `thinking`, `answer`.
5. Write grouped by `image`.

### Arguments

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `-i` / `--input-file` | str | **Yes** | — | Input JSONL |
| `-o` / `--output-file` | str | **Yes** | — | Output JSONL |
| `--source-lang` | str | No | `English` | Source language |
| `--target-lang` | str | No | `Chinese` | Target language |
| `--max-concurrent` | int | No | `1` | Max concurrent translation requests |

### Input Format (JSONL)

Each line must contain:

```json
{
  "image":    "string",
  "question": "string",
  "thinking": "string",
  "answer":   "string"
}
```

### Output Format (JSONL)

Each line:

```json
{
  "image":             "string",
  "question":          "string — translated",
  "thinking":          "string — translated",
  "answer":            "string — translated",
  "original_question": "string — pre-translation question",
  "report":            "string — preserved from input"
}
```

Supports resume by `image` field (always checks output file).

---

## Subcommand: `report_translation`

Translate report thinking data (thinking_after, report) from source to target language. Questions are mapped via a pre-built `general_questions` → `general_questions_chinese` lookup table (not LLM-translated).

### Pipeline

1. Load input JSONL, skip already-processed images if `--resume`.
2. For each record: if `question` not in lookup table, reassign to a random `general_questions` entry.
3. Batch translate `thinking_after` and `report` fields via LLM.
4. Map `question` to Chinese equivalent via lookup.
5. Drop `thinking_before` field from output.

### Arguments

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `-i` / `--input-file` | str | **Yes** | — | Input JSONL |
| `-o` / `--output-file` | str | **Yes** | — | Output JSONL |
| `--source-lang` | str | No | `English` | Source language |
| `--target-lang` | str | No | `Chinese` | Target language |
| `--max-concurrent` | int | No | `1` | Max concurrent translation requests |
| `--resume` / `--no-resume` | bool | No | `True` | Skip already-processed images |

### Input Format (JSONL)

Each line must contain:

```json
{
  "image":           "string",
  "question":        "string — English general question",
  "thinking_after":  "string",
  "report":          "string"
}
```

### Output Format (JSONL)

Each line:

```json
{
  "image":          "string",
  "question":       "string — Chinese (via lookup table)",
  "thinking_after": "string — translated",
  "report":         "string — translated"
}
```

`thinking_before` field is removed. Supports resume by `image` field.

---

## Resume Mechanism

All subcommands support resume. Implementation: scan output JSONL for existing `image` values, skip any input record whose `image` is already present. Output is appended in `"a"` mode when resuming.

## LLM Backend

All LLM calls go through an OpenAI-compatible API configured in `config["openai_server"]` (`base_url`, `api_key`, `model_name`). Two parameter presets:

| Preset | temperature | top_p | presence_penalty | max_tokens | enable_thinking |
|--------|-------------|-------|------------------|------------|-----------------|
| Thinking | 0.6 | 0.95 | — | 8192 | True |
| Non-Thinking | 0.7 | 0.8 | 1.5 | 8192 | False |

Retries use exponential backoff on `RateLimitError`, `APIError`, `APITimeoutError`, `APIConnectionError`.

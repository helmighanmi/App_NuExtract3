# Extraction guide

## Template types

A NuExtract template is JSON shaped like the desired output. Leaves describe types:

| Type | Meaning |
| --- | --- |
| `verbatim-string` | Exact text from the document (no paraphrasing) |
| `string` | Generic string; light abstraction allowed |
| `integer`, `number` | Numeric values |
| `boolean` | `true` / `false` |
| `date`, `time`, `date-time` | ISO-8601 values |
| `country`, `currency`, `email`, `phone`, `url` | Constrained strings |
| `["a", "b"]` | Enum |
| `[["a", "b"]]` | Multi-enum |
| `[schema]` | Array of items |
| `{...}` | Nested object |

Missing information is returned as `null` (scalars) or `[]` (arrays).

Example:

```json
{
  "invoice_number": "verbatim-string",
  "invoice_date": "date",
  "currency": ["USD", "EUR", "TND", "Other"],
  "line_items": [
    {
      "description": "verbatim-string",
      "quantity": "number",
      "unit_price": "number",
      "total": "number"
    }
  ]
}
```

Bundled templates live in [`configs/templates`](../configs/templates). The optional
`_description` key is stripped before the template is sent to the model.

## Instructions

Use `instructions` for output conventions that the schema cannot express:

- "Use the buyer as `customer` when no bill-to party is present."
- "Output dates as ISO-8601. If the day is missing, output `null`."
- "For `total`, use the amount including tax."
- "Only extract line items that have a non-zero quantity."

Keep instructions short and specific. Overly long instructions can compete with the
document content for attention.

## Thinking mode

| Mode | Temperature | Use when |
| --- | --- | --- |
| Non-thinking (default) | 0.2 | invoices, receipts, forms, clean scans |
| Thinking | 0.6 | ambiguous layouts, multi-column contracts, tables spanning pages |

NuMind recommends starting without reasoning and enabling it selectively. Thinking
outputs are longer and slower, and small models can fall into repetition loops on
long documents; keep `max_output_tokens` bounded.

For Markdown conversion the model card uses `temperature=1.0`; that is the default in
this project (`NUEXTRACT_MARKDOWN_TEMPERATURE`).

## Page routing and DPI

- **Digital PDFs** (native text ≥ `native_text_min_chars` per page) are sent as text.
  This is cheaper and avoids OCR errors on machine-generated text.
- **Scanned PDFs** and images are rendered and sent as images.
- **Mixed documents** are routed as images for the whole chunk if any page is scanned,
  so page ordering stays unambiguous.

DPI guidance:

| DPI | Use |
| --- | --- |
| 150 | fast drafts, large clean text |
| 200 | default for invoices/receipts |
| 250–300 | small fonts, dense tables, handwriting |

Higher DPI means more image tokens, more latency and more KV cache. Cap the input
with `max_pages`.

## Long documents

Documents are split into chunks of `pages_per_request` pages (default 8). Each chunk
is a separate model call; results are merged:

- dicts merge recursively;
- arrays concatenate (line items, transactions, signatories);
- scalars keep the first non-null value.

Consequences:

- Cross-page relationships are only visible within a chunk.
- If a field legitimately appears on multiple pages (e.g. a total), the first
  non-null value wins. Put the authoritative page first or lower
  `pages_per_request` to 1 for strict per-page extraction.
- Set vLLM's `--limit-mm-per-prompt` ≥ `pages_per_request`.

For very long contracts, a two-step pipeline is often better:

1. `POST /v1/markdown` to get clean Markdown;
2. send relevant sections (text-only) to `/v1/extract`.

## JSON Schema → template

```bash
curl -X POST http://localhost:8080/v1/templates/convert \
  -H 'Content-Type: application/json' \
  -d '{"schema": {"type":"object","properties":{"total":{"type":"number"}}}}'
```

From Python:

```python
from pydantic import BaseModel
from nuextract_app.core.templates import convert_pydantic


class Invoice(BaseModel):
    invoice_number: str
    total: float


template, dropped = convert_pydantic(Invoice)
```

The converter handles `$ref`/`$defs`, `anyOf`/`oneOf`, enums, `const`, arrays,
nested objects and common string formats. Branches it cannot represent are returned
in `dropped` and defaulted to `string`.

## Output guarantees (and non-guarantees)

Guaranteed:

- The response is valid JSON.
- Every key present in the template exists in the result (`null`/`[]` if missing).
- Arrays from multiple chunks are merged.

Not guaranteed:

- Type coercion (the model may return `"1,250.00"` for a `number`).
- Business-rule validity (e.g. `total == subtotal + tax`).
- That an enum value was respected.

For strict contracts, validate the result with Pydantic after extraction and route
failures to review. This is intentionally left to the caller because validation
strictness is domain-specific.

## Evaluation

`eval/run_eval.py` computes per-field scores against a golden JSONL set:

- strings: normalized indel similarity;
- numbers/booleans: exact match;
- objects: recursive alignment by key;
- arrays: index alignment with length penalty.

```bash
make sample
make eval ARGS="--dataset eval/datasets/example.jsonl --template invoice"
```

Reports are written to `reports/`. Track parse-failure rate and p95 latency alongside
accuracy; a model that is 3 points better but 2× slower may not be the right choice.

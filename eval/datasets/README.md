# Evaluation datasets

JSONL, one document per line:

```json
{
  "id": "invoice-001",
  "file": "relative/to/this/directory.pdf",
  "template": "invoice",
  "instructions": "Optional extraction instructions",
  "ground_truth": { "invoice_number": "INV-1", "total": 42.0 }
}
```

- `file` is resolved relative to the dataset file.
- `template` can be a bundled name or an inline template object.
- Generate the example invoice with `make sample` before running
  `make eval ARGS="--dataset eval/datasets/example.jsonl"`.

Keep datasets small but representative: 20–50 documents covering the layouts,
languages and edge cases you actually care about beats a large synthetic set.

# Demo POC Analysis Notes

## Evidence
- Checkpoints: `models/v1/model.pt`, `models/latest/model.pt`.
- Metrics: `artifacts/training_metrics.json` with phase gate pass.

## Demo Pipeline
- `data/synthetic.py` → `ingestion/adapters/mock.py` → `ingestion/anonymizer.py` →
  `ingestion/feature_extractor.py` → `ingestion/scheduler.py` (persist) →
  `intelligence/graph_builder.py` → `intelligence/inference.py` →
  `intelligence/cascade.py` → `api/routes/scores.py`, `api/routes/cascade.py`.

## Demo Gaps
- `api/routes/pipeline.py` is stubbed (no trigger endpoint).
- No Docker Compose or E2E demo runner.
- Demo readiness requires a stable seeded run and API verification.

## API Wiring Notes
- `api/routes/scores.py` and `api/routes/cascade.py` are fully implemented against DB tables.
- `api/routes/pipeline.py` is a placeholder; manual run trigger missing.

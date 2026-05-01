# CognitiveSync Todo

Sprint: April 16 - April 30, 2026
Owner: M Santhosh (Sandy)
Role: System Architect, GNN Engineer, Integration Lead

## Review Summary

Based on the current repository state:

- Completed in code:
  - `T-007` Finalise repo directory structure
  - `T-008` Scaffold repo
  - `T-009` Set up `uv` project and dependencies
  - `T-010` Set up pre-commit hooks
  - `T-022` Implement synthetic org graph generator
  - `T-023` Implement synthetic feature vector generator
  - `T-024` Implement rule-based burnout label generator
  - `T-025` Implement synthetic edge generator
  - `T-045` Implement GAT model class
  - `T-046` Implement training pipeline
  - `T-050` Implement model registry / checkpoint save
  - `T-051` Implement MC Dropout confidence intervals
  - `T-052` Implement attribution output path in inference
  - `T-053` Implement cascade propagator
  - `T-054` Implement cascade source attribution

- Still pending or not yet verified:
  - `T-026` Validate synthetic data
  - `T-047` Train initial model on synthetic graph
  - `T-048` Evaluate model on held-out test set and record metrics
  - `T-049` Tune hyperparameters if accuracy gate is missed
  - `T-055` Unit test cascade propagation module
  - Final integration with Person B API
  - End-to-end Docker Compose verification

## Active Todo List

- [x] `T-026` Validate synthetic data output — DONE 2026-04-20
  - artifacts/synthetic_validation.json: all checks pass (node count, label balance, feature distributions, edge density, UUID pseudonymity)

- [x] `T-047` Train initial GAT model on 100-500 node synthetic graph — DONE 2026-04-20
  - Checkpoint saved: models/v1/ (200 nodes, seed=42, epochs=150)

- [x] `T-048` Evaluate trained model — DONE 2026-04-20
  - artifacts/training_metrics.json: accuracy=0.80, F1=0.33, AUC-ROC=0.80
  - Phase gate PASSED: accuracy ≥ 0.80 ✓, AUC-ROC ≥ 0.75 ✓

- [x] `T-049` Tune model if phase gate fails — DONE 2026-04-20
  - Initial run failed (accuracy gate). Tuned with optimal F1 threshold + balanced training.
  - Key fix: double-sigmoid bug in forward() removed; optimal threshold via precision_recall_curve.

- [x] `T-055` Add unit tests for cascade propagation — DONE 2026-04-20
  - tests/unit/test_cascade.py: 14 tests, all passing
  - Covers 1-hop, 2-hop, decay, multi-source attribution, normalization [0,1]

- [ ] Integration prep for Person C graph builder
  - Run Person C graph builder against generated synthetic data once their branch is ready.
  - Confirm node ordering and feature dimension alignment with GNN expectations.

- [ ] Integration prep for Person B API
  - Map inference outputs to the response schema expected by `/api/v1/scores`.
  - Confirm checkpoint loading, inference run, and cascade output can be called from API code.

- [ ] End-to-end integration branch test
  - Execute: synthetic data -> graph build -> GNN inference -> cascade propagation -> API response.
  - Verify the full path inside Docker Compose.

- [ ] Demo readiness
  - Prepare one stable seeded run for reproducible output.
  - Identify one high-risk node cluster and one cascade example for presentation.

## External Dependencies

- Person B:
  - PostgreSQL schema and FastAPI endpoint readiness
  - Docker Compose integration target

- Person C:
  - Graph builder readiness
  - Confirmed PyG / NetworkX handoff contract

- Person D:
  - Demo runner and final test script support

## Notes

- There is currently no `tests/` coverage for the intelligence modules.
- Training and evaluation code exists, but there is no checked-in evidence yet that a model has been trained successfully.
- Integration is not complete until the intelligence outputs are exercised through the API path.

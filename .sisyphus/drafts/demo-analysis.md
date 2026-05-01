# Draft: Demo Analysis

## Requirements (confirmed)
- User wants to "begin a analysis and tell if its a POC now"
- Follow-up focus: "Let's being to work towards the demo"
- Demo focus confirmed: end-to-end flow

## Technical Decisions
- Demo target: end-to-end pipeline first, not isolated API-only polish
- Status: this is already a POC, but not yet demo-ready

## Research Findings
- Background analysis found the end-to-end path:
  synthetic data → anonymization → feature extraction → graph building → GNN inference → cascade propagation → API routes
- Demo-support artifacts exist:
  scripts, checkpoints, metrics, unit tests, mock adapter, synthetic data generator
- Missing demo gaps:
  Docker Compose, API integration verification, full E2E demo runner, stable seeded demo script

## Open Questions
- Should the demo stop at a successful pipeline script run, or include a live FastAPI response too?

## Scope Boundaries
- INCLUDE: end-to-end demo flow, reproducible seeded run, API verification
- EXCLUDE: broader product hardening beyond demo readiness

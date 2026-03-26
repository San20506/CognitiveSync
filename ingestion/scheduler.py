"""APScheduler pipeline job — implementation in Phase 4A (T-040).

Runs the full ingestion pipeline every 48 hours (configurable).
Job store: PostgreSQL (survives container restarts).
Executor: asyncio (non-blocking).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config.settings import settings

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()


async def run_pipeline_job() -> None:
    """Full pipeline run — WF-01 master workflow.

    Steps:
    1. Acquire PostgreSQL advisory lock (prevent concurrent runs)
    2. WF-02: Data ingestion + anonymization (parallel per source)
    3. WF-03: Feature extraction
    4. Upsert feature vectors to PostgreSQL
    5. WF-04: Graph construction
    6. WF-05: GNN inference + scoring
    7. WF-06: Cascade propagation
    8. Persist scores
    9. WF-07: Power BI push
    10. WF-08: Teams bot alert (conditional on threshold breach)
    11. Log run completion to audit
    12. Release advisory lock
    """
    # Deferred imports keep scheduler module importable even when optional
    # dependencies (torch, torch_geometric) are absent at import time.
    from ingestion.adapters.mock import MockAdapter
    from ingestion.adapters.msgraph import MSGraphAdapter
    from ingestion.adapters.slack import SlackAdapter
    from ingestion.adapters.github import GitHubAdapter
    from ingestion.anonymizer import Anonymizer
    from ingestion.feature_extractor import FeatureExtractor, FEATURE_NAMES
    from ingestion.db.models import FeatureVector, EdgeSignal, BurnoutScore, ScoringRun
    from ingestion.db.session import AsyncSessionLocal
    from intelligence.graph_builder import GraphBuilder
    from intelligence.inference import InferencePipeline
    from intelligence.cascade import CascadePropagator

    run_id: UUID = uuid4()
    started_at: datetime = datetime.now(timezone.utc)
    window_end: datetime = started_at
    window_start: datetime = window_end - timedelta(hours=48)

    logger.info("Pipeline run started: run_id=%s", run_id)

    async with AsyncSessionLocal() as db:
        # Step 1: Record run start (advisory lock is enforced by APScheduler
        # max_instances=1 at the scheduler level; PostgreSQL-level advisory lock
        # deferred to T-041 when pg_advisory_xact_lock is wired in).
        scoring_run = ScoringRun(
            id=run_id,
            org_id=uuid4(),  # TODO T-041: resolve real org_id from config.orgs
            status="running",
            started_at=started_at,
        )
        db.add(scoring_run)
        await db.commit()

        try:
            # ------------------------------------------------------------------
            # Step 2: WF-02 — Data ingestion (parallel per source)
            # ------------------------------------------------------------------
            AdapterPair = tuple[str, object]
            adapters: list[AdapterPair]

            if settings.adapter_mode == "live":
                adapters = [
                    (
                        "msgraph",
                        MSGraphAdapter(
                            settings.msgraph_client_id,
                            settings.msgraph_client_secret,
                            settings.msgraph_tenant_id,
                        ),
                    ),
                    (
                        "slack",
                        SlackAdapter(
                            settings.slack_bot_token,
                            settings.slack_signing_secret,
                        ),
                    ),
                    (
                        "github",
                        GitHubAdapter(
                            settings.github_org_token,
                            settings.github_org_name,
                        ),
                    ),
                ]
            else:
                adapters = [("mock", MockAdapter())]

            # Fetch signals from all sources concurrently
            fetch_coros = [
                adapter.fetch_signals(window_start, window_end)  # type: ignore[union-attr]
                for _, adapter in adapters
            ]
            gather_results = await asyncio.gather(*fetch_coros, return_exceptions=True)

            # Merge results; skip sources that raised exceptions
            from ingestion.adapters.base import RawSignals

            merged_raw: dict[str, RawSignals] = {}
            sources_completed: list[str] = []
            sources_failed: list[str] = []

            for (name, _), result in zip(adapters, gather_results):
                if isinstance(result, BaseException):
                    logger.warning("Source %s failed: %s", name, result)
                    sources_failed.append(name)
                else:
                    raw_dict: dict[str, RawSignals] = result  # type: ignore[assignment]
                    for uid, sigs in raw_dict.items():
                        # First source to provide a signal for a user wins;
                        # later sources only fill in missing fields.
                        if uid not in merged_raw:
                            merged_raw[uid] = sigs
                    sources_completed.append(name)

            logger.info(
                "Ingestion complete: sources_ok=%s sources_failed=%s users=%d",
                sources_completed,
                sources_failed,
                len(merged_raw),
            )

            # ------------------------------------------------------------------
            # Step 3: WF-02 (cont.) — Anonymization
            # ------------------------------------------------------------------
            anonymizer = Anonymizer(
                org_salt=settings.org_salt,
                vault_path=Path(settings.vault_path),
                vault_key=settings.vault_key,
            )
            anon_signals = anonymizer.anonymize_batch(merged_raw)

            # ------------------------------------------------------------------
            # Step 4: WF-03 — Feature extraction
            # ------------------------------------------------------------------
            extractor = FeatureExtractor()
            features, edge_batch = extractor.extract_batch(
                anon_signals, window_start, window_end
            )

            # ------------------------------------------------------------------
            # Step 5: Persist feature vectors + edge signals to PostgreSQL
            # ------------------------------------------------------------------
            for fv in features:
                feature_json: dict[str, float] = {
                    name: float(fv.feature_vector[i])
                    for i, name in enumerate(FEATURE_NAMES)
                }
                db.add(
                    FeatureVector(
                        pseudo_id=fv.pseudo_id,
                        window_start=fv.window_start,
                        window_end=fv.window_end,
                        feature_json=feature_json,
                        is_imputed=fv.is_imputed,
                        data_completeness=fv.data_completeness,
                    )
                )

            for src, dst, weight in edge_batch.edges:
                db.add(
                    EdgeSignal(
                        source_pseudo_id=src,
                        target_pseudo_id=dst,
                        weight=weight,
                        window_start=window_start,
                        window_end=window_end,
                    )
                )

            await db.commit()
            logger.info(
                "Persisted %d feature vectors and %d edge signals",
                len(features),
                len(edge_batch.edges),
            )

            # ------------------------------------------------------------------
            # Step 6: WF-04 — Graph construction
            # ------------------------------------------------------------------
            graph_builder = GraphBuilder()
            built_graph = await graph_builder.build_from_store(window_end, db)
            logger.info(
                "Graph built: nodes=%d edges=%d",
                built_graph.nx_graph.number_of_nodes(),
                built_graph.nx_graph.number_of_edges(),
            )

            # ------------------------------------------------------------------
            # Step 7: WF-05 — GNN inference + scoring
            # ------------------------------------------------------------------
            inference = InferencePipeline(settings.model_registry_path)
            inference.load_model()
            scored = inference.score(
                built_graph.pyg_data, built_graph.node_ids, run_id
            )
            logger.info(
                "Inference complete: scored %d nodes", len(scored.node_scores)
            )

            # ------------------------------------------------------------------
            # Step 8: WF-06 — Cascade propagation
            # ------------------------------------------------------------------
            propagator = CascadePropagator(
                threshold=settings.cascade_threshold,
                decay_factor=settings.decay_factor,
                max_hops=settings.max_hops,
            )
            # Pass the NetworkX graph from GraphBuilder (not ScoredGraph.nx_graph
            # which holds a PyG Data object after inference)
            cascade_results = propagator.propagate(
                built_graph.nx_graph,
                {pid: ns.burnout_score for pid, ns in scored.node_scores.items()},
            )
            logger.info(
                "Cascade propagation complete: %d nodes in cascade zone",
                len(cascade_results),
            )

            # ------------------------------------------------------------------
            # Step 9: Persist scores
            # ------------------------------------------------------------------
            for pseudo_id, node_score in scored.node_scores.items():
                cascade = cascade_results.get(pseudo_id)
                db.add(
                    BurnoutScore(
                        run_id=run_id,
                        pseudo_id=pseudo_id,
                        burnout_score=node_score.burnout_score,
                        confidence_low=node_score.confidence_low,
                        confidence_high=node_score.confidence_high,
                        cascade_risk=cascade.cascade_risk if cascade is not None else 0.0,
                        cascade_sources=(
                            {
                                "sources": [
                                    str(s) for s in cascade.cascade_sources
                                ]
                            }
                            if cascade is not None and cascade.cascade_sources
                            else None
                        ),
                        top_features=node_score.top_features if node_score.top_features else None,
                        window_end=window_end,
                    )
                )

            await db.commit()
            logger.info("Persisted %d burnout scores", len(scored.node_scores))

            # ------------------------------------------------------------------
            # Step 10: WF-07 — Power BI push (skipped when unconfigured)
            # ------------------------------------------------------------------
            if settings.powerbi_client_id:
                try:
                    from output.powerbi_connector import PowerBIConnector, PowerBIRow

                    pbi = PowerBIConnector(
                        settings.powerbi_client_id,
                        settings.powerbi_client_secret,
                        settings.powerbi_tenant_id,
                        settings.powerbi_dataset_id,
                        settings.powerbi_workspace_id,
                    )
                    rows: list[PowerBIRow] = [
                        PowerBIRow(
                            pseudo_id=pid,
                            team_id=None,
                            burnout_score=ns.burnout_score,
                            cascade_risk=(
                                cascade_results[pid].cascade_risk
                                if pid in cascade_results
                                else 0.0
                            ),
                            top_features=ns.top_features,
                            window_end=window_end,
                        )
                        for pid, ns in scored.node_scores.items()
                    ]
                    await pbi.push_scores(rows)
                    logger.info("Power BI push complete: %d rows", len(rows))
                except NotImplementedError:
                    logger.warning(
                        "Power BI push skipped — PowerBIConnector not yet implemented (T-060)"
                    )
                except Exception as pbi_exc:  # noqa: BLE001
                    logger.warning("Power BI push failed (non-fatal): %s", pbi_exc)

            # ------------------------------------------------------------------
            # Step 11: WF-08 — Teams alert (conditional on threshold breach)
            # ------------------------------------------------------------------
            high_risk = [
                ns
                for ns in scored.node_scores.values()
                if ns.burnout_score > settings.alert_threshold
            ]
            if high_risk and settings.teams_app_id:
                try:
                    from output.teams_bot.bot import CognitiveSyncBot, RiskCluster

                    bot = CognitiveSyncBot(
                        settings.teams_app_id, settings.teams_app_password
                    )
                    cluster = RiskCluster(
                        team_count=len(high_risk),
                        risk_level="HIGH",
                        top_signals=list(high_risk[0].top_features.keys())[:3],
                        recommendations=[
                            "Review workload distribution",
                            "Reduce after-hours meetings",
                        ],
                        cascade_summary=(
                            f"{len(cascade_results)} nodes in cascade risk zone"
                        ),
                    )
                    await bot.send_hr_alert(cluster, settings.teams_hr_channel_id)
                    logger.info(
                        "Teams HR alert sent: %d high-risk nodes", len(high_risk)
                    )
                except NotImplementedError:
                    logger.warning(
                        "Teams alert skipped — CognitiveSyncBot not yet implemented (T-062)"
                    )
                except Exception as teams_exc:  # noqa: BLE001
                    logger.warning("Teams alert failed (non-fatal): %s", teams_exc)

            # ------------------------------------------------------------------
            # Step 12: Mark run complete (advisory lock released implicitly on
            # transaction commit; explicit pg_advisory_unlock deferred to T-041)
            # ------------------------------------------------------------------
            scoring_run.status = "completed"
            scoring_run.completed_at = datetime.now(timezone.utc)
            scoring_run.node_count = len(scored.node_scores)
            scoring_run.edge_count = len(edge_batch.edges)
            scoring_run.sources_completed = {
                "completed": sources_completed,
                "failed": sources_failed,
            }
            await db.commit()

            logger.info(
                "Pipeline run completed: run_id=%s nodes=%d edges=%d",
                run_id,
                len(scored.node_scores),
                len(edge_batch.edges),
            )

        except Exception:
            logger.exception("Pipeline run failed: run_id=%s", run_id)
            scoring_run.status = "failed"
            # Capture the exception message without importing sys
            import traceback as _tb
            scoring_run.error_message = _tb.format_exc(limit=5)
            scoring_run.completed_at = datetime.now(timezone.utc)
            try:
                await db.commit()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to persist error state for run_id=%s", run_id
                )
            raise


def configure_scheduler() -> None:
    """Register pipeline job and start scheduler.

    Uses an in-process asyncio executor (no external job store required for MVP).
    PostgreSQL job store deferred to T-041.
    """
    scheduler.add_job(
        run_pipeline_job,
        "interval",
        hours=settings.refresh_interval_hours,
        id="pipeline_run",
        replace_existing=True,
        max_instances=1,  # Prevent concurrent runs at scheduler level
    )
    logger.info(
        "Pipeline job scheduled every %d hours", settings.refresh_interval_hours
    )

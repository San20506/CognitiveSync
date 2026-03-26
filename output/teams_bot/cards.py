"""Adaptive Card builders for HR and Manager alerts.

Implementation in Phase 4C (T-063, T-064).

Card schemas (per WORKFLOWS.md WF-08):

HR Card fields:       Risk level badge, teams affected count, cascade risk summary,
                      top contributing signals, redistribution recommendations,
                      focus time protection suggestion, Power BI dashboard link.
                      NEVER: individual names, individual scores.

Manager Card fields:  Teams affected count, redistribution recommendations,
                      meeting reduction suggestion, focus time suggestion.
                      NEVER: risk scores, pseudo_ids, individual references.
"""

from __future__ import annotations

from output.teams_bot.bot import RiskCluster


def build_hr_adaptive_card(
    cluster: RiskCluster,
    dashboard_url: str,
) -> dict[str, object]:
    """Build HR Adaptive Card payload (T-063).

    Returns an Adaptive Card 1.4 JSON dict containing only aggregated
    cluster-level data.  Individual names and scores are NEVER included.
    """
    title_color: str = "Warning" if cluster.risk_level == "MEDIUM" else "Attention"

    body: list[dict[str, object]] = [
        {
            "type": "TextBlock",
            "text": "CognitiveSync Burnout Risk Alert",
            "weight": "Bolder",
            "size": "Large",
            "color": title_color,
        },
        {
            "type": "FactSet",
            "facts": [
                {"title": "Risk Level", "value": cluster.risk_level},
                {"title": "Teams Affected", "value": str(cluster.team_count)},
                {"title": "Top Signals", "value": ", ".join(cluster.top_signals)},
            ],
        },
    ]

    if cluster.cascade_summary:
        body.append(
            {
                "type": "TextBlock",
                "text": f"Cascade Summary: {cluster.cascade_summary}",
                "wrap": True,
                "color": "Warning",
            }
        )

    body.extend(
        [
            {
                "type": "TextBlock",
                "text": "Recommendations",
                "weight": "Bolder",
            },
            {
                "type": "TextBlock",
                "text": "\n".join(f"• {r}" for r in cluster.recommendations),
                "wrap": True,
            },
        ]
    )

    actions: list[dict[str, object]] = (
        [
            {
                "type": "Action.OpenUrl",
                "title": "View Power BI Dashboard",
                "url": dashboard_url,
            }
        ]
        if dashboard_url
        else []
    )

    return {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": body,
        "actions": actions,
    }


def build_manager_adaptive_card(
    recommendations: list[str],
    meeting_reduction_suggestion: str | None = None,
    focus_time_suggestion: str | None = None,
) -> dict[str, object]:
    """Build Manager Adaptive Card payload (T-064).

    Returns an Adaptive Card 1.4 JSON dict containing only team-level
    redistribution guidance.  Burnout scores, pseudo_ids, and any
    individual references are NEVER included.
    """
    body: list[dict[str, object]] = [
        {
            "type": "TextBlock",
            "text": "Team Wellbeing Update",
            "weight": "Bolder",
            "size": "Large",
        },
        {
            "type": "TextBlock",
            "text": (
                "Your team may benefit from workload adjustments. "
                "No individual scores are shared here."
            ),
            "wrap": True,
            "isSubtle": True,
        },
        {
            "type": "TextBlock",
            "text": "Recommendations",
            "weight": "Bolder",
        },
        {
            "type": "TextBlock",
            "text": "\n".join(f"• {r}" for r in recommendations),
            "wrap": True,
        },
    ]

    if meeting_reduction_suggestion:
        body.append(
            {
                "type": "TextBlock",
                "text": f"Meeting Guidance: {meeting_reduction_suggestion}",
                "wrap": True,
                "color": "Accent",
            }
        )

    if focus_time_suggestion:
        body.append(
            {
                "type": "TextBlock",
                "text": f"Focus Time: {focus_time_suggestion}",
                "wrap": True,
                "color": "Good",
            }
        )

    return {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": body,
    }

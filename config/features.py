"""Canonical feature definitions — single source of truth for all modules.

This module provides the authoritative list of 13 behavioral features used throughout
the CognitiveSync pipeline. All consumers (ingestion, intelligence, synthetic data)
should import FEATURE_NAMES from here.

Feature Index | Name                  | Source     | Notes
------------- | --------------------- | ---------- | ---------------------------
0             | meeting_density       | MS Graph   |
1             | after_hours_meetings  | MS Graph   |
2             | focus_blocks          | MS Graph   |
3             | email_response_latency| MS Graph   |
4             | meeting_accept_rate   | MS Graph   |
5             | message_volume        | Slack      |
6             | after_hours_messages  | Slack      |
7             | response_time_slack   | Slack      |
8             | mention_frequency     | Slack      |
9             | commit_frequency      | GitHub     |
10            | after_hours_commits   | GitHub     |
11            | pr_review_load        | GitHub     |
12            | context_switch_rate   | GitHub     |

Model Compatibility Note:
--------------------------
The GNN model (BurnoutGAT/SmallBurnoutGAT) was trained on a 10-feature subset
from CSV data. The 13 features above are the canonical source; the graph builder
should select/transform features as needed for model input.

Historical Naming (DO NOT USE):
-------------------------------
The intelligence module previously used different names that are now deprecated:
- after_hours_ratio → use after_hours_meetings
- response_latency_avg → use email_response_latency
- focus_time_blocks → use focus_blocks
- msg_volume_daily → use message_volume
- msg_response_time → use response_time_slack
- mention_load → use mention_frequency

All consumers must use the canonical names above.
"""

from __future__ import annotations

# Canonical 13-dimensional feature list
FEATURE_NAMES: list[str] = [
    "meeting_density",
    "after_hours_meetings",
    "focus_blocks",
    "email_response_latency",
    "meeting_accept_rate",
    "message_volume",
    "after_hours_messages",
    "response_time_slack",
    "mention_frequency",
    "commit_frequency",
    "after_hours_commits",
    "pr_review_load",
    "context_switch_rate",
]

FEATURE_DIM = len(FEATURE_NAMES)  # 13

# Neutral baseline for missing features
NEUTRAL_BASELINE = 0.5

# Alias for compatibility - do not use in new code
DEPRECATED_FEATURE_NAMES: list[str] = [
    "meeting_density",
    "after_hours_ratio",  # deprecated: use after_hours_meetings
    "response_latency_avg",  # deprecated: use email_response_latency
    "focus_time_blocks",  # deprecated: use focus_blocks
    "msg_volume_daily",  # deprecated: use message_volume
    "msg_response_time",  # deprecated: use response_time_slack
    "mention_load",  # deprecated: use mention_frequency
    "context_switch_rate",
    "hrv_avg",
    "sleep_score",
]

# Mapping from canonical names to model-compatible names (for graph builder)
# The model was trained on 10 features with different naming conventions
FEATURE_NAME_MAPPING: dict[str, str] = {
    "after_hours_meetings": "after_hours_ratio",
    "email_response_latency": "response_latency_avg",
    "focus_blocks": "focus_time_blocks",
    "message_volume": "msg_volume_daily",
    "response_time_slack": "msg_response_time",
    "mention_frequency": "mention_load",
}

# Model input features (10-dim) - used by BurnoutGAT/SmallBurnoutGAT
# These are the features the trained model expects
MODEL_FEATURE_NAMES: list[str] = [
    "meeting_density",
    "after_hours_meetings",  # model sees as "after_hours_ratio"
    "email_response_latency",  # model sees as "response_latency_avg"
    "focus_blocks",  # model sees as "focus_time_blocks"
    "message_volume",  # model sees as "msg_volume_daily"
    "response_time_slack",  # model sees as "msg_response_time"
    "mention_frequency",  # model sees as "mention_load"
    "context_switch_rate",
    "hrv_avg",
    "sleep_score",
]

MODEL_FEATURE_DIM = len(MODEL_FEATURE_NAMES)  # 10
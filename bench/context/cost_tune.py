"""Development-only arm declarations for paired cost-per-task tuning."""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import longbench as L

L.MODEL = os.environ["COST_ROOT_MODEL"]
L.RELAY = "zai/glm-5.3-flash"
L.ARMS.update({
    "rule-age5": dict(strategy="clm", policy="rule", rule=dict(offload=8000, age=5, restated=0, tokens=24000, every=0)),
    "rule-off4": dict(strategy="clm", policy="rule", rule=dict(offload=4000, age=3, restated=0, tokens=24000, every=0)),
    "rule-tok32": dict(strategy="clm", policy="rule", rule=dict(offload=8000, age=3, restated=0, tokens=32000, every=0)),
    "host-only": dict(strategy="clm", policy="rule", rule=dict(offload=2000, age=0, restated=0, tokens=0, every=0)),
    "host-recall": dict(strategy="clm", policy="rule", rule=dict(offload=2000, age=0, restated=0, tokens=0, every=0, recall=24000)),
    "tier2-24": dict(strategy="clm", policy="rule", rule=dict(offload=0, age=1, restated=0, tokens=10**9, every=0, summary=400, high=24000, low=12000, recall=24000)),
})
if __name__ == "__main__":
    L.main()

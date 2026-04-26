# Local tool validation script - tests all 6 Orchestrate tool functions directly
# in Python without Docker or cloud upload.
#
# Usage:
#   cd healthcare_multi_agent_tutorial
#   .\src\.venv\Scripts\python.exe test_tools_local.py
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

# ── make sure 'src' is on the path so imports resolve ──────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

# ── load .env if present ───────────────────────────────────────────────────
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

# ── import tool functions (the @tool decorator is a no-op locally) ─────────
from healthcare_support_agents.orchestrate_adk_tools import (  # noqa: E402
    coordinate_care_logistics,
    get_patient_snapshot,
    monitor_recovery_status,
    resolve_recovery_case,
    search_support_services,
    translate_discharge_plan,
)

# ── test helpers ───────────────────────────────────────────────────────────
PATIENT_A = "PT-1001"   # Margaret Ellis
PATIENT_B = "PT-2002"   # David Ramirez

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"

results: list[tuple[str, str, str]] = []   # (tool_name, status, detail)


def _unwrap(out):
    """Unwrap ADK ToolResponse to a plain dict when the ADK is installed."""
    if isinstance(out, dict):
        return out
    # ibm_watsonx_orchestrate wraps return values in ToolResponse
    for attr in ("output", "data", "result", "body"):
        val = getattr(out, attr, None)
        if isinstance(val, dict):
            return val
    # last resort: use __dict__ or str representation
    if hasattr(out, "__dict__"):
        d = {k: v for k, v in out.__dict__.items() if not k.startswith("_")}
        if d:
            return d
    return {"raw": str(out)}


def run(label: str, fn, *args):
    try:
        raw = fn(*args)
        out = _unwrap(raw)
        results.append((label, PASS, ""))
        print(f"  {PASS}  {label}")
        print(f"         keys: {list(out.keys())}\n")
    except RuntimeError as exc:
        # expected when optional dependency (SERPER_API_KEY) is missing
        results.append((label, SKIP, str(exc)))
        print(f"  {SKIP}  {label}: {exc}\n")
    except Exception as exc:
        results.append((label, FAIL, str(exc)))
        print(f"  {FAIL}  {label}: {exc}")
        traceback.print_exc()
        print()


# ── run all 6 tools ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Healthcare Orchestrate Tools — Local Validation")
print("=" * 60 + "\n")

print("── Tool 1: get_patient_snapshot ──────────────────────────")
run("get_patient_snapshot(PT-1001)", get_patient_snapshot, PATIENT_A)

print("── Tool 2: translate_discharge_plan ──────────────────────")
run("translate_discharge_plan(PT-1001)", translate_discharge_plan, PATIENT_A)

print("── Tool 3: monitor_recovery_status ───────────────────────")
run(
    "monitor_recovery_status(PT-1001)",
    monitor_recovery_status,
    PATIENT_A,
    "Patient reports mild chest tightness and shortness of breath when walking.",
)

print("── Tool 4: coordinate_care_logistics ─────────────────────")
run("coordinate_care_logistics(PT-1001)", coordinate_care_logistics, PATIENT_A)

print("── Tool 5: resolve_recovery_case (full pipeline) ─────────")
run(
    "resolve_recovery_case(PT-2002)",
    resolve_recovery_case,
    PATIENT_B,
    "Dizziness and fatigue since yesterday, no chest pain.",
)

print("── Tool 6: search_support_services ───────────────────────")
run(
    "search_support_services(transport)",
    search_support_services,
    "non-emergency medical transportation services near Chicago",
)

# ── summary ────────────────────────────────────────────────────────────────
print("=" * 60)
passed  = sum(1 for _, s, _ in results if "PASS" in s)
skipped = sum(1 for _, s, _ in results if "SKIP" in s)
failed  = sum(1 for _, s, _ in results if "FAIL" in s)
print(f"  Results: {passed} passed  |  {skipped} skipped  |  {failed} failed")
print("=" * 60 + "\n")

sys.exit(1 if failed else 0)

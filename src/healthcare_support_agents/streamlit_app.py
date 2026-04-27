from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, replace
from typing import Any

import streamlit as st

try:
    from .config import AppConfig
    from .orchestrate_deployment import invoke_via_api
    from .orchestrator import Orchestrator
    from .repository import DataRepository
    from .serper_client import SerperSearchClient
    from .tool_wrappers import HealthcareToolRuntime, build_repository
    from .watsonx_client import WatsonxClient
    from .watsonx_orchestrator import WatsonxCareOrchestrator
except ImportError:
    from healthcare_support_agents.config import AppConfig
    from healthcare_support_agents.orchestrate_deployment import invoke_via_api
    from healthcare_support_agents.orchestrator import Orchestrator
    from healthcare_support_agents.repository import DataRepository
    from healthcare_support_agents.serper_client import SerperSearchClient
    from healthcare_support_agents.tool_wrappers import HealthcareToolRuntime, build_repository
    from healthcare_support_agents.watsonx_client import WatsonxClient
    from healthcare_support_agents.watsonx_orchestrator import WatsonxCareOrchestrator


def load_orchestrator() -> tuple[Orchestrator, DataRepository]:
    repository = build_repository()
    return Orchestrator(repository), repository


def load_watsonx_orchestrator(repository: DataRepository) -> WatsonxCareOrchestrator | None:
    config = AppConfig.from_env()
    if not config.watsonx_ready:
        return None
    search_client = SerperSearchClient(config.serper_api_key) if config.serper_ready else None
    runtime = HealthcareToolRuntime(repository, search_client=search_client)
    return WatsonxCareOrchestrator(WatsonxClient(config), runtime)


def severity_label(risk_status: str) -> tuple[str, str]:
    if risk_status == "high_priority_health_shift":
        return "High Priority", "error"
    if risk_status == "watch_closely":
        return "Watch Closely", "warning"
    return "Routine", "success"


def build_orchestrate_prompt(patient_id: str, symptom_report: str) -> str:
    return (
        f'Use patient_id "{patient_id}" and symptom_report "{symptom_report}". '
        "Call resolve_recovery_case first. "
        "Return ONLY a concise tool-grounded summary with four lines exactly: "
        "risk_status, escalation_required, key_findings, immediate_actions. "
        "Do not include code blocks, pseudocode, or implementation examples. "
        "Do not ask follow-up questions."
    )


def clean_orchestrate_summary(text: str) -> str:
    cleaned = re.sub(r"```[\s\S]*?```", "", text or "")
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""

    ignored_prefixes = (
        "### recovery case resolution",
        "#### patient id:",
        "#### symptom report:",
        "output",
        "print(",
        "def ",
        "patient_id =",
        "symptom_report =",
    )
    filtered = [
        line for line in lines
        if not line.lower().startswith(ignored_prefixes)
    ]
    summary = "\n".join(filtered).strip()
    if not summary:
        summary = "\n".join(lines[:6]).strip()
    return summary


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item and isinstance(item["text"], str):
                    parts.append(item["text"].strip())
                elif "content" in item:
                    nested = extract_text(item["content"])
                    if nested:
                        parts.append(nested)
        return "\n".join(part for part in parts if part).strip()
    if isinstance(content, dict):
        if "text" in content and isinstance(content["text"], str):
            return content["text"].strip()
        if "content" in content:
            return extract_text(content["content"])
    return ""


def parse_orchestrate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    run = payload.get("run", {})
    run_meta = run
    if isinstance(run, dict):
        nested_run = run.get("run")
        if isinstance(nested_run, dict):
            run_meta = nested_run

    events = payload.get("events")
    if not isinstance(events, list) and isinstance(run, dict) and isinstance(run.get("events"), list):
        events = run.get("events")
    if not isinstance(events, list):
        events = []

    assistant_messages: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    terminal_event_status = "unknown"
    failure_reason = ""
    terminal_map = {
        "run.completed": "completed",
        "done": "completed",
        "run.failed": "failed",
        "run.cancelled": "cancelled",
    }

    for event in events:
        if not isinstance(event, dict):
            continue
        event_name = event.get("event")
        if isinstance(event_name, str) and event_name in terminal_map:
            terminal_event_status = terminal_map[event_name]
            if event_name in {"run.failed", "run.cancelled"}:
                data = event.get("data", {})
                if isinstance(data, dict):
                    detail = (
                        data.get("last_error")
                        or data.get("error")
                        or data.get("reason")
                        or data.get("message")
                    )
                    if isinstance(detail, str) and detail.strip():
                        failure_reason = detail.strip()

        data = event.get("data", event)
        if not isinstance(data, dict):
            continue
        message = data.get("message")
        if not isinstance(message, dict):
            continue

        if message.get("role") == "assistant":
            text = extract_text(message.get("content"))
            if text:
                assistant_messages.append(text)

        event_tool_calls = message.get("tool_calls", [])
        if isinstance(event_tool_calls, list):
            tool_calls.extend(call for call in event_tool_calls if isinstance(call, dict))

    return {
        "run_id": (
            run_meta.get("id")
            or run_meta.get("run_id")
            or run_meta.get("task_id")
            or run_meta.get("thread_id")
            or "n/a"
        ),
        "run_status": (
            run_meta.get("status")
            or run_meta.get("state")
            or run_meta.get("run_status")
            or run_meta.get("a2a_task_status")
            or terminal_event_status
            or "unknown"
        ),
        "assistant_messages": assistant_messages,
        "tool_calls": tool_calls,
        "final_message": assistant_messages[-1] if assistant_messages else "",
        "events": events,
        "failure_reason": failure_reason,
    }


def main() -> None:
    st.set_page_config(
        page_title="Healthcare Multi-Agent Coordinator",
        page_icon=":hospital:",
        layout="wide",
    )

    # ---------------------------------------------------------------------------
    # Google Authentication gate — handles both null and expired sessions.
    #
    # NULL SESSION  : user.is_logged_in is False  → show login page, st.stop()
    # EXPIRED SESSION: logged in but idle > SESSION_TIMEOUT_HOURS → st.logout()
    #
    # The gate is a no-op when [auth] is absent or still has REPLACE placeholders
    # so the app works locally without OAuth configured.
    # ---------------------------------------------------------------------------
    _SESSION_TIMEOUT_HOURS = 8  # idle timeout; set to 0 to disable

    try:
        auth_configured = (
            "auth" in st.secrets
            and "google" in st.secrets["auth"]
            and not str(st.secrets["auth"]["cookie_secret"]).startswith("REPLACE")
            and not str(st.secrets["auth"]["google"]["client_id"]).startswith("REPLACE")
        )
    except Exception:
        auth_configured = False

    current_user = None
    if auth_configured:
        user = st.user  # st.experimental_user was removed; st.user is the current API
        current_user = user

        # --- NULL SESSION: not logged in at all ---
        if not user.is_logged_in:
            st.set_page_config(
                page_title="Sign in — Healthcare Coordinator",
                page_icon=":hospital:",
                layout="centered",
            ) if False else None  # page_config already set above; skip duplicate
            st.markdown(
                """
                <div style="display:flex;flex-direction:column;align-items:center;padding-top:4rem;">
                  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/"
                           "Google_2015_logo.svg/640px-Google_2015_logo.svg.png"
                       width="110" style="margin-bottom:1.5rem;" />
                  <h2 style="margin-bottom:0.4rem;">Healthcare Multi-Agent Coordinator</h2>
                  <p style="color:#555;margin-bottom:2rem;text-align:center;max-width:420px;">
                    Sign in with your Google account to access patient care workflows,
                    discharge planning, and risk monitoring.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.login("google")
            st.stop()

        # --- EXPIRED SESSION: logged in but idle too long ---
        now = time.time()
        if _SESSION_TIMEOUT_HOURS > 0:
            last_active = st.session_state.get("_auth_last_active", now)
            idle_hours = (now - last_active) / 3600
            if idle_hours > _SESSION_TIMEOUT_HOURS:
                st.session_state.clear()
                st.warning("Your session has expired. Please sign in again.")
                st.login("google")
                st.logout()
                st.stop()
        st.session_state["_auth_last_active"] = now

        # Authenticated user controls are rendered in the page header (top-right).

    orchestrator, repository = load_orchestrator()
    watsonx_orchestrator = load_watsonx_orchestrator(repository)
    config = AppConfig.from_env()
    patients = repository.patients
    patient_options = {
        f"{record['patient_id']} - {record['name']}": record["patient_id"] for record in patients.values()
    }

    st.title("Healthcare Multi-Agent Care Coordinator")
    st.caption("Interactive demo for discharge translation, symptom monitoring, logistics coordination, and nurse escalation.")

    if auth_configured and current_user is not None and current_user.is_logged_in:
        # Handle sign-out via query param (triggered from the HTML tooltip link)
        if st.query_params.get("signout") == "1":
            st.query_params.clear()
            st.session_state.clear()
            st.logout()

        pic = getattr(current_user, "picture", None) or getattr(current_user, "avatar_url", None)
        name = getattr(current_user, "name", None) or getattr(current_user, "email", "User")
        email = getattr(current_user, "email", "")
        initials = "".join(w[0].upper() for w in (name or "U").split()[:2])

        # Avatar button: photo if available, else colored initials circle
        if pic:
            avatar_btn_html = f'<img src="{pic}" class="hc-avatar-img" alt="{initials}" />'
            avatar_large_html = f'<img src="{pic}" class="hc-drop-photo" alt="{initials}" />'
        else:
            avatar_btn_html = f'<span class="hc-avatar-init">{initials}</span>'
            avatar_large_html = (
                f'<div class="hc-drop-initials">{initials}</div>'
            )

        st.markdown(
            f"""
            <style>
            /* ── Profile widget: fixed in top toolbar, left of Deploy button ─── */
            .hc-profile-wrap {{
                position: fixed;
                top: 0.42rem;
                right: 9.2rem;
                z-index: 99999999;
            }}
            .hc-avatar-btn {{
                width: 36px;
                height: 36px;
                border-radius: 50%;
                overflow: hidden;
                cursor: pointer;
                border: 2px solid #d0d5dd;
                background: #4285F4;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .hc-avatar-img {{
                width: 36px;
                height: 36px;
                border-radius: 50%;
                object-fit: cover;
                display: block;
            }}
            .hc-avatar-init {{
                color: white;
                font-weight: 700;
                font-size: 0.85rem;
                font-family: sans-serif;
                line-height: 1;
            }}
            /* ── Hover dropdown card ────────────────────────────────────────── */
            .hc-profile-dropdown {{
                display: none;
                position: absolute;
                top: 44px;
                right: 0;
                background: #ffffff;
                border: 1px solid #e0e4ea;
                border-radius: 14px;
                box-shadow: 0 10px 32px rgba(0,0,0,0.14);
                padding: 1.2rem 1.4rem 1rem;
                min-width: 230px;
                text-align: center;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            }}
            .hc-profile-wrap:hover .hc-profile-dropdown {{
                display: block;
            }}
            .hc-drop-photo {{
                width: 60px;
                height: 60px;
                border-radius: 50%;
                object-fit: cover;
                margin-bottom: 0.6rem;
                border: 2px solid #e8eaed;
            }}
            .hc-drop-initials {{
                width: 60px;
                height: 60px;
                border-radius: 50%;
                background: #4285F4;
                color: white;
                font-weight: 700;
                font-size: 1.3rem;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 0.6rem;
            }}
            .hc-drop-name {{
                font-weight: 700;
                font-size: 0.95rem;
                color: #1a1a2e;
                margin-bottom: 0.25rem;
            }}
            .hc-drop-email {{
                font-size: 0.8rem;
                color: #5b6472;
                margin-bottom: 0.8rem;
                word-break: break-all;
            }}
            .hc-drop-divider {{
                border: none;
                border-top: 1px solid #eeeff2;
                margin: 0.5rem 0 0.7rem;
            }}
            .hc-signout-link {{
                display: inline-block;
                padding: 0.38rem 1.4rem;
                border: 1px solid #d0d3d9;
                border-radius: 7px;
                font-size: 0.84rem;
                color: #333;
                text-decoration: none;
                background: #fafafa;
            }}
            .hc-signout-link:hover {{
                background: #f0f0f0;
                color: #111;
                text-decoration: none;
            }}
            </style>

            <div class="hc-profile-wrap">
              <div class="hc-avatar-btn" title="Account">
                {avatar_btn_html}
              </div>
              <div class="hc-profile-dropdown">
                {avatar_large_html}
                <div class="hc-drop-name">{name}</div>
                <div class="hc-drop-email">{email}</div>
                <hr class="hc-drop-divider" />
                <a href="?signout=1" class="hc-signout-link">Sign out</a>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.sidebar:
        st.subheader("Case Setup")
        selected_label = st.selectbox("Patient", list(patient_options.keys()))
        patient_id = patient_options[selected_label]
        patient = patients[patient_id]
        discharge_plan = repository.discharge_plans[patient_id]
        default_report = {
            "PT-1001": "A little tired after walking, but no fever and breathing is normal.",
            "PT-2002": "My leg is more swollen and I missed my antibiotic pickup.",
            "PT-1002": "I feel short of breath and dizzy this morning.",
            "PT-1003": "My knee wound looks clean and the pain is manageable, but I feel a little stiff when walking.",
        }.get(patient_id, "")

        symptom_report = st.text_area(
            "Symptom report",
            value=default_report,
            height=140,
            help="Describe what the patient or caregiver is reporting today.",
        )
        execution_mode = st.radio(
            "Execution mode",
            options=["Local deterministic", "Live watsonx tools", "Orchestrate REST API"],
            index=0,
            help="Switch between local deterministic orchestration, direct watsonx tool-calling, or hosted Orchestrate API runs.",
        )
        token_key = "orchestrate_bearer_token"
        env_token = os.environ.get("ORCHESTRATE_BEARER_TOKEN", "")
        if token_key not in st.session_state and env_token:
            st.session_state[token_key] = env_token
        default_token = st.session_state.get(token_key) or env_token or config.orchestrate_bearer_token or ""
        bearer_token = st.text_input(
            "Orchestrate bearer token (optional override)",
            value=default_token,
            type="password",
            help="Used for Orchestrate REST API mode. Stored in this Streamlit session only.",
        )
        if bearer_token:
            st.session_state[token_key] = bearer_token
            os.environ["ORCHESTRATE_BEARER_TOKEN"] = bearer_token
        elif token_key in st.session_state:
            # Allow clearing a previously entered token from session.
            st.session_state.pop(token_key, None)
            os.environ.pop("ORCHESTRATE_BEARER_TOKEN", None)

        active_token = st.session_state.get(token_key) or os.environ.get("ORCHESTRATE_BEARER_TOKEN") or config.orchestrate_bearer_token
        token_source = "none"
        if st.session_state.get(token_key):
            token_source = "session"
        elif os.environ.get("ORCHESTRATE_BEARER_TOKEN"):
            token_source = "env"
        elif config.orchestrate_bearer_token:
            token_source = "config"

        st.caption(f"Token source: `{token_source}`")
        st.caption(f"API key present: `{bool(config.orchestrate_api_key)}`")
        if active_token:
            if len(active_token) > 24:
                preview = f"{active_token[:12]}...{active_token[-8:]}"
            else:
                preview = active_token
            st.caption(f"Token preview: `{preview}`")
            if st.checkbox("Show active token (local debug only)", value=False):
                st.code(active_token, language="text")
        else:
            st.caption("Token preview: `none`")
        submitted = st.button("Run care coordination", type="primary", use_container_width=True)

        st.markdown("### Engine Readiness")
        st.write(f"watsonx tools: {'Ready' if watsonx_orchestrator else 'Missing config'}")
        st.write(f"Orchestrate API: {'Ready' if config.orchestrate_endpoint else 'Missing endpoint'}")

        st.markdown("### Patient Snapshot")
        st.write(f"Literacy level: `{patient['literacy_level']}`")
        st.write(discharge_plan["summary"])
        if watsonx_orchestrator is None:
            missing = ", ".join(config.missing_watsonx_fields())
            st.caption(f"watsonx disabled until these values exist in .env: {missing}")
        if not config.orchestrate_endpoint:
            st.caption("Orchestrate API disabled until ORCHESTRATE_API_ENDPOINT (or ORCHESTRATE_INSTANCE_URL) is set.")
        st.caption("Quick PowerShell launcher for Orchestrate mode:")
        st.code(r".\run_streamlit_orchestrate.local.ps1", language="powershell")

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&display=swap');
        html, body, [class*="css"] {
            font-family: 'Manrope', sans-serif;
        }
        .result-card {
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 18px;
            padding: 1.1rem 1.2rem;
            background: linear-gradient(140deg, rgba(233,245,255,1) 0%, rgba(255,252,244,1) 100%);
        }
        .small-label {
            font-size: 0.8rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #5b6472;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if not submitted:
        st.info("Choose a patient, adjust the symptom report if needed, and run the workflow to see the agents coordinate.")
        return

    result = orchestrator.resolve(patient_id, symptom_report)
    selected_summary = result.recovery_summary
    selected_mode_badge = execution_mode
    watsonx_result = None
    watsonx_error = None
    orchestrate_payload = None
    orchestrate_error = None
    parsed_orchestrate: dict[str, Any] | None = None

    if execution_mode == "Live watsonx tools":
        if watsonx_orchestrator is None:
            watsonx_error = "watsonx credentials are missing. Add WATSONX_APIKEY and WATSONX_PROJECT_ID to .env."
        else:
            try:
                watsonx_result = watsonx_orchestrator.resolve(patient_id, symptom_report)
                selected_summary = watsonx_result.final_response
            except Exception as exc:  # pragma: no cover
                watsonx_error = str(exc)

    if execution_mode == "Orchestrate REST API":
        prompt = build_orchestrate_prompt(patient_id, symptom_report)
        try:
            run_config = replace(
                config,
                orchestrate_bearer_token=st.session_state.get("orchestrate_bearer_token", config.orchestrate_bearer_token),
            )
            with st.spinner("Running Orchestrate workflow..."):
                orchestrate_payload = invoke_via_api(run_config, prompt, poll_timeout_sec=45)
            refreshed_token = os.environ.get("ORCHESTRATE_BEARER_TOKEN")
            if refreshed_token and not st.session_state.get("orchestrate_bearer_token"):
                st.session_state["orchestrate_bearer_token"] = refreshed_token
            parsed_orchestrate = parse_orchestrate_payload(orchestrate_payload)
            run_status = str(parsed_orchestrate.get("run_status", "unknown")).lower()
            if run_status in {"failed", "cancelled"}:
                reason = parsed_orchestrate.get("failure_reason") or f"Orchestrate run status is {run_status}."
                orchestrate_error = str(reason)
                selected_mode_badge = "Orchestrate REST API (fallback: local deterministic)"
            elif parsed_orchestrate.get("final_message"):
                selected_summary = clean_orchestrate_summary(str(parsed_orchestrate["final_message"])) or selected_summary
            else:
                selected_mode_badge = "Orchestrate REST API (no assistant message, fallback: local deterministic)"
        except Exception as exc:  # pragma: no cover
            orchestrate_error = str(exc)
            selected_mode_badge = "Orchestrate REST API (fallback: local deterministic)"

    monitoring_output = result.details["monitoring_output"]
    logistics_output = result.details["logistics_output"]
    translator_output = result.details["translator_output"]
    risk_label, risk_style = severity_label(result.risk_status)

    overview_col, status_col, actions_col = st.columns((1.5, 1, 1))

    with overview_col:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-label">Recovery Summary</div>', unsafe_allow_html=True)
        st.subheader(selected_summary)
        st.caption(f"Execution mode: {selected_mode_badge}")
        st.write(f"Patient ID: `{result.patient_id}`")
        st.write(f"Symptom report: {symptom_report}")
        st.markdown("</div>", unsafe_allow_html=True)

    with status_col:
        st.metric("Risk Status", risk_label)
        if risk_style == "error":
            st.error("Human nurse escalation required.")
        elif risk_style == "warning":
            st.warning("Close observation recommended today.")
        else:
            st.success("Recovery appears stable.")

    with actions_col:
        st.metric("Checklist Items", len(result.checklist))
        st.metric("Coordination Actions", len(result.coordination_status))
        st.metric("Escalated", "Yes" if result.escalated else "No")

    checklist_col, logistics_col = st.columns(2)

    with checklist_col:
        st.subheader("Daily Checklist")
        for item in translator_output["checklist"]:
            st.checkbox(item, value=False, disabled=True)

        st.subheader("Medication Schedule")
        for item in translator_output["medication_schedule"]:
            st.write(f"- {item}")

        st.subheader("Safety Tips")
        for item in translator_output["safety_tips"]:
            st.write(f"- {item}")

    with logistics_col:
        st.subheader("Risk Review")
        st.write(f"Triage level: `{monitoring_output['triage_level']}`")
        st.write(monitoring_output["recommended_action"])
        if monitoring_output["concerning_signals"]:
            for signal in monitoring_output["concerning_signals"]:
                st.write(f"- {signal}")
        else:
            st.write("- No concerning signals were detected.")

        st.subheader("Logistics Coordination")
        for line in logistics_output["medication_status"]:
            st.write(f"- {line}")
        for line in logistics_output["appointment_status"]:
            st.write(f"- {line}")
        for line in logistics_output["resolved_actions"]:
            st.write(f"- {line}")

    with st.expander("Transparent reasoning log", expanded=True):
        for entry in result.reasoning_log:
            st.write(f"- {entry}")

    if watsonx_result:
        with st.expander("watsonx Tool Trace", expanded=True):
            for call in watsonx_result.tool_trace:
                label = call.get("tool") or call.get("tool_name") or call.get("name", "unknown")
                st.write(f"Tool: `{label}`")
                st.json(call)
    elif watsonx_error:
        st.error(f"watsonx invocation failed: {watsonx_error}")

    if orchestrate_payload:
        parsed = parsed_orchestrate or parse_orchestrate_payload(orchestrate_payload)
        with st.expander("Orchestrate Run Trace", expanded=True):
            st.write(f"Run ID: `{parsed['run_id']}`")
            st.write(f"Run status: `{parsed['run_status']}`")
            if parsed["assistant_messages"]:
                st.write("Assistant responses:")
                for message in parsed["assistant_messages"]:
                    st.write(f"- {message}")
            if parsed["tool_calls"]:
                st.write("Tool calls:")
                for call in parsed["tool_calls"]:
                    st.json(call)
            st.write("Raw payload:")
            st.json(orchestrate_payload)
    if orchestrate_error:
        st.error(f"Orchestrate API invocation failed: {orchestrate_error}")

    with st.expander("Structured payload"):
        st.json(asdict(result))


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
DEFAULT_WATSONX_MODEL = "watsonx/ibm/granite-3-8b-instruct"
DEFAULT_WATSONX_VERSION = "2024-10-10"
DEFAULT_IAM_URL = "https://iam.cloud.ibm.com/identity/token"
DEFAULT_MCSP_TOKEN_URL = "https://iam.platform.saas.ibm.com/siusermgr/api/1.0/apikeys/token"


def _load_local_env() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _read_setting(key: str, default: str | None = None) -> str | None:
    env_value = os.getenv(key)
    if env_value is not None:
        return env_value

    # Streamlit Community Cloud stores app secrets in st.secrets.
    try:
        import streamlit as st  # type: ignore

        if key in st.secrets:
            value = st.secrets.get(key)
            if value is None:
                return default
            return str(value)
    except Exception:
        pass
    return default


@dataclass(slots=True)
class AppConfig:
    watsonx_apikey: str | None
    watsonx_project_id: str | None
    watsonx_url: str
    watsonx_model: str
    serper_api_key: str | None
    orchestrate_instance_url: str | None = None
    orchestrate_api_endpoint: str | None = None
    orchestrate_api_key: str | None = None
    orchestrate_bearer_token: str | None = None
    orchestrate_env_name: str = "healthcare-dev"
    orchestrate_agent_name: str = "Healthcare_Care_Coordinator"
    orchestrate_agent_id: str | None = None
    orchestrate_auth_type: str = "ibm_iam"
    orchestrate_iam_url: str = DEFAULT_IAM_URL
    orchestrate_mcsp_token_url: str = DEFAULT_MCSP_TOKEN_URL
    watsonx_version: str = DEFAULT_WATSONX_VERSION
    iam_url: str = DEFAULT_IAM_URL

    @classmethod
    def from_env(cls) -> "AppConfig":
        _load_local_env()
        return cls(
            watsonx_apikey=_read_setting("WATSONX_APIKEY"),
            watsonx_project_id=_read_setting("WATSONX_PROJECT_ID"),
            watsonx_url=_read_setting("WATSONX_URL", DEFAULT_WATSONX_URL) or DEFAULT_WATSONX_URL,
            watsonx_model=_read_setting("WATSONX_MODEL", DEFAULT_WATSONX_MODEL) or DEFAULT_WATSONX_MODEL,
            serper_api_key=_read_setting("SERPER_API_KEY"),
            orchestrate_instance_url=_read_setting("ORCHESTRATE_INSTANCE_URL"),
            orchestrate_api_endpoint=_read_setting("ORCHESTRATE_API_ENDPOINT"),
            orchestrate_api_key=_read_setting("ORCHESTRATE_API_KEY"),
            orchestrate_bearer_token=_read_setting("ORCHESTRATE_BEARER_TOKEN"),
            orchestrate_env_name=_read_setting("ORCHESTRATE_ENV_NAME", "healthcare-dev") or "healthcare-dev",
            orchestrate_agent_name=_read_setting("ORCHESTRATE_AGENT_NAME", "Healthcare_Care_Coordinator")
            or "Healthcare_Care_Coordinator",
            orchestrate_agent_id=_read_setting("ORCHESTRATE_AGENT_ID"),
            orchestrate_auth_type=_read_setting("ORCHESTRATE_AUTH_TYPE", "ibm_iam") or "ibm_iam",
            orchestrate_iam_url=_read_setting("ORCHESTRATE_IAM_URL", DEFAULT_IAM_URL) or DEFAULT_IAM_URL,
            orchestrate_mcsp_token_url=_read_setting("ORCHESTRATE_MCSP_TOKEN_URL", DEFAULT_MCSP_TOKEN_URL)
            or DEFAULT_MCSP_TOKEN_URL,
        )

    @property
    def normalized_model(self) -> str:
        if self.watsonx_model.startswith("watsonx/"):
            return self.watsonx_model.split("/", 1)[1]
        return self.watsonx_model

    @property
    def watsonx_ready(self) -> bool:
        return bool(self.watsonx_apikey and self.watsonx_project_id and self.watsonx_url and self.watsonx_model)

    @property
    def serper_ready(self) -> bool:
        return bool(self.serper_api_key)

    def missing_watsonx_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.watsonx_apikey:
            missing.append("WATSONX_APIKEY")
        if not self.watsonx_project_id:
            missing.append("WATSONX_PROJECT_ID")
        if not self.watsonx_url:
            missing.append("WATSONX_URL")
        if not self.watsonx_model:
            missing.append("WATSONX_MODEL")
        return missing

    @property
    def orchestrate_endpoint(self) -> str | None:
        return self.orchestrate_api_endpoint or self.orchestrate_instance_url

    @property
    def orchestrate_api_ready(self) -> bool:
        return bool(self.orchestrate_endpoint and (self.orchestrate_bearer_token or self.orchestrate_api_key))

    def missing_orchestrate_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.orchestrate_endpoint:
            missing.append("ORCHESTRATE_API_ENDPOINT (or ORCHESTRATE_INSTANCE_URL)")
        if not self.orchestrate_bearer_token and not self.orchestrate_api_key:
            missing.append("ORCHESTRATE_BEARER_TOKEN (or ORCHESTRATE_API_KEY)")
        return missing

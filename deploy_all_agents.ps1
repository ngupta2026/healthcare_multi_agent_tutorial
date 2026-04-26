# deploy_all_agents.ps1
# ---------------------------------------------------------------------------
# Deploys all healthcare agents to IBM Orchestrate in the correct order:
#   1. Register Python tools (shared by all agents)
#   2. Import 4 specialist agents (no inter-agent dependencies)
#   3. Import the supervisor coordinator last (depends on collaborator names)
# ---------------------------------------------------------------------------
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\deploy_all_agents.ps1
#   powershell -ExecutionPolicy Bypass -File .\deploy_all_agents.ps1 -SkipToolImport
# ---------------------------------------------------------------------------

param(
    [string]$EnvName = "healthcare-iam",
    [switch]$SkipToolImport
)

$ErrorActionPreference = "Stop"

$REPO_ROOT = $PSScriptRoot
$VENV_PY = Join-Path $REPO_ROOT "src\.venv\Scripts\python.exe"
$VENV_ORC = Join-Path $REPO_ROOT "src\.venv\Scripts\orchestrate.exe"
$DEPLOY_DIR = Join-Path $REPO_ROOT "deploy\orchestrate"
$TOOL_FILE = Join-Path $REPO_ROOT "src\healthcare_support_agents\orchestrate_adk_tools.py"
$REQUIREMENTS = Join-Path $REPO_ROOT "requirements-orchestrate.txt"
$PACKAGE_ROOT = Join-Path $REPO_ROOT "src"
# Staging root: only the healthcare_support_agents package, no .venv (avoids 300 MB upload)
$STAGING_ROOT = Join-Path $env:TEMP "orchestrate_hca_staging"

# Specialist agent YAMLs - deployed BEFORE the coordinator
$SPECIALIST_AGENTS = @(
    "patient_context_agent.agent.yaml",
    "discharge_translator_agent.agent.yaml",
    "recovery_monitoring_agent.agent.yaml",
    "care_logistics_agent.agent.yaml"
)

# Supervisor coordinator YAML - deployed LAST
$COORDINATOR_AGENT = "healthcare_care_coordinator.agent.yaml"

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
function Print-Step($msg) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
}

function Bail($msg) {
    Write-Host ""
    Write-Host "ERROR: $msg" -ForegroundColor Red
    exit 1
}

function Get-DotEnvValue([string]$Key) {
    $envFile = Join-Path $REPO_ROOT ".env"
    if (-not (Test-Path $envFile)) {
        return $null
    }
    $line = Get-Content $envFile | Select-String -Pattern "^\s*$Key\s*=\s*(.+)\s*$" | Select-Object -First 1
    if (-not $line) {
        return $null
    }
    return $line.Matches[0].Groups[1].Value.Trim().Trim('"').Trim("'")
}

function Invoke-Orchestrate {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Args
    )

    & $VENV_ORC @Args
    if ($LASTEXITCODE -ne 0) {
        if ($Args.Length -ge 2 -and $Args[0] -eq "env" -and $Args[1] -eq "activate") {
            Bail "Failed to activate Orchestrate environment. Run '.\\src\\.venv\\Scripts\\orchestrate.exe env list' and activate a valid environment."
        }
        Bail "Orchestrate command failed (exit=$LASTEXITCODE): orchestrate $($Args -join ' ')"
    }
}

function Ensure-OrchestrateEnvironment {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $listOutput = & $VENV_ORC env list 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        Bail "Unable to list Orchestrate environments. Ensure CLI authentication is completed."
    }

    $exists = $false
    foreach ($line in ($listOutput -split "`r?`n")) {
        if ($line -match "^\s*([A-Za-z0-9_.-]+)\s+https?://") {
            if ($matches[1] -eq $Name) {
                $exists = $true
                break
            }
        }
    }

    if ($exists) {
        return
    }

    $endpoint = Get-DotEnvValue "ORCHESTRATE_API_ENDPOINT"
    if (-not $endpoint) {
        $endpoint = Get-DotEnvValue "ORCHESTRATE_INSTANCE_URL"
    }
    if (-not $endpoint) {
        Bail "Environment '$Name' is missing and ORCHESTRATE_API_ENDPOINT is not set in .env. Add the environment manually with: orchestrate env add -n $Name -u <instance_url> -t ibm_iam"
    }

    $authType = Get-DotEnvValue "ORCHESTRATE_AUTH_TYPE"
    if (-not $authType) {
        $authType = "ibm_iam"
    }
    if ($authType -eq "iam") {
        $authType = "ibm_iam"
    }

    Write-Host "  env          : '$Name' not found. Creating it from .env..." -ForegroundColor Yellow
    Invoke-Orchestrate @("env", "add", "-n", $Name, "-u", $endpoint, "-t", $authType)
}

function Assert-RequiredToolsPresent {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$ToolNames
    )

    $toolsJson = & $VENV_ORC tools list -v 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        Bail "Unable to list Orchestrate tools in the active environment."
    }

    $parsed = $toolsJson | ConvertFrom-Json
    $existing = @{}
    foreach ($tool in $parsed) {
        if ($null -ne $tool.name) {
            $existing[$tool.name] = $true
        }
    }

    $missing = @()
    foreach ($name in $ToolNames) {
        if (-not $existing.ContainsKey($name)) {
            $missing += $name
        }
    }

    if ($missing.Count -gt 0) {
        Bail "SkipToolImport cannot continue because the active cloud environment is missing required tools: $($missing -join ', '). Re-run without -SkipToolImport after IBM resolves the tool upload HTTP 500 issue."
    }
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
Print-Step "Pre-flight checks"

if (-not (Test-Path $VENV_PY)) {
    Bail "Python venv not found at $VENV_PY. Run:  py -3.13 -m venv src\.venv"
}
Write-Host "  Python venv  : OK ($VENV_PY)" -ForegroundColor Green

if (-not (Test-Path $VENV_ORC)) {
    Bail "orchestrate CLI not found. Run:  .\src\.venv\Scripts\pip.exe install ibm-watsonx-orchestrate"
}
Write-Host "  orchestrate  : OK ($VENV_ORC)" -ForegroundColor Green

if (-not (Test-Path $TOOL_FILE)) {
    Bail "Tool file not found: $TOOL_FILE"
}
Write-Host "  Tool file    : OK" -ForegroundColor Green

if (-not (Test-Path $REQUIREMENTS)) {
    Bail "Requirements file not found: $REQUIREMENTS"
}
Write-Host "  Requirements : OK" -ForegroundColor Green

foreach ($yaml in $SPECIALIST_AGENTS + $COORDINATOR_AGENT) {
    $full = Join-Path $DEPLOY_DIR $yaml
    if (-not (Test-Path $full)) {
        Bail "Agent YAML not found: $full"
    }
    Write-Host "  YAML         : OK ($yaml)" -ForegroundColor Green
}

Ensure-OrchestrateEnvironment -Name $EnvName
Invoke-Orchestrate @("env", "activate", $EnvName)
Write-Host "  env          : active ($EnvName)" -ForegroundColor Green

# ---------------------------------------------------------------------------
# Step 1 - Register Python tools
# ---------------------------------------------------------------------------
if ($SkipToolImport) {
    Print-Step "Step 1/6 - Skipping Python tool import"
    Write-Host "  SkipToolImport was specified." -ForegroundColor Yellow
    Write-Host "  Agent YAMLs will still be imported/updated, but cloud tool code will not be refreshed." -ForegroundColor Yellow
    Write-Host "  Use this mode while IBM Orchestrate tool uploads are returning HTTP 500." -ForegroundColor Yellow
    Assert-RequiredToolsPresent @(
        "get_patient_snapshot",
        "translate_discharge_plan",
        "monitor_recovery_status",
        "coordinate_care_logistics",
        "resolve_recovery_case",
        "search_support_services"
    )
}
else {
    Print-Step "Step 1/6 - Register Python tools with IBM Orchestrate"
    Write-Host "  Importing tools from: $TOOL_FILE"
    Write-Host "  This registers get_patient_snapshot, translate_discharge_plan,"
    Write-Host "  monitor_recovery_status, coordinate_care_logistics,"
    Write-Host "  resolve_recovery_case, and search_support_services."
    Write-Host ""

    # Bug 3 Fix (Part A) — IBM HTTP 500 on UPDATE/upload endpoint.
    # The original script had no pre-clean step. If any of the 6 tools already existed
    # in the cloud, the CLI found them and issued a PUT to the /upload endpoint to update
    # them. IBM's /upload endpoint returns HTTP 500 on every UPDATE operation.
    # The CREATE path (first-time import) works correctly.
    # Fix: remove all 6 tools first so every subsequent import hits the CREATE path.
    $TOOL_NAMES = @(
        "get_patient_snapshot",
        "translate_discharge_plan",
        "monitor_recovery_status",
        "coordinate_care_logistics",
        "resolve_recovery_case",
        "search_support_services"
    )
    Write-Host "  Pre-cleaning existing tools to force fresh CREATE (avoids IBM 500 on UPDATE)..." -ForegroundColor Yellow
    foreach ($toolName in $TOOL_NAMES) {
        $result = & $VENV_ORC tools remove -n $toolName 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "    Removed: $toolName" -ForegroundColor DarkGray
        }
        # Silently ignore if tool didn't exist
    }
    Write-Host "  Pre-clean complete." -ForegroundColor Yellow
    Write-Host ""

    # Bug 3 Fix (Part B) — Oversized upload zip caused IBM HTTP 500.
    # The original command used '--package-root .\src', which caused the CLI to zip the
    # entire src\ directory including src\.venv\ (~300 MB of installed packages).
    # IBM's upload endpoint has a size limit; a 300 MB zip triggers HTTP 500 before
    # any tool code is even inspected.
    # Fix: copy only the healthcare_support_agents\ package (~282 KB) to a temp staging
    # directory and use that as --package-root instead. The .venv is excluded entirely.
    Write-Host "  Building lean upload staging area (no .venv)..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $STAGING_ROOT
    New-Item -ItemType Directory -Path $STAGING_ROOT | Out-Null
    Copy-Item -Recurse -Force (Join-Path $REPO_ROOT "src\healthcare_support_agents") `
    (Join-Path $STAGING_ROOT "healthcare_support_agents")
    $STAGING_TOOL_FILE = Join-Path $STAGING_ROOT "healthcare_support_agents\orchestrate_adk_tools.py"
    Write-Host "  Staging ready: $STAGING_ROOT (~$(([math]::Round((Get-ChildItem $STAGING_ROOT -Recurse -File | Measure-Object -Property Length -Sum).Sum/1KB, 1))) KB)" -ForegroundColor Yellow
    Write-Host ""

    Invoke-Orchestrate @("tools", "import", "--kind", "python", "--file", $STAGING_TOOL_FILE, "--package-root", $STAGING_ROOT)
    Write-Host "  Tools registered successfully." -ForegroundColor Green

    # Cleanup staging
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $STAGING_ROOT

    Invoke-Orchestrate @("tools", "list", "-v")
}

# ---------------------------------------------------------------------------
# Steps 2-5 - Import 4 specialist agents
# ---------------------------------------------------------------------------
$step = 2
foreach ($yaml in $SPECIALIST_AGENTS) {
    $agentName = $yaml -replace "\.agent\.yaml$", "" -replace "_", " "
    Print-Step "Step $step/6 - Import specialist agent: $agentName"

    $fullPath = Join-Path $DEPLOY_DIR $yaml
    Write-Host "  YAML: $fullPath"
    Write-Host ""

    Invoke-Orchestrate @("agents", "import", "-f", $fullPath)
    Write-Host "  Agent imported successfully." -ForegroundColor Green

    $step++
}

# ---------------------------------------------------------------------------
# Step 6 - Import coordinator (must come last)
# ---------------------------------------------------------------------------
Print-Step "Step 6/6 - Import supervisor coordinator (Healthcare_Care_Coordinator)"
Write-Host "  This agent lists all 4 specialists as collaborators."
Write-Host "  IBM Orchestrate resolves collaborators by name - they must exist first."
Write-Host ""

$coordinatorPath = Join-Path $DEPLOY_DIR $COORDINATOR_AGENT
Invoke-Orchestrate @("agents", "import", "-f", $coordinatorPath)
Write-Host "  Coordinator imported successfully." -ForegroundColor Green

# ---------------------------------------------------------------------------
# Final - List all agents
# ---------------------------------------------------------------------------
Print-Step "All agents deployed - listing registered agents"
Invoke-Orchestrate @("agents", "list", "-v")

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Deployment complete!" -ForegroundColor Green
Write-Host "  5 agents are now live in IBM Orchestrate:" -ForegroundColor Green
Write-Host "    - Patient_Context_Agent" -ForegroundColor Green
Write-Host "    - Discharge_Translator_Agent" -ForegroundColor Green
Write-Host "    - Recovery_Monitoring_Agent" -ForegroundColor Green
Write-Host "    - Care_Logistics_Agent" -ForegroundColor Green
Write-Host "    - Healthcare_Care_Coordinator (supervisor)" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green


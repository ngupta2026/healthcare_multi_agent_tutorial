param(
    [string]$EnvName = "local"
)

function Get-DotEnvValue([string]$Key) {
    $envFile = Join-Path $PSScriptRoot ".env"
    if (-not (Test-Path $envFile)) {
        return $null
    }
    $line = Get-Content $envFile | Select-String -Pattern "^\s*$Key\s*=\s*(.+)\s*$" | Select-Object -First 1
    if (-not $line) {
        return $null
    }
    return $line.Matches[0].Groups[1].Value.Trim().Trim('"').Trim("'")
}

$authType = $env:ORCHESTRATE_AUTH_TYPE
if (-not $authType) {
    $authType = Get-DotEnvValue "ORCHESTRATE_AUTH_TYPE"
}
if (-not $authType) {
    $authType = "ibm_iam"
}

if ($authType -eq "mcsp") {
    try {
        orchestrate env activate $EnvName | Out-Null
        Write-Host "Activated orchestrate environment: $EnvName"
    }
    catch {
        Write-Warning "Could not activate orchestrate environment '$EnvName'. Continuing with existing auth settings."
    }
}

if ($authType -ne "mcsp") {
    Write-Host "Using ORCHESTRATE_AUTH_TYPE=$authType. Skipping MCSP token cache lookup."
    $env:PYTHONPATH = "src"
    streamlit run src/healthcare_support_agents/streamlit_app.py
    exit $LASTEXITCODE
}

$credentialsPath = "$HOME\.cache\orchestrate\credentials.yaml"

if (-not (Test-Path $credentialsPath)) {
    Write-Warning "Credentials cache not found at $credentialsPath. Continuing without ORCHESTRATE_BEARER_TOKEN."
    $env:PYTHONPATH = "src"
    streamlit run src/healthcare_support_agents/streamlit_app.py
    exit $LASTEXITCODE
}

$tokenMatch = Get-Content $credentialsPath | Select-String -Pattern 'wxo_mcsp_token:\s*(.+)$' | Select-Object -First 1

if (-not $tokenMatch) {
    Write-Warning "No wxo_mcsp_token found in $credentialsPath. Falling back to ORCHESTRATE_API_KEY / ORCHESTRATE_BEARER_TOKEN from environment."
    $env:PYTHONPATH = "src"
    streamlit run src/healthcare_support_agents/streamlit_app.py
    exit $LASTEXITCODE
}

$token = $tokenMatch.Matches[0].Groups[1].Value.Trim()
if (-not $token) {
    Write-Warning "Resolved wxo_mcsp_token is empty in $credentialsPath. Falling back to ORCHESTRATE_API_KEY / ORCHESTRATE_BEARER_TOKEN from environment."
    $env:PYTHONPATH = "src"
    streamlit run src/healthcare_support_agents/streamlit_app.py
    exit $LASTEXITCODE
}

$env:ORCHESTRATE_BEARER_TOKEN = $token
$env:PYTHONPATH = "src"

streamlit run src/healthcare_support_agents/streamlit_app.py

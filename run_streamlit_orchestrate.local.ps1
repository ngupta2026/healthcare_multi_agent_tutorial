param(
    [string]$EnvName = "healthcare-aws"
)

orchestrate env activate $EnvName

$credentialsPath = "$HOME\.cache\orchestrate\credentials.yaml"
$tokenMatch = Get-Content $credentialsPath | Select-String -Pattern 'wxo_mcsp_token:\s*(.+)$' | Select-Object -First 1

if (-not $tokenMatch) {
    throw "Could not find wxo_mcsp_token in $credentialsPath"
}

$token = $tokenMatch.Matches[0].Groups[1].Value.Trim()
if (-not $token) {
    throw "Resolved token is empty in $credentialsPath"
}

$env:ORCHESTRATE_BEARER_TOKEN = $token
$env:PYTHONPATH = "src"

streamlit run src/healthcare_support_agents/streamlit_app.py

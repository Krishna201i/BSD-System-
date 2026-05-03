param(
    [string]$DatasetPath = "",
    [string]$HostUrl = "127.0.0.1",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendDir = Join-Path $repoRoot "backend"
$dataDir = Join-Path $backendDir "data"
$modelsDir = Join-Path $backendDir "models"
$venvDir = Join-Path $repoRoot ".venv"
$pythonExe = Join-Path $venvDir "Scripts\python.exe"
$datasetTarget = Join-Path $dataDir "healthcare-dataset-stroke-data.csv"
$modelTarget = Join-Path $modelsDir "stroke_xgboost.pkl"
$requirementsPath = Join-Path $backendDir "requirements.txt"

if (-not (Test-Path $backendDir)) {
    throw "backend folder not found at $backendDir"
}

New-Item -ItemType Directory -Path $dataDir -Force | Out-Null
New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null

if (-not (Test-Path $venvDir)) {
    Write-Host "Creating virtual environment..."
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $pyLauncher) {
        & py -3 -m venv $venvDir
    } else {
        & python -m venv $venvDir
    }
}

if (-not (Test-Path $pythonExe)) {
    throw "Python executable not found in virtual environment: $pythonExe"
}

Write-Host "Upgrading pip..."
& $pythonExe -m pip install --upgrade pip 2>&1 | Out-Null

Write-Host "Installing requirements..."
& $pythonExe -m pip install -r $requirementsPath 2>&1 | Out-Null

if (Test-Path $datasetTarget) {
    Write-Host "Dataset already exists at $datasetTarget"
} elseif ($DatasetPath) {
    if (-not (Test-Path $DatasetPath)) {
        throw "Provided -DatasetPath does not exist: $DatasetPath"
    }
    Copy-Item -Path $DatasetPath -Destination $datasetTarget -Force
    Write-Host "Copied dataset to $datasetTarget"
} else {
    Write-Host ""
    Write-Host "==============================================================="
    Write-Host "Dataset not found. Manual download required."
    Write-Host "1. Go to: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset"
    Write-Host "2. Click Download (need free Kaggle account)"
    Write-Host "3. Extract and copy healthcare-dataset-stroke-data.csv"
    Write-Host "   to: $dataDir"
    Write-Host "4. Run again: .\setup_windows_v2.ps1"
    Write-Host "OR provide local CSV path:"
    Write-Host ".\setup_windows_v2.ps1 -DatasetPath C:\path\to\healthcare-dataset-stroke-data.csv"
    Write-Host "==============================================================="
    Write-Host ""
    throw "Dataset file required at: $datasetTarget"
}

Write-Host "Training model..."
& $pythonExe (Join-Path $backendDir "train_model.py")

if (-not (Test-Path $modelTarget)) {
    throw "Training finished but model not found at $modelTarget"
}

Write-Host "Model saved: $modelTarget"
Write-Host ""

$serverScript = @"
import uvicorn
uvicorn.run("app.main:app", app_dir="backend", host="$HostUrl", port=$Port, reload=False)
"@
Start-Process -FilePath $pythonExe -ArgumentList "-c", $serverScript -WorkingDirectory $repoRoot | Out-Null

$appUrl = "http://$HostUrl`:$Port"
Write-Host "Starting server at $appUrl ..."

$healthUrl = "$appUrl/health"
$maxAttempts = 20
$attempt = 0
$isUp = $false
while ($attempt -lt $maxAttempts) {
    $attempt++
    Start-Sleep -Milliseconds 700
    try {
        $resp = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($resp.StatusCode -eq 200) {
            $isUp = $true
            break
        }
    } catch {
        continue
    }
}

if (-not $isUp) {
    throw "Server did not become healthy at $healthUrl. Check if port $Port is already in use."
}

Start-Process $appUrl | Out-Null
Write-Host ""
Write-Host "Browser opened to $appUrl"
Write-Host "Dashboard: $appUrl/dashboard"
Write-Host ""

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

function Invoke-CheckedNative {
    param(
        [Parameter(Mandatory = $true)][string]$Description,
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $false)][string[]]$Arguments = @()
    )

    Write-Host $Description
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed (exit code: $LASTEXITCODE)."
    }
}

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

Invoke-CheckedNative -Description "Upgrading pip..." -FilePath $pythonExe -Arguments @("-m", "pip", "install", "--upgrade", "pip")
Invoke-CheckedNative -Description "Installing requirements..." -FilePath $pythonExe -Arguments @("-m", "pip", "install", "-r", $requirementsPath)
Invoke-CheckedNative -Description "Installing Kaggle CLI..." -FilePath $pythonExe -Arguments @("-m", "pip", "install", "kaggle")

if (Test-Path $datasetTarget) {
    Write-Host "Dataset already exists at $datasetTarget"
} elseif ($DatasetPath) {
    if (-not (Test-Path $DatasetPath)) {
        throw "Provided -DatasetPath does not exist: $DatasetPath"
    }
    Copy-Item -Path $DatasetPath -Destination $datasetTarget -Force
    Write-Host "Copied dataset to $datasetTarget"
} else {
    $kaggleConfig = Join-Path $env:USERPROFILE ".kaggle\kaggle.json"
    if (-not (Test-Path $kaggleConfig)) {
        throw "Kaggle token missing. Place kaggle.json at $kaggleConfig or pass -DatasetPath."
    }

    $datasetSlug = "fedesoriano/stroke-prediction-dataset"
    $zipPath = Join-Path $dataDir "stroke-prediction-dataset.zip"
    $directCsvPath = Join-Path $dataDir "healthcare-dataset-stroke-data.csv"

    Invoke-CheckedNative `
        -Description "Downloading stroke dataset CSV from Kaggle..." `
        -FilePath $pythonExe `
        -Arguments @("-m", "kaggle", "datasets", "download", "-d", $datasetSlug, "-f", "healthcare-dataset-stroke-data.csv", "-p", $dataDir, "--force")

    if (Test-Path $zipPath) {
        Expand-Archive -Path $zipPath -DestinationPath $dataDir -Force
    }

    if (-not (Test-Path $directCsvPath)) {
        Invoke-CheckedNative `
            -Description "Retrying with full dataset unzip..." `
            -FilePath $pythonExe `
            -Arguments @("-m", "kaggle", "datasets", "download", "-d", $datasetSlug, "-p", $dataDir, "--unzip", "--force")
    }

    if (-not (Test-Path $datasetTarget)) {
        $csvCandidates = Get-ChildItem -Path $dataDir -Filter *.csv -File -Recurse
        if ($csvCandidates.Count -eq 0) {
            throw "No CSV found in $dataDir after Kaggle download. Verify kaggle.json credentials and dataset access."
        }

        $preferred = $csvCandidates | Where-Object { $_.Name -ieq "healthcare-dataset-stroke-data.csv" } | Select-Object -First 1
        if ($null -eq $preferred) {
            $preferred = $csvCandidates | Select-Object -First 1
        }
        Copy-Item -Path $preferred.FullName -Destination $datasetTarget -Force
    }

    Write-Host "Dataset ready at $datasetTarget"
}

Invoke-CheckedNative -Description "Training model..." -FilePath $pythonExe -Arguments @((Join-Path $backendDir "train_model.py"))

if (-not (Test-Path $modelTarget)) {
    throw "Training finished but model not found at $modelTarget"
}

Write-Host "Model saved: $modelTarget"

$serverScript = "import uvicorn; uvicorn.run('app.main:app', app_dir='backend', host='$HostUrl', port=$Port, reload=False)"
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
        $resp = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 2
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
Write-Host "Browser opened. Dashboard: $appUrl/dashboard"

param(
    [string]$Python = "python",
    [string]$Root = ".",
    [int]$Records = 10000,
    [int]$Steps = 2000,
    [int]$Seed = 1448,
    [int]$Batch = 16,
    [int]$Context = 256
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Repo = (Resolve-Path $Root).Path
$DataDir = Join-Path $Repo 'data\generated'
$ArtifactDir = Join-Path $Repo 'artifacts\verified-math-v1'
$Data = Join-Path $DataDir 'verified_math_v1.jsonl'
$Model = Join-Path $ArtifactDir 'niyah-ref.pt'
$Receipt = Join-Path $ArtifactDir 'receipt.txt'

New-Item -ItemType Directory -Force -Path $DataDir,$ArtifactDir | Out-Null

Push-Location $Repo
try {
    & $Python tools/generate_verified_math.py --out $Data --records $Records --seed $Seed
    if ($LASTEXITCODE -ne 0) { throw "dataset generation failed: $LASTEXITCODE" }

    & $Python tools/validate_verified_math.py $Data --expect-records $Records
    if ($LASTEXITCODE -ne 0) { throw "dataset validation failed: $LASTEXITCODE" }

    $DataHash = (Get-FileHash $Data -Algorithm SHA256).Hash

    & $Python python/niyah_ref.py train `
        --data $Data `
        --out $Model `
        --steps $Steps `
        --batch $Batch `
        --context $Context `
        --seed $Seed
    if ($LASTEXITCODE -ne 0) { throw "training failed: $LASTEXITCODE" }

    $ModelHash = (Get-FileHash $Model -Algorithm SHA256).Hash
    $GitHead = (git rev-parse HEAD).Trim()
    $Torch = & $Python -c "import torch; print(torch.__version__)"
    $Cuda = & $Python -c "import torch; print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

    @(
        "STATUS=PASS"
        "GIT_HEAD=$GitHead"
        "RECORDS=$Records"
        "STEPS=$Steps"
        "SEED=$Seed"
        "DATA_SHA256=$DataHash"
        "MODEL_SHA256=$ModelHash"
        "PYTHON=$(& $Python --version 2>&1)"
        "TORCH=$Torch"
        "DEVICE=$($Cuda -join ' / ')"
    ) | Set-Content $Receipt -Encoding utf8NoBOM

    Get-Content $Receipt
}
finally {
    Pop-Location
}

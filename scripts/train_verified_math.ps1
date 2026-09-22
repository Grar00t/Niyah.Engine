param(
    [string]$Python = "python",
    [string]$Root = ".",
    [int]$Records = 10000,
    [int]$Steps = 2000,
    [int]$Seed = 1448,
    [int]$Batch = 16,
    [int]$Context = 256,
    [double]$ValidationPercent = 5.0
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Repo = (Resolve-Path $Root).Path
$DataDir = Join-Path $Repo 'data\generated'
$ArtifactDir = Join-Path $Repo 'artifacts\verified-math-v1'
$Data = Join-Path $DataDir 'verified_math_v1.jsonl'
$Train = Join-Path $DataDir 'train.jsonl'
$Validation = Join-Path $DataDir 'validation.jsonl'
$Model = Join-Path $ArtifactDir 'niyah-ref.pt'
$Receipt = Join-Path $ArtifactDir 'receipt.txt'

New-Item -ItemType Directory -Force -Path $DataDir,$ArtifactDir | Out-Null

Push-Location $Repo
try {
    & $Python tools/generate_verified_math.py --out $Data --records $Records --seed $Seed
    if ($LASTEXITCODE -ne 0) { throw "dataset generation failed: $LASTEXITCODE" }

    & $Python tools/validate_verified_math.py $Data --expect-records $Records
    if ($LASTEXITCODE -ne 0) { throw "dataset validation failed: $LASTEXITCODE" }

    & $Python tools/split_jsonl.py $Data `
        --train $Train `
        --validation $Validation `
        --validation-percent $ValidationPercent `
        --seed $Seed
    if ($LASTEXITCODE -ne 0) { throw "dataset split failed: $LASTEXITCODE" }

    & $Python tools/validate_verified_math.py $Train
    if ($LASTEXITCODE -ne 0) { throw "train split validation failed: $LASTEXITCODE" }

    & $Python tools/validate_verified_math.py $Validation
    if ($LASTEXITCODE -ne 0) { throw "validation split validation failed: $LASTEXITCODE" }

    $DataHash = (Get-FileHash $Data -Algorithm SHA256).Hash
    $TrainHash = (Get-FileHash $Train -Algorithm SHA256).Hash
    $ValidationHash = (Get-FileHash $Validation -Algorithm SHA256).Hash

    & $Python python/train_verified.py `
        --train $Train `
        --validation $Validation `
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
        "VALIDATION_PERCENT=$ValidationPercent"
        "DATA_SHA256=$DataHash"
        "TRAIN_SHA256=$TrainHash"
        "VALIDATION_SHA256=$ValidationHash"
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

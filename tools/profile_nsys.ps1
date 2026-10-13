param(
    [string]$Exe = ".\build-ps76-cuda-audit\Release\niyah-train.exe",
    [string[]]$TrainerArguments = @(),
    [string]$OutputDirectory = "profiles",
    [string]$Nsys = "nsys"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not (Get-Command $Nsys -ErrorAction SilentlyContinue)) {
    throw "NSYS_NOT_FOUND"
}
if (-not (Test-Path -LiteralPath $Exe -PathType Leaf)) {
    throw "TRAINER_NOT_FOUND: $Exe"
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$stamp = (Get-Date -Format "yyyyMMdd-HHmmss") + '-' + [guid]::NewGuid().ToString('N')
$report = Join-Path $OutputDirectory "niyah-$stamp"

$profileArguments = @('profile', '-t', 'cuda,nvtx,osrt', '--stats=true', '-o', $report,
    '--force-overwrite', 'false', $Exe) + $TrainerArguments
& $Nsys @profileArguments
if ($LASTEXITCODE -ne 0) { throw "NSYS_PROFILE_FAILED: $LASTEXITCODE" }

$rep = "$report.nsys-rep"
if (-not (Test-Path -LiteralPath $rep -PathType Leaf)) { throw "NSYS_REPORT_MISSING: $rep" }

foreach ($kind in @("cuda_gpu_kern_sum", "cuda_gpu_mem_time_sum")) {
    & $Nsys stats --report $kind $rep
    if ($LASTEXITCODE -ne 0) { throw "NSYS_STATS_FAILED: $kind exit=$LASTEXITCODE" }
}

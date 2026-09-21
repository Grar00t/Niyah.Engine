[CmdletBinding()]
param(
    [ValidateSet('Release', 'Debug')]
    [string]$Configuration = 'Release',

    [string]$BuildDir = '',

    [ValidateRange(1, 128)]
    [int]$Jobs = 2
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = (Resolve-Path (Join-Path $ScriptDir '..')).Path

if ([string]::IsNullOrWhiteSpace($BuildDir)) {
    $BuildDir = Join-Path $RootDir ("build\verify-{0}" -f $Configuration.ToLowerInvariant())
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,

        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $Command $($Arguments -join ' ')"
    }
}

Invoke-Checked cmake `
    -S $RootDir `
    -B $BuildDir `
    '-DNIYAH_BUILD_TESTS=ON'

Invoke-Checked cmake `
    --build $BuildDir `
    --config $Configuration `
    --parallel $Jobs

Invoke-Checked ctest `
    --test-dir $BuildDir `
    -C $Configuration `
    --output-on-failure

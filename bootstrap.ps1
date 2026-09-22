$ErrorActionPreference = 'Stop'

cmake -S . -B build -A x64
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure

if ($LASTEXITCODE -ne 0) {
    throw "Niyah.Engine bootstrap failed with exit code $LASTEXITCODE"
}

Write-Host 'NIYAH_BOOTSTRAP=PASS'

@echo off
setlocal
set "STAGE=%~1"
if "%STAGE%"=="" set "STAGE=all"

if "%BUILD_TYPE%"=="" set "BUILD_TYPE=Release"

if /I "%STAGE%"=="native" goto native
if /I "%STAGE%"=="search" goto search
if /I "%STAGE%"=="storage" goto storage
if /I "%STAGE%"=="python" goto python
if /I "%STAGE%"=="all" goto native

echo unknown stage: %STAGE% ^(native^|search^|storage^|python^|all^)
exit /b 2

:native
cmake -S native -B build\native -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DBUILD_TESTING=ON || exit /b 1
cmake --build build\native --config %BUILD_TYPE% || exit /b 1
ctest --test-dir build\native -C %BUILD_TYPE% --output-on-failure || exit /b 1
if /I not "%STAGE%"=="all" goto done

:search
cmake -S search -B build\search -DCMAKE_BUILD_TYPE=%BUILD_TYPE% || exit /b 1
cmake --build build\search --config %BUILD_TYPE% || exit /b 1
ctest --test-dir build\search -C %BUILD_TYPE% --output-on-failure || exit /b 1
if /I not "%STAGE%"=="all" goto done

:storage
cmake -S storage -B build\storage -DCMAKE_BUILD_TYPE=%BUILD_TYPE% || exit /b 1
cmake --build build\storage --config %BUILD_TYPE% || exit /b 1
ctest --test-dir build\storage -C %BUILD_TYPE% --output-on-failure || exit /b 1
if /I not "%STAGE%"=="all" goto done

:python
python -m compileall -q tools scripts src tests || exit /b 1
python -m unittest discover -s tests -p "test_*.py" || exit /b 1
python tools\tests\test_convert_gguf.py || exit /b 1
python tools\tests\test_kquants.py || exit /b 1

:done
echo ok: %STAGE%
endlocal

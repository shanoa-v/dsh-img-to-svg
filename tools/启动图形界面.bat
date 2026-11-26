@echo off
chcp 65001 >nul
title dsh-img-to-svg
cd /d "%~dp0"

rem ============================================================
rem  查找 Python 解释器
rem  优先级：PY 环境变量 > PATH 上的 python > py 启动器 > 常见安装位置
rem  注意：路径与参数分开存放，否则 "%VAR%" 会把 "py.exe -3" 当成一个文件名
rem ============================================================
set "PYRUN="
set "PYARG="

if defined PY if exist "%PY%" set "PYRUN=%PY%"

set "FALLBACK="
if not defined PYRUN (
  for /f "delims=" %%i in ('where python 2^>nul') do (
    if not defined PYRUN (
      "%%i" -X utf8 -c "import numpy,PIL,sklearn,skimage" >nul 2>nul
      if not errorlevel 1 set "PYRUN=%%i"
    )
    if not defined FALLBACK set "FALLBACK=%%i"
  )
)

if not defined PYRUN (
  for /f "delims=" %%i in ('where py 2^>nul') do (
    if not defined PYRUN (
      "%%i" -3 -X utf8 -c "import numpy,PIL,sklearn,skimage" >nul 2>nul
      if not errorlevel 1 (
        set "PYRUN=%%i"
        set "PYARG=-3"
      )
    )
  )
)

if not defined PYRUN (
  for %%p in (
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "C:\Python314\python.exe"
    "C:\Python313\python.exe"
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
    "C:\Python310\python.exe"
    "D:\anaconda3\python.exe"
    "C:\ProgramData\anaconda3\python.exe"
    "C:\ProgramData\miniconda3\python.exe"
  ) do (
    if not defined PYRUN if exist %%p (
      %%p -X utf8 -c "import numpy,PIL,sklearn,skimage" >nul 2>nul
      if not errorlevel 1 set "PYRUN=%%p"
    )
  )
)

rem 一个带依赖的都没有，但有可用解释器：用它去装依赖
if not defined PYRUN (
  if defined FALLBACK (
    set "PYRUN=%FALLBACK%"
    echo.
    echo   [提示] 找到的 Python 都没有装齐依赖，将用下面这个来安装:
    echo        %FALLBACK%
    echo        如需指定别的解释器: set PY=^<python.exe 的完整路径^>
    echo.
  )
)

if not defined PYRUN (
  echo.
  echo   [错误] 找不到 Python 解释器
  echo.
  echo   请安装 Python 3.10 或更高版本:
  echo       https://www.python.org/downloads/
  echo   安装时务必勾选 "Add python.exe to PATH"
  echo.
  echo   若已安装但仍报此错，可手动指定后重试:
  echo       set PY=C:\Python313\python.exe
  echo       %~nx0
  echo.
  pause
  exit /b 1
)

rem 引号只加在路径上；参数（如 -3）单独展开
"%PYRUN%" %PYARG% -X utf8 -c "import numpy,PIL,sklearn,skimage" 2>nul
if errorlevel 1 (
  echo.
  echo   [提示] 缺少第三方依赖，正在安装...
  "%PYRUN%" %PYARG% -X utf8 -m pip install -r "%~dp0..\requirements.txt"
  if errorlevel 1 (
    echo.
    echo   [错误] 依赖安装失败，请手动执行:
    echo       "%PYRUN%" %PYARG% -m pip install -r "%~dp0..\requirements.txt"
    echo.
    pause
    exit /b 1
  )
  echo   依赖安装完成。
)

echo.
echo   Python: %PYRUN% %PYARG%
echo.

"%PYRUN%" %PYARG% -X utf8 "gui.py"
if errorlevel 1 pause

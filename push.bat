@echo off
REM ============================================================================
REM  MAYA — publish current changes to GitHub (origin/main)
REM  Runs from your machine where your GitHub credentials already exist.
REM  Safe: shows you what will be included and waits for confirmation.
REM ============================================================================
setlocal
cd /d "%~dp0"

echo.
echo === Current changes that will be committed ===
git status --short
echo.
echo Local branch vs origin/main:
git rev-list --left-right --count origin/main...HEAD 2>nul
echo   (left = commits only on origin, right = commits only on local)
echo.

set /p MSG=Commit message (blank = default):
if "%MSG%"=="" set MSG=Security hardening + image pipeline fixes

echo.
echo Press Ctrl+C now to abort, or
pause

REM --- stage + commit ---------------------------------------------------------
git add -A
git commit -m "%MSG%"
if errorlevel 1 (
    echo.
    echo Nothing to commit, or commit failed. Stopping.
    goto :end
)

REM --- reconcile with remote (you are behind origin/main) ----------------------
echo.
echo Rebasing onto origin/main...
git pull --rebase origin main
if errorlevel 1 (
    echo.
    echo *** Rebase hit a conflict. ***
    echo Resolve the conflicted files, then run:
    echo     git rebase --continue
    echo     git push origin main
    echo (or "git rebase --abort" to back out).
    goto :end
)

REM --- publish ----------------------------------------------------------------
echo.
echo Pushing to origin/main...
git push origin main
if errorlevel 1 (
    echo.
    echo Push failed — check your GitHub authentication and try again.
    goto :end
)

echo.
echo === Done. Changes are published to GitHub. ===

:end
echo.
pause
endlocal

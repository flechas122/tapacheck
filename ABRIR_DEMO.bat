@echo off
rem Doble clic: demo con fotos de valid + test del dataset.
rem Arrastrar una foto o carpeta sobre este archivo: inspecciona esa foto o carpeta.
cd /d "%~dp0"
".venv\Scripts\python.exe" demo_tapacheck.py %*
pause

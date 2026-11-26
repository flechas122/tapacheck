@echo off
rem TapaCheck 2.0 - comprobacion en tiempo real (Entrega 4)
rem Doble clic: flujo continuo con fotos de valid + test del dataset (simula la banda transportadora).
rem Arrastrar una foto o carpeta sobre este archivo: inspecciona esas fotos.
rem Camara del computador, desde una terminal:  ABRIR_DEMO_TIEMPO_REAL.bat --camara 0
cd /d "%~dp0"
".venv\Scripts\python.exe" demo_tiempo_real.py %*
pause

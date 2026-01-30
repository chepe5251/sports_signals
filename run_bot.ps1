$ErrorActionPreference = 'Stop'

$python = '.\venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    throw "No se encontro $python"
}

# Asegura dataset actualizado con 10 partidos
& $python 'src\ml\build_dataset.py'
if ($LASTEXITCODE -ne 0) {
    throw "Paso fallo: build_dataset"
}

# Listener infinito
& $python 'src\scripts\telegram_listener.py' --minutes 0
if ($LASTEXITCODE -ne 0) {
    throw "Paso fallo: Listener Telegram"
}

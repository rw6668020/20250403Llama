# Enable GPU support
$env:LLAMA_CUBLAS = "1"

# Go to project folder
cd "C:\Users\Desktop\Desktop\20250403Llama"

# Optional: deactivate virtual environment first (avoids conflicts)
if ($env:VIRTUAL_ENV) { deactivate }

# Activate venv
& ".\.venv\Scripts\Activate.ps1"

# Clear pip cache (to avoid "Permission denied" on wheel file)
Remove-Item "$env:LOCALAPPDATA\pip\cache\wheels\*" -Recurse -Force -ErrorAction SilentlyContinue

# Rebuild llama-cpp-python from source with GPU support
pip install llama-cpp-python --force-reinstall --upgrade --no-binary :all:

# Run test
& ".\.venv\Scripts\python.exe" ".\test.py"

# Pause so you can read the output
Read-Host -Prompt "Press Enter to close"

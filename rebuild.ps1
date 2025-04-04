# Run this script as Administrator!

# Enable GPU support with CUDA
$env:LLAMA_CUBLAS = "1"
$env:CMAKE_ARGS = "-DLLAMA_CUBLAS=on -DLLAMA_CUDA_FORCE_MMQ=on"
$env:FORCE_CMAKE = "1"

# Go to project folder
cd "C:\Users\Desktop\Desktop\20250403Llama"

# Optional: deactivate virtual environment first (avoids conflicts)
if ($env:VIRTUAL_ENV) { deactivate }

# Activate venv
& ".\.venv\Scripts\Activate.ps1"

# Clear pip cache completely
Write-Host "Clearing pip cache..."
Remove-Item "$env:LOCALAPPDATA\pip\cache\*" -Recurse -Force -ErrorAction SilentlyContinue

# Verify CUDA is installed and detected
Write-Host "Checking CUDA installation..."
nvidia-smi

# Install without cache and with admin privileges
Write-Host "Installing llama-cpp-python with CUDA support..."
pip install llama-cpp-python --no-cache-dir --force-reinstall --verbose --no-binary llama-cpp-python

# Verify installation
Write-Host "Verifying installation..."
pip list | findstr llama-cpp-python

# Run test to verify GPU usage
Write-Host "Running GPU test..."
& ".\.venv\Scripts\python.exe" ".\test.py"

# Pause so you can read the output
Read-Host -Prompt "Press Enter to close"
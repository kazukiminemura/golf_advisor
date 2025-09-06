param(
  [string]$RepoId = "bartowski/Qwen2.5-1.5B-Instruct-GGUF",
  [string]$Filename = "qwen2.5-1.5b-instruct-q4_k_m.gguf",
  [string]$LocalDir = "models"
)

Write-Host "Downloading $RepoId :: $Filename to $LocalDir" -ForegroundColor Cyan
if (-not (Get-Command "huggingface-cli" -ErrorAction SilentlyContinue)) {
  Write-Host "huggingface-cli not found. Install via: pip install -U huggingface-hub" -ForegroundColor Yellow
  exit 1
}

New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null
huggingface-cli download $RepoId $Filename --local-dir $LocalDir
if ($LASTEXITCODE -ne 0) {
  Write-Host "Download failed with exit code $LASTEXITCODE" -ForegroundColor Red
  exit $LASTEXITCODE
}
Write-Host "Done: $LocalDir/$Filename" -ForegroundColor Green


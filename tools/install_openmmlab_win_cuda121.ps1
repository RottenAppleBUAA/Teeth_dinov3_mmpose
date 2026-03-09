param(
    [string]$EnvName = "openmmlab",
    [string]$PythonVersion = "3.11",
    [switch]$KeepExistingEnv,
    [switch]$SkipDemo
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$env:PYTHONNOUSERSITE = "1"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$SmokeTestImage = Join-Path $RepoRoot "tests\data\coco\000000000785.jpg"
$SmokeTestOutput = Join-Path $RepoRoot "vis_results"

$TorchVersion = "2.1.0"
$TorchVisionVersion = "0.16.0"
$TorchAudioVersion = "2.1.0"
$MMCVVersion = "2.1.0"
$MMEngineVersion = "0.10.5"
$MMDetVersion = "3.2.0"
$NumPyVersion = "1.26.4"
$SetuptoolsVersion = "69.5.1"

$TorchIndexUrl = "https://download.pytorch.org/whl/cu121"
$MMCVFindLinks = "https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html"

$DetConfig = Join-Path $RepoRoot "demo\mmdetection_cfg\rtmdet_m_640-8xb32_coco-person.py"
$DetCheckpoint = "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
$PoseConfig = Join-Path $RepoRoot "configs\body_2d_keypoint\topdown_heatmap\coco\td-hm_hrnet-w32_8xb64-210e_coco-256x192.py"
$PoseCheckpoint = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth"

function Invoke-CondaHook {
    $condaHook = & conda shell.powershell hook | Out-String
    if (-not $condaHook) {
        throw "Failed to initialize conda shell integration."
    }
    Invoke-Expression $condaHook
}

function Install-OptionalPackage {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Arguments
    )

    try {
        Invoke-Expression "python -m pip install $Arguments"
    }
    catch {
        Write-Warning "Optional install failed: python -m pip install $Arguments"
    }
}

Write-Host "Initializing conda..."
Invoke-CondaHook

if (-not $KeepExistingEnv) {
    $existingEnv = conda env list | Select-String -Pattern "^\s*$EnvName\s"
    if ($existingEnv) {
        Write-Host "Removing existing conda environment: $EnvName"
        conda remove -n $EnvName --all -y
    }
}

$envExists = conda env list | Select-String -Pattern "^\s*$EnvName\s"
if (-not $envExists) {
    Write-Host "Creating conda environment: $EnvName (Python $PythonVersion)"
    conda create -n $EnvName python=$PythonVersion -y
}

Write-Host "Activating conda environment: $EnvName"
conda activate $EnvName

Write-Host "Installing packaging toolchain..."
python -m pip install -U pip
python -m pip install --force-reinstall setuptools==$SetuptoolsVersion wheel packaging
python -m pip install --force-reinstall numpy==$NumPyVersion

Write-Host "Installing PyTorch with CUDA 12.1..."
python -m pip install `
    torch==$TorchVersion `
    torchvision==$TorchVisionVersion `
    torchaudio==$TorchAudioVersion `
    --index-url $TorchIndexUrl

Write-Host "Installing OpenMMLab runtime packages..."
python -m pip install mmengine==$MMEngineVersion
python -m pip install mmcv==$MMCVVersion -f $MMCVFindLinks
python -m pip install mmdet==$MMDetVersion

Write-Host "Installing MMPose Python dependencies..."
python -m pip install json_tricks matplotlib munkres opencv-python pillow scipy xtcocotools
Install-OptionalPackage "--no-build-isolation --no-use-pep517 chumpy"

Write-Host "Installing current MMPose repo in editable mode..."
Set-Location $RepoRoot
python -m pip install -e . --no-deps

Write-Host "Verifying imports..."
python -c "import numpy, setuptools, torch, mmcv, mmengine, mmdet, mmpose; print('numpy=' + numpy.__version__); print('setuptools=' + setuptools.__version__); print('torch=' + torch.__version__); print('cuda=' + str(torch.version.cuda)); print('gpu=' + str(torch.cuda.is_available())); print('mmcv=' + mmcv.__version__); print('mmengine=' + mmengine.__version__); print('mmdet=' + mmdet.__version__); print('mmpose=' + mmpose.__version__)"

if (-not $SkipDemo) {
    Write-Host "Running top-down demo smoke test..."
    New-Item -ItemType Directory -Force -Path $SmokeTestOutput | Out-Null

    python demo\topdown_demo_with_mmdet.py `
        $DetConfig `
        $DetCheckpoint `
        $PoseConfig `
        $PoseCheckpoint `
        --input $SmokeTestImage `
        --device cuda:0 `
        --output-root $SmokeTestOutput

    Write-Host "Smoke test output saved to: $SmokeTestOutput"
}

Write-Host "Environment setup complete."

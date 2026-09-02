param(
    [ValidateSet('live', 'demo')]
    [string]$Mode = 'live'
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$simulator = Join-Path $projectRoot '.venv\Scripts\wildlife-simulator.exe'
$startedProcesses = [System.Collections.Generic.List[System.Diagnostics.Process]]::new()

if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing Python environment. Run scripts\setup_windows.bat first."
}

if (-not $env:NUMBA_CACHE_DIR) {
    $env:NUMBA_CACHE_DIR = Join-Path $env:TEMP 'wildlife-soundscape-numba-cache'
}

function Test-LocalPort {
    param([int]$Port)

    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $connect = $client.ConnectAsync('127.0.0.1', $Port)
        if (-not $connect.Wait(500)) {
            return $false
        }
        return $client.Connected
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

function Wait-ForService {
    param(
        [System.Diagnostics.Process]$Process,
        [int]$Port,
        [string]$Name,
        [int]$TimeoutSeconds = 45
    )

    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        if ($Process.HasExited) {
            throw "$Name exited before opening port $Port (exit $($Process.ExitCode))."
        }
        if (Test-LocalPort -Port $Port) {
            return
        }
        Start-Sleep -Milliseconds 250
    }
    throw "$Name did not open port $Port within $TimeoutSeconds seconds."
}

function Start-WildlifeProcess {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )

    $startParameters = @{
        FilePath = $FilePath
        WorkingDirectory = $projectRoot
        WindowStyle = 'Normal'
        PassThru = $true
    }
    if ($Arguments.Count -gt 0) {
        $startParameters.ArgumentList = $Arguments
    }

    $process = Start-Process @startParameters
    $startedProcesses.Add($process)
    return $process
}

if (Test-LocalPort -Port 5001) {
    throw 'Receiver port 5001 is already in use.'
}
if (Test-LocalPort -Port 8501) {
    throw 'Dashboard port 8501 is already in use.'
}

try {
    Write-Host '[1/3] Starting receiver...'
    $receiverProcess = Start-WildlifeProcess `
        -FilePath $python `
        -Arguments @(
            '-m', 'wildlife_soundscape',
            '--auto-start',
            '--session-label', $Mode
        )
    Wait-ForService -Process $receiverProcess -Port 5001 -Name 'Receiver'

    if ($Mode -eq 'demo') {
        Write-Host '[2/3] Starting three-node simulator...'
        if (-not (Test-Path -LiteralPath $simulator)) {
            throw 'Missing wildlife-simulator executable. Run setup again.'
        }
        $simulatorProcess = Start-WildlifeProcess `
            -FilePath $simulator `
            -Arguments @()
    }
    else {
        Write-Host '[2/3] Live mode: receiver is waiting for physical nodes.'
    }

    Write-Host '[3/3] Starting Streamlit dashboard...'
    $dashboardProcess = Start-WildlifeProcess `
        -FilePath $python `
        -Arguments @('-m', 'streamlit', 'run', 'dashboard\app.py')
    Wait-ForService -Process $dashboardProcess -Port 8501 -Name 'Dashboard'

    if ($Mode -eq 'demo') {
        Start-Sleep -Seconds 2
        if ($simulatorProcess.HasExited) {
            throw "Simulator exited during startup (exit $($simulatorProcess.ExitCode))."
        }
    }

    Write-Host ''
    Write-Host "Wildlife Soundscape launched successfully in $Mode mode."
    Write-Host 'Dashboard: http://localhost:8501'
    Write-Host 'Close each service window or press Ctrl+C in it to stop that service.'
}
catch {
    foreach ($process in $startedProcesses) {
        if (-not $process.HasExited) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        }
    }
    throw
}

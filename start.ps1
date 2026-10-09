#Requires -Version 5.1
<#
.SYNOPSIS
    One-command launcher for a local Honcho stack on Windows.

.DESCRIPTION
    Runs the standard host-process setup in one window:

      1. verifies uv is on PATH
      2. checks .env (copies .env.template when it is missing)
      3. syncs dependencies                                        (-Quick skips)
      4. confirms the PostgreSQL from DB_CONNECTION_URI is listening
      5. applies Alembic migrations                                (-Quick skips)
      6. starts the API server and the deriver worker, streaming both
         logs into this window
      7. waits for /health and prints the endpoints

    Nothing is started in Docker: the database comes from DB_CONNECTION_URI
    and must already be running.

    Ctrl+C stops the API server and the deriver.

.PARAMETER Port
    Port for the API server. Default 8000.

.PARAMETER NoDeriver
    Start only the API server.

.PARAMETER Quick
    Skip `uv sync` and `alembic upgrade head`; just start the processes.

.EXAMPLE
    .\start.ps1

.EXAMPLE
    .\start.ps1 -Port 8901

.EXAMPLE
    .\start.ps1 -Quick

.EXAMPLE
    .\start.ps1 -NoDeriver
#>
[CmdletBinding()]
param(
    [int]$Port = 8000,
    [switch]$NoDeriver,
    [switch]$Quick
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = $PSScriptRoot
$HealthUrl = "http://127.0.0.1:$Port/health"
$EnvFile = Join-Path $Root '.env'

# ---------------------------------------------------------------- output ----

function Write-Step {
    param([Parameter(Mandatory)][string]$Message)
    Write-Host ''
    Write-Host "[*] $Message" -ForegroundColor Cyan
}

function Write-Ok {
    param([Parameter(Mandatory)][string]$Message)
    Write-Host "    + $Message" -ForegroundColor Green
}

function Write-Info {
    param([Parameter(Mandatory)][string]$Message)
    Write-Host "    $Message" -ForegroundColor DarkGray
}

function Write-Warn {
    param([Parameter(Mandatory)][string]$Message)
    Write-Host "    ! $Message" -ForegroundColor Yellow
}

function Stop-Launcher {
    param([Parameter(Mandatory)][string]$Message)
    Write-Host ''
    Write-Host "[x] $Message" -ForegroundColor Red
    exit 1
}

# --------------------------------------------------------------- helpers ----

function Test-Command {
    param([Parameter(Mandatory)][string]$Name)
    return ($null -ne (Get-Command -Name $Name -ErrorAction SilentlyContinue))
}

function Invoke-Native {
    param(
        [Parameter(Mandatory)][string]$Exe,
        [Parameter(Mandatory)][string[]]$Arguments,
        [string]$WorkDir = $Root
    )
    Push-Location -LiteralPath $WorkDir
    try {
        & $Exe @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "'$Exe $($Arguments -join ' ')' exited with code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}

function Test-TcpPort {
    param(
        [Parameter(Mandatory)][string]$TargetHost,
        [Parameter(Mandatory)][int]$TargetPort,
        [int]$TimeoutMs = 1500
    )
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $task = $client.ConnectAsync($TargetHost, $TargetPort)
        if (-not $task.Wait($TimeoutMs)) { return $false }
        return $client.Connected
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

function Test-HealthEndpoint {
    param([Parameter(Mandatory)][string]$Url)
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
        return ($response.StatusCode -eq 200)
    }
    catch {
        return $false
    }
}

# Minimal .env reader: KEY=VALUE, `#` comments, optional surrounding quotes.
function Get-DotEnvValue {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Key
    )
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    foreach ($line in (Get-Content -LiteralPath $Path)) {
        $trimmed = $line.Trim()
        if ($trimmed.Length -eq 0 -or $trimmed.StartsWith('#')) { continue }
        $separator = $trimmed.IndexOf('=')
        if ($separator -lt 1) { continue }
        if ($trimmed.Substring(0, $separator).Trim() -ne $Key) { continue }
        $value = $trimmed.Substring($separator + 1).Trim()
        if ($value.Length -ge 2 -and $value.StartsWith('"') -and $value.EndsWith('"')) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        return $value
    }
    return $null
}

# postgresql+psycopg://user:pass@host:port/db -> host/port (5432 when omitted).
function Get-DbEndpoint {
    param([Parameter(Mandatory)][string]$Uri)
    $match = [regex]::Match($Uri, '@([^/:@]+)(?::(\d+))?/')
    if (-not $match.Success) { return $null }
    $dbPort = 5432
    if ($match.Groups[2].Success) { $dbPort = [int]$match.Groups[2].Value }
    return New-Object -TypeName psobject -Property @{
        Host = $match.Groups[1].Value
        Port = $dbPort
    }
}

# Name of a stopped local PostgreSQL service, when the machine has one, so the
# "database is down" error can name the exact command to run.
function Get-StoppedPostgresService {
    $services = Get-Service -Name 'postgresql*' -ErrorAction SilentlyContinue
    if ($null -eq $services) { return $null }
    $stopped = $services | Where-Object { $_.Status -ne 'Running' } | Select-Object -First 1
    if ($null -eq $stopped) { return $null }
    return $stopped.Name
}

function Show-Banner {
    Write-Host ''
    Write-Host '  Honcho is up' -ForegroundColor Green
    Write-Host "    API       http://127.0.0.1:$Port" -ForegroundColor Gray
    Write-Host "    Docs      http://127.0.0.1:$Port/docs" -ForegroundColor Gray
    Write-Host "    Health    http://127.0.0.1:$Port/health" -ForegroundColor Gray
    if (-not $NoDeriver) {
        Write-Host '    Deriver   running in this window' -ForegroundColor Gray
    }
    Write-Host ''
    Write-Host '  Ctrl+C stops the API server and the deriver.' -ForegroundColor Gray
    Write-Host ''
}

# ------------------------------------------------------------------ main ----

Write-Host ''
Write-Host 'Honcho local launcher' -ForegroundColor White
Write-Info "repo    $Root"

Write-Step 'Checking prerequisites'
if (-not (Test-Command -Name 'uv')) {
    Stop-Launcher 'uv is not on PATH. Install it: https://docs.astral.sh/uv/getting-started/installation/'
}
Write-Ok "uv $((& uv --version | Out-String).Trim())"

Write-Step 'Checking configuration'
if (-not (Test-Path -LiteralPath $EnvFile)) {
    if (Test-Path -LiteralPath (Join-Path $Root '.env.template')) {
        Copy-Item -LiteralPath (Join-Path $Root '.env.template') -Destination $EnvFile
        Stop-Launcher 'created .env from .env.template - fill in DB_CONNECTION_URI and your LLM API keys, then re-run.'
    }
    Stop-Launcher 'no .env found - copy .env.template to .env and configure it.'
}
Write-Ok 'using .env'

$dbUri = Get-DotEnvValue -Path $EnvFile -Key 'DB_CONNECTION_URI'
if ([string]::IsNullOrWhiteSpace($dbUri)) {
    Stop-Launcher 'DB_CONNECTION_URI is not set in .env (example: postgresql+psycopg://postgres:postgres@localhost:5432/postgres).'
}
$dbEndpoint = Get-DbEndpoint -Uri $dbUri
if ($null -ne $dbEndpoint) {
    Write-Ok "database target $($dbEndpoint.Host):$($dbEndpoint.Port)"
}
else {
    Write-Warn 'could not parse host/port from DB_CONNECTION_URI'
}

Write-Step 'Checking the database'
if ($null -ne $dbEndpoint) {
    if (-not (Test-TcpPort -TargetHost $dbEndpoint.Host -TargetPort $dbEndpoint.Port)) {
        $hint = ''
        $service = Get-StoppedPostgresService
        if ($null -ne $service) {
            $hint = " The local service '$service' is not running - start it (Start-Service $service) and re-run."
        }
        Stop-Launcher "no PostgreSQL listening on $($dbEndpoint.Host):$($dbEndpoint.Port) (DB_CONNECTION_URI in .env).$hint"
    }
    Write-Ok "PostgreSQL reachable at $($dbEndpoint.Host):$($dbEndpoint.Port)"
}
else {
    Write-Warn 'database reachability not checked'
}

if (-not $Quick) {
    Write-Step 'Syncing dependencies (uv sync)'
    Invoke-Native -Exe 'uv' -Arguments @('sync')
    Write-Ok 'environment ready'

    Write-Step 'Applying database migrations (alembic upgrade head)'
    try {
        Invoke-Native -Exe 'uv' -Arguments @('run', 'alembic', 'upgrade', 'head')
    }
    catch {
        Stop-Launcher "migrations failed: $($_.Exception.Message). Check DB_CONNECTION_URI in .env and that the database exists."
    }
    Write-Ok 'schema up to date'
}

# --------------------------------------------------------------- launch -----

# Keep child output unbuffered so log lines appear as they are produced.
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'

$api = $null
$deriver = $null
$exitReason = $null

try {
    Write-Step 'Starting the deriver worker'
    if ($NoDeriver) {
        Write-Info 'skipped (-NoDeriver)'
    }
    else {
        $deriver = Start-Process -FilePath 'uv' `
            -ArgumentList @('run', 'python', '-m', 'src.deriver') `
            -WorkingDirectory $Root -NoNewWindow -PassThru
        Write-Ok "deriver pid $($deriver.Id)"
    }

    Write-Step 'Starting the API server'
    $api = Start-Process -FilePath 'uv' `
        -ArgumentList @('run', 'fastapi', 'dev', 'src/main.py', '--host', '127.0.0.1', '--port', "$Port") `
        -WorkingDirectory $Root -NoNewWindow -PassThru
    Write-Ok "api pid $($api.Id)"

    Write-Step 'Waiting for the API'
    $healthy = $false
    $deadline = (Get-Date).AddSeconds(180)
    while ((Get-Date) -lt $deadline -and $null -eq $exitReason) {
        if ($api.HasExited) {
            $exitReason = "the API server exited with code $($api.ExitCode) before becoming healthy"
            break
        }
        if ($null -ne $deriver -and $deriver.HasExited) {
            $exitReason = "the deriver exited with code $($deriver.ExitCode)"
            break
        }
        if (Test-HealthEndpoint -Url $HealthUrl) { $healthy = $true; break }
        Start-Sleep -Milliseconds 500
    }
    if ($healthy) {
        Write-Ok "healthy at $HealthUrl"
    }
    elseif ($null -eq $exitReason) {
        Write-Warn 'no /health response after 180s - check the log output above'
    }

    if ($null -eq $exitReason) {
        Show-Banner
        while ($null -eq $exitReason) {
            if ($api.HasExited) {
                $exitReason = "the API server stopped with exit code $($api.ExitCode)"
            }
            elseif ($null -ne $deriver -and $deriver.HasExited) {
                $exitReason = "the deriver stopped with exit code $($deriver.ExitCode)"
            }
            else {
                Start-Sleep -Seconds 1
            }
        }
    }
}
finally {
    # uv spawns python as a child on Windows, so kill the whole tree.
    if ($null -ne $api -and -not $api.HasExited) {
        $null = cmd /c "taskkill /PID $($api.Id) /T /F >nul 2>&1"
    }
    if ($null -ne $deriver -and -not $deriver.HasExited) {
        $null = cmd /c "taskkill /PID $($deriver.Id) /T /F >nul 2>&1"
    }
    if ($null -ne $api -or $null -ne $deriver) {
        Write-Host ''
        Write-Host '  Stopped.' -ForegroundColor Gray
    }
}

if ($null -ne $exitReason) {
    Stop-Launcher $exitReason
}

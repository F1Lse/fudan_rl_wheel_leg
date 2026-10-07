param(
    [int]$LocalPort = 5901,
    [int]$RemotePort = 5901,
    [switch]$NoViewer
)

$ErrorActionPreference = "Stop"
$sshExe = (Get-Command ssh.exe -ErrorAction Stop).Source
$configPath = Join-Path $PSScriptRoot ".remote_vnc_ssh.xml"

function Read-SshCommand {
    $text = (Read-Host "Enter latest SSH command (example: ssh -p 30704 root@183.147.142.40)").Trim()
    $text = $text -replace '\\@', '@'
    $match = [regex]::Match($text, '(?i)(?:ssh(?:\.exe)?\s+)?(?:-p\s+([0-9]+)\s+)?([^@\s]+)@([^\s]+)')
    if (-not $match.Success) { throw "Could not parse SSH command." }
    $port = if ($match.Groups[1].Success) { [int]$match.Groups[1].Value } else { 22 }
    return [pscustomobject]@{ Host = $match.Groups[3].Value; Port = $port; User = $match.Groups[2].Value }
}

function Read-SshPassword { return (Read-Host "Enter SSH password (hidden)" -AsSecureString) }

function Save-SshConfig($target, [securestring]$password) {
    [pscustomobject]@{ Host = $target.Host; Port = $target.Port; User = $target.User; Password = $password | ConvertFrom-SecureString } | Export-Clixml -LiteralPath $configPath
}

function Load-SshConfig {
    if (-not (Test-Path -LiteralPath $configPath)) { return $null }
    try {
        $saved = Import-Clixml -LiteralPath $configPath
        if ($saved.Host -and $saved.Port -and $saved.User -and $saved.Password) {
            return [pscustomobject]@{ Host = [string]$saved.Host; Port = [int]$saved.Port; User = [string]$saved.User; Password = ConvertTo-SecureString $saved.Password }
        }
    } catch { }
    return $null
}

function Enable-SshAskPass([securestring]$password) {
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($password)
    try { $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr) } finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }
    $env:REMOTE_VNC_SSH_PASSWORD = $plain
    $askPass = Join-Path $env:TEMP "codex_remote_vnc_askpass.cmd"
    $askPassContent = @'
@echo off
powershell.exe -NoProfile -Command "[Console]::Write($env:REMOTE_VNC_SSH_PASSWORD)"
'@
    Set-Content -LiteralPath $askPass -Encoding ASCII -Value $askPassContent
    $env:SSH_ASKPASS = $askPass
    $env:SSH_ASKPASS_REQUIRE = "force"
    $env:DISPLAY = "codex-askpass"
}

function Invoke-RemoteCheck($target, [securestring]$password, $script) {
    Enable-SshAskPass $password
    & $sshExe -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o PreferredAuthentications=password -o PubkeyAuthentication=no -p $target.Port ("{0}@{1}" -f $target.User, $target.Host) $script
    return ($LASTEXITCODE -eq 0)
}

$target = Load-SshConfig
$password = if ($target) { $target.Password } else { $null }
$viewerCandidates = @("C:\Program Files\TurboVNC\bin\vncviewer.exe", "C:\Program Files\TurboVNC\vncviewer.exe", "C:\Program Files (x86)\TurboVNC\bin\vncviewer.exe", "C:\Program Files (x86)\TurboVNC\vncviewer.exe")
$viewerPath = $viewerCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
$forward = "{0}:127.0.0.1:{1}" -f $LocalPort, $RemotePort
$remoteCheck = @'
set -e
echo VNC_RESTARTING
vncserver -kill :1 >/tmp/codex_vnc_kill.log 2>&1 || true
rm -f /tmp/.X11-unix/X1 /tmp/.X1-lock /root/.vnc/*.pid
vncserver :1 -geometry 1904x967 -depth 24 >/tmp/codex_vnc_start.log 2>&1
sleep 2
ss -ltn 2>/dev/null | grep -q ":5901"
echo VNC_READY
'@

$connected = $false
if ($target -and $password) {
    Write-Host ("Using saved SSH: ssh -p {0} {1}@{2}" -f $target.Port, $target.User, $target.Host)
    $connected = Invoke-RemoteCheck $target $password $remoteCheck
}
if (-not $connected) {
    Write-Host "Saved SSH failed. Enter the latest connection information."
    $target = Read-SshCommand
    $password = Read-SshPassword
    if (-not (Invoke-RemoteCheck $target $password $remoteCheck)) { throw "SSH or remote VNC check failed." }
    Save-SshConfig $target $password
    Write-Host ("SSH config saved with Windows encryption: {0}" -f $configPath)
}

$sshArgs = @("-N", "-T", "-o", "StrictHostKeyChecking=accept-new", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30", "-o", "PreferredAuthentications=password", "-o", "PubkeyAuthentication=no", "-p", "$($target.Port)", "-L", $forward, ("{0}@{1}" -f $target.User, $target.Host))
Write-Host ("SSH tunnel: 127.0.0.1:{0} -> {1}:{2}" -f $LocalPort, $target.Host, $RemotePort)
if (-not $NoViewer -and $viewerPath) { Start-Process -FilePath $viewerPath -ArgumentList ("127.0.0.1:{0}" -f $LocalPort); Write-Host ("TurboVNC started: 127.0.0.1:{0}" -f $LocalPort) }
elseif (-not $viewerPath) { Write-Host ("TurboVNC not found. Connect manually to 127.0.0.1:{0}" -f $LocalPort) }
Write-Host "SSH tunnel is running. Press Ctrl+C to stop."
Enable-SshAskPass $password
$sshProcess = Start-Process -FilePath $sshExe -ArgumentList $sshArgs -PassThru -NoNewWindow
$sshProcess.WaitForExit()
Write-Host ("SSH tunnel ended, exit code: {0}" -f $sshProcess.ExitCode)

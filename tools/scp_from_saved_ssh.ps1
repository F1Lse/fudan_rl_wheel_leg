param(
    [Parameter(Mandatory=$true)] [string]$Source,
    [Parameter(Mandatory=$true)] [string]$Destination
)
$ErrorActionPreference = 'Stop'
$cfg = Import-Clixml (Join-Path $PSScriptRoot '..\.remote_vnc_ssh.xml')
$secure = ConvertTo-SecureString $cfg.Password
$ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try { $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr) }
finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }
$env:REMOTE_VNC_SSH_PASSWORD = $plain
$ask = Join-Path $env:TEMP 'codex_saved_ssh_askpass.cmd'
@'
@echo off
powershell.exe -NoProfile -Command "[Console]::Write($env:REMOTE_VNC_SSH_PASSWORD)"
'@ | Set-Content -LiteralPath $ask -Encoding ASCII
$env:SSH_ASKPASS = $ask
$env:SSH_ASKPASS_REQUIRE = 'force'
$env:DISPLAY = 'codex-askpass'
$sourceSpec = '{0}@{1}:{2}' -f $cfg.User, $cfg.Host, $Source
& scp.exe -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o PreferredAuthentications=password -o PubkeyAuthentication=no -P $cfg.Port -- $sourceSpec $Destination
exit $LASTEXITCODE

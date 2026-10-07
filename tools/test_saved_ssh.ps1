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
$target = '{0}@{1}' -f $cfg.User, $cfg.Host
& ssh.exe -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o PreferredAuthentications=password -o PubkeyAuthentication=no -p $cfg.Port $target 'echo SSH_OK; hostname; pwd'
exit $LASTEXITCODE

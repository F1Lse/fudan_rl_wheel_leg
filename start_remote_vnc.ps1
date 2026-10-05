param(
    [int]$LocalPort = 5901,
    [int]$RemotePort = 5901,
    [switch]$NoViewer
)

$ErrorActionPreference = "Stop"
$sshExe = (Get-Command ssh.exe -ErrorAction Stop).Source
$hostName = "183.147.142.40"
$sshPort = 30248
$userName = "root"

$viewerCandidates = @(
    "C:\Program Files\TurboVNC\bin\vncviewer.exe",
    "C:\Program Files\TurboVNC\vncviewer.exe",
    "C:\Program Files (x86)\TurboVNC\bin\vncviewer.exe",
    "C:\Program Files (x86)\TurboVNC\vncviewer.exe"
)
$viewerPath = $viewerCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1

$forward = "{0}:127.0.0.1:{1}" -f $LocalPort, $RemotePort
$remoteCheck = @'
set -e
if ss -ltn 2>/dev/null | grep -q ":5901"; then
  echo VNC_OK
else
  echo VNC_NOT_LISTENING
  vncserver -kill :1 >/tmp/codex_vnc_kill.log 2>&1 || true
  rm -f /tmp/.X11-unix/X1 /root/.vnc/*.pid
  vncserver :1 -geometry 1920x1080 -depth 24
  sleep 2
  ss -ltn 2>/dev/null | grep -q ":5901"
  echo VNC_STARTED
fi
'@

Write-Host "检查远端 VNC（第一次 SSH 密码提示）..."
& $sshExe -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -p $sshPort ("{0}@{1}" -f $userName, $hostName) $remoteCheck
if ($LASTEXITCODE -ne 0) {
    throw "远端 VNC 检查或启动失败，未建立本地隧道。"
}

$sshArgs = @(
    "-N",
    "-T",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=30",
    "-p", "$sshPort",
    "-L", $forward,
    ("{0}@{1}" -f $userName, $hostName)
)

Write-Host ("SSH tunnel: 127.0.0.1:{0} -> {1}:{2}" -f $LocalPort, $hostName, $RemotePort)
Write-Host "远端 VNC 已确认，下面建立 SSH 隧道（第二次 SSH 密码提示）。"

if (-not $NoViewer -and $viewerPath) {
    Start-Process -FilePath $viewerPath -ArgumentList ("127.0.0.1:{0}" -f $LocalPort)
    Write-Host ("已打开 TurboVNC：127.0.0.1:{0}" -f $LocalPort)
} elseif (-not $viewerPath) {
    Write-Host ("未找到 TurboVNC Viewer。请手动打开 TurboVNC，连接 127.0.0.1:{0}。" -f $LocalPort)
}

Write-Host "SSH 隧道保持在当前窗口。输入密码后不要关闭此窗口。"
Write-Host "按 Ctrl+C 可停止本地 SSH 隧道。"

$sshProcess = Start-Process -FilePath $sshExe -ArgumentList $sshArgs -PassThru -NoNewWindow
$sshProcess.WaitForExit()

if ($sshProcess.ExitCode -ne 0) {
    Write-Host ("SSH 隧道已退出，代码：{0}" -f $sshProcess.ExitCode)
} else {
    Write-Host "SSH 隧道已结束。"
}

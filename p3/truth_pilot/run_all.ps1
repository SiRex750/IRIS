# Truth pilot (PLAN.md, e7ac3b0): CoTracker3 online, then DOT, then the gates. Runs detached from any terminal or
# Claude session; needs no network. Keeps the PC awake only while this script runs (no power setting is changed).
# Logs: run_cotracker3.console.log, run_dot.console.log, gates.console.log, run_all.status.log (this folder).
$ErrorActionPreference = "Continue"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent (Split-Path -Parent $here)
Set-Location $repo
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
$status = Join-Path $here "run_all.status.log"

Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[W.P]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

function Step($name, $argList, $log) {
    "$(Get-Date -Format s) start $name" | Out-File -Append -Encoding utf8 $status
    $cmdline = "python " + ($argList -join " ") + " > `"" + (Join-Path $here $log) + "`" 2>&1"
    & cmd.exe /c $cmdline
    "$(Get-Date -Format s) end $name exit $LASTEXITCODE" | Out-File -Append -Encoding utf8 $status
}

"$(Get-Date -Format s) run_all started, pid $PID" | Out-File -Encoding utf8 $status
Step "cotracker3" @("p3/truth_pilot/run_truth.py", "--cand", "cotracker3") "run_cotracker3.console.log"
Step "dot" @("p3/truth_pilot/run_truth.py", "--cand", "dot") "run_dot.console.log"
Step "gates" @("p3/truth_pilot/truth_gates.py") "gates.console.log"
"$(Get-Date -Format s) ALL DONE" | Out-File -Append -Encoding utf8 $status
[W.P]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null

import subprocess
import os
import time

cwd = r"c:\Users\akash\Documents\Iris"
python_path = os.path.join(cwd, ".venv", "Scripts", "python.exe")
script_path = os.path.join(cwd, "scratch", "test_vram_lifecycle.py")

# DETACHED_PROCESS = 0x00000008
# CREATE_NEW_PROCESS_GROUP = 0x00000200
# Let's try DETACHED_PROCESS and CREATE_NEW_PROCESS_GROUP
creation_flags = 0x00000008 | 0x00000200

print("Spawning process...")
p = subprocess.Popen(
    [python_path, "-u", script_path],
    creationflags=creation_flags,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    cwd=cwd,
    close_fds=True
)
print(f"Spawned. PID: {p.pid}")
time.sleep(2)
status = p.poll()
if status is None:
    print("Process is still running.")
else:
    print(f"Process exited immediately with code {status}")



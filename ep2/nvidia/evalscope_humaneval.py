import runpy
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path

from evalscope.benchmarks.humaneval import utils

ISOLATION = [
    "/usr/bin/prlimit", "--as=1073741824", "--cpu=12", "--fsize=65536", "--nofile=64", "--core=0", "--",
    "/usr/bin/bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL",
    "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
    "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--chdir", "/tmp", "--clearenv",
    "--", "/usr/bin/python3", "-I", "-B", "-c",
]

HARNESS = """
import ctypes
import errno
import sys
s = ctypes.CDLL("libseccomp.so.2")
s.seccomp_init.argtypes = (ctypes.c_uint32,)
s.seccomp_init.restype = ctypes.c_void_p
s.seccomp_syscall_resolve_name.argtypes = (ctypes.c_char_p,)
s.seccomp_rule_add.argtypes = (ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint)
s.seccomp_load.argtypes = (ctypes.c_void_p,)
s.seccomp_release.argtypes = (ctypes.c_void_p,)
allow = 0x7fff0000
deny = 0x00050000 | errno.EPERM
policy = s.seccomp_init(allow)
if not policy:
    raise RuntimeError("cannot create syscall filter")
for name in (b"clone", b"clone3", b"fork", b"vfork", b"execve", b"execveat"):
    syscall = s.seccomp_syscall_resolve_name(name)
    if syscall >= 0 and s.seccomp_rule_add(policy, deny, syscall, 0):
        raise RuntimeError("cannot set syscall filter")
if s.seccomp_load(policy):
    raise RuntimeError("cannot load syscall filter")
s.seccomp_release(policy)
marker = sys.argv[1]
code = compile(sys.stdin.buffer.read(), "<candidate>", "exec")
exec(code, {"__name__": "__main__"})
print(marker)
"""

def check_correctness(problem, completion, timeout, completion_id=None):
    program = problem["prompt"] + completion + "\n" + problem["test"] + "\n" + f'check({problem["entry_point"]})'
    result, passed = "failed", False
    marker = secrets.token_hex(24).encode()
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            process = subprocess.run(ISOLATION + [HARNESS, marker.decode()], input=program.encode(),
                stdout=stdout, stderr=stderr, timeout=float(timeout)+1, check=False)
            stdout.seek(0)
            passed = process.returncode == 0 and marker in stdout.read(65536).splitlines()
            if passed:
                result = "passed"
            else:
                stderr.seek(0)
                error = stderr.read(4096).decode("utf-8",errors="replace").strip().splitlines()
                result = error[-1][:300] if error else f"exit code {process.returncode}"
        except subprocess.TimeoutExpired:
            result = "timed out"
    return {"task_id":problem["task_id"], "passed":passed, "result":result, "completion_id":completion_id}

def self_test():
    p = {"task_id":"isolation-test", "prompt":"def f(x):\n", "entry_point":"f",
         "test":"def check(candidate):\n    assert candidate(2) == 3\n"}
    safe = """    import pathlib, socket, os
    assert not pathlib.Path('/data/admin1/ep2/.secrets').exists()
    assert len(socket.if_nameindex()) <= 1
    try:
        os.fork()
    except PermissionError:
        pass
    else:
        raise AssertionError("fork was allowed")
    return x + 1
"""
    good = check_correctness(p, safe, 5)
    assert good["passed"], good
    assert not check_correctness(p, "    return x\n", 5)["passed"]
    assert not check_correctness(p, "    __import__('os')._exit(0)\n", 5)["passed"]
    print("HumanEval Bubblewrap isolation and pass/fail self-test verified",flush=True)

if __name__ == "__main__":
    self_test()
    if sys.argv[1:] != ["--self-test"]:
        utils.check_correctness = check_correctness
        runpy.run_path(str(Path(sys.executable).with_name("evalscope")), run_name="__main__")

"""
Sandbox for safe execution of LLM-generated Python scripts.

Runs code in a subprocess with:
- Whitelisted imports (blocks subprocess, shutil, socket, etc.)
- Restricted file access (only allowed directories)
- Resource limits: timeout, memory, CPU time
- Disabled dangerous os/sys functions
"""

import subprocess
import sys
import os
import json
import tempfile
from dataclasses import dataclass, field


@dataclass
class SandboxResult:
    """Result of a sandboxed script execution."""
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


# Default whitelist: standard library safe modules + data science stack
DEFAULT_ALLOWED_MODULES: set[str] = {
    # --- Standard library (safe subset) ---
    'math', 'cmath', 'statistics', 'decimal', 'fractions',
    'collections', 'itertools', 'functools', 'operator',
    'json', 'csv', 're', 'string', 'textwrap',
    'datetime', 'time', 'calendar',
    'typing', 'dataclasses', 'abc', 'enum',
    'copy', 'io', 'pathlib',
    'random', 'hashlib',
    'contextlib', 'warnings',
    'pprint', 'logging',
    'unicodedata', 'struct',
    'bisect', 'heapq',
    'os',       # allowed but dangerous attrs will be removed
    'sys',      # allowed but dangerous attrs will be removed

    # --- Data science ---
    'numpy', 'pandas', 'scipy', 'sklearn',
    'matplotlib', 'seaborn', 'plotly',
    'statsmodels', 'xgboost', 'lightgbm', 'catboost',
    'tqdm',
    'PIL', 'cv2',
}


class Sandbox:
    """
    Executes Python code in a restricted subprocess.

    Usage:
        sandbox = Sandbox(
            allowed_dirs=['./workspace', './data'],
            timeout=60,
        )
        result = sandbox.run_file('workspace/analysis.py')
        # or
        result = sandbox.run_code('print(1 + 2)')
    """

    def __init__(
        self,
        allowed_dirs: list[str] | None = None,
        allowed_modules: set[str] | None = None,
        timeout: int = 30,
        max_memory_mb: int = 512,
    ):
        """
        Args:
            allowed_dirs: Directories the script can read/write.
                          The script's own directory is always allowed.
            allowed_modules: Top-level module names that can be imported.
                             Defaults to DEFAULT_ALLOWED_MODULES.
            timeout: Max execution time in seconds.
            max_memory_mb: Max virtual memory in MB (Linux only, best-effort on macOS).
        """
        self.allowed_dirs = [os.path.abspath(d) for d in (allowed_dirs or [])]
        self.allowed_modules = allowed_modules or DEFAULT_ALLOWED_MODULES
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb

    def run_file(self, script_path: str) -> SandboxResult:
        """Execute a .py file inside the sandbox."""
        script_path = os.path.abspath(script_path)
        if not os.path.isfile(script_path):
            raise FileNotFoundError(f"Script not found: {script_path}")

        with open(script_path, 'r') as f:
            code = f.read()

        return self._execute(code, script_path)

    def run_code(self, code: str, working_dir: str | None = None) -> SandboxResult:
        """Execute a code string inside the sandbox."""
        cwd = os.path.abspath(working_dir) if working_dir else os.getcwd()
        virtual_path = os.path.join(cwd, '<sandbox>')
        return self._execute(code, virtual_path, cwd=cwd)

    def _execute(self, code: str, script_path: str, cwd: str | None = None) -> SandboxResult:
        """Build wrapper, write to temp file, execute in subprocess."""
        if cwd is None:
            cwd = '.'

        # Script's directory is always accessible
        all_allowed_dirs = list({cwd, *self.allowed_dirs})

        wrapper = self._build_wrapper(code, script_path, all_allowed_dirs)

        fd, wrapper_path = tempfile.mkstemp(suffix='.py', prefix='_sandbox_')
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(wrapper)

            try:
                result = subprocess.run(
                    [sys.executable, '-u', wrapper_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=cwd,
                    env=self._make_env(),
                )
                return SandboxResult(
                    stdout=result.stdout,
                    stderr=result.stderr,
                    exit_code=result.returncode,
                    timed_out=False,
                )
            except subprocess.TimeoutExpired as e:
                return SandboxResult(
                    stdout=(e.stdout or b'').decode('utf-8', errors='replace'),
                    stderr=(e.stderr or b'').decode('utf-8', errors='replace')
                            + f"\n[Sandbox] Execution timed out after {self.timeout}s",
                    exit_code=-1,
                    timed_out=True,
                )
        finally:
            os.unlink(wrapper_path)

    def _make_env(self) -> dict[str, str]:
        """Create a clean environment for the subprocess."""
        env = os.environ.copy()
        # Force non-interactive matplotlib backend
        env['MPLBACKEND'] = 'Agg'
        return env

    def _build_wrapper(self, code: str, script_path: str, allowed_dirs: list[str]) -> str:
        """
        Build a Python wrapper script that:
        1. Sets resource limits (memory, CPU)
        2. Installs an import guard (whitelist)
        3. Restricts builtins.open to allowed directories
        4. Removes dangerous os/sys attributes
        5. Executes the user code
        """
        config = {
            'allowed_modules': sorted(self.allowed_modules),
            'allowed_dirs': allowed_dirs,
            'max_memory_mb': self.max_memory_mb,
            'script_path': script_path,
        }

        # The user code is passed as a JSON-encoded string to avoid
        # any escaping issues with triple quotes, backslashes, etc.
        code_json = json.dumps(code)

        return f'''\
import sys
import os
import json
import builtins

# ====================== Configuration ======================
_CONFIG = json.loads({json.dumps(json.dumps(config))})
_USER_CODE = json.loads({json.dumps(code_json)})

# ====================== Resource Limits ======================
def _set_resource_limits():
    try:
        import resource
    except ImportError:
        return  # Windows — skip resource limits

    max_mem = _CONFIG['max_memory_mb'] * 1024 * 1024

    # Memory limit (RLIMIT_AS on Linux, best-effort on macOS)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (max_mem, max_mem))
    except (ValueError, OSError):
        pass  # macOS may not enforce this

    # CPU time limit: generous but finite (timeout is the real guard)
    try:
        soft = 300  # 5 min CPU
        resource.setrlimit(resource.RLIMIT_CPU, (soft, soft + 5))
    except (ValueError, OSError):
        pass

_set_resource_limits()

# ====================== Import Guard ======================
_ALLOWED_MODULES = set(_CONFIG['allowed_modules'])

# Modules that are already loaded and needed by the sandbox itself
_ALWAYS_ALLOWED = {{'builtins', '_io', 'sys', 'os', 'json', 'resource',
                    'posix', 'nt', 'encodings', 'codecs', 'abc',
                    'importlib', '_frozen_importlib', '_frozen_importlib_external',
                    'marshal', 'io', '_signal', 'errno', 'stat',
                    '_stat', 'posixpath', 'ntpath', 'genericpath',
                    '_collections_abc', 'os.path', 'keyword', 'types',
                    '_abc', '_thread', '_weakref', 'warnings'}}

class _ImportGuard:
    """Meta-path finder that blocks imports of non-whitelisted modules."""

    def find_module(self, fullname, path=None):
        top = fullname.split('.')[0]
        if top in _ALLOWED_MODULES or fullname in _ALWAYS_ALLOWED:
            return None   # let the default mechanism handle it
        return self       # we will block it

    def load_module(self, fullname):
        raise ImportError(
            f"[Sandbox] Import of '{{fullname}}' is not allowed. "
            f"Allowed top-level modules: {{sorted(_ALLOWED_MODULES)}}"
        )

sys.meta_path.insert(0, _ImportGuard())

# ====================== File Access Guard ======================
_ALLOWED_DIRS = [os.path.abspath(d) for d in _CONFIG['allowed_dirs']]
_original_open = builtins.open

def _restricted_open(file, mode='r', *args, **kwargs):
    if isinstance(file, (str, bytes, os.PathLike)):
        filepath = os.path.abspath(str(file))
        if not any(filepath.startswith(d) for d in _ALLOWED_DIRS):
            raise PermissionError(
                f"[Sandbox] Access denied: {{filepath}}\\n"
                f"Allowed directories: {{_ALLOWED_DIRS}}"
            )
    return _original_open(file, mode, *args, **kwargs)

builtins.open = _restricted_open

# ====================== Disable Dangerous Functions ======================
_DANGEROUS_OS_ATTRS = [
    'system', 'popen', 'exec', 'execl', 'execle', 'execlp',
    'execlpe', 'execv', 'execve', 'execvp', 'execvpe',
    'spawn', 'spawnl', 'spawnle', 'spawnlp', 'spawnlpe',
    'spawnv', 'spawnve', 'spawnvp', 'spawnvpe',
    'fork', 'forkpty', 'kill', 'killpg',
    'putenv', 'unsetenv',
]

def _blocked(*args, **kwargs):
    raise PermissionError("[Sandbox] This operation is not allowed.")

for _attr in _DANGEROUS_OS_ATTRS:
    if hasattr(os, _attr):
        setattr(os, _attr, _blocked)

# Also block sys.exit so the script doesn't kill the sandbox wrapper
_original_exit = sys.exit
def _noop_exit(code=0):
    raise SystemExit(code)
sys.exit = _noop_exit

# ====================== Execute User Code ======================
try:
    _compiled = compile(_USER_CODE, _CONFIG['script_path'], 'exec')
    exec(_compiled, {{"__name__": "__main__", "__file__": _CONFIG['script_path']}})
except SystemExit as e:
    # Script called sys.exit() — propagate the exit code
    _original_exit(e.code if isinstance(e.code, int) else 1)
except Exception as e:
    import traceback
    traceback.print_exc()
    _original_exit(1)
'''


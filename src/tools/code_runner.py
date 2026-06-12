"""CodeRunner — 沙箱代码执行工具"""

import asyncio
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CodeRunResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    execution_time_ms: float = 0
    language: str = "python"


class CodeRunner:
    """安全的代码沙箱执行器

    支持两种模式:
    - subprocess: 本地子进程（默认），通过超时限制
    - docker: Docker 容器隔离（需 Docker 可用）
    """

    DEFAULT_TIMEOUT = 30  # 秒
    MAX_OUTPUT_BYTES = 64 * 1024  # 64KB 输出限制

    def __init__(self, mode: str = "subprocess", timeout: int = DEFAULT_TIMEOUT):
        self.mode = mode
        self.timeout = timeout

    async def run_python(self, code: str, stdin: str = "") -> CodeRunResult:
        """在沙箱中执行 Python 代码"""
        import time
        start = time.time()

        tmp_path = os.path.join(tempfile.gettempdir(), f"coderunner_{uuid.uuid4().hex[:8]}.py")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(code)

            if self.mode == "docker" and self._docker_available():
                return await self._run_docker("python", tmp_path, stdin, start)

            return await self._run_subprocess(["python", tmp_path], stdin, start)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    async def run_javascript(self, code: str, stdin: str = "") -> CodeRunResult:
        """在沙箱中执行 JavaScript 代码 (Node.js)"""
        import time
        start = time.time()

        tmp_path = os.path.join(tempfile.gettempdir(), f"coderunner_{uuid.uuid4().hex[:8]}.js")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(code)

            if self.mode == "docker" and self._docker_available():
                return await self._run_docker("node", tmp_path, stdin, start)

            return await self._run_subprocess(["node", tmp_path], stdin, start)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    async def _run_subprocess(self, cmd: list, stdin: str, start: float) -> CodeRunResult:
        import time

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(stdin.encode("utf-8") if stdin else None),
                timeout=self.timeout,
            )
            elapsed = (time.time() - start) * 1000

            stdout = stdout_bytes.decode("utf-8", errors="replace")[:self.MAX_OUTPUT_BYTES]
            stderr = stderr_bytes.decode("utf-8", errors="replace")[:self.MAX_OUTPUT_BYTES]

            return CodeRunResult(
                success=proc.returncode == 0,
                stdout=stdout,
                stderr=stderr,
                execution_time_ms=round(elapsed, 1),
                language="python" if "python" in cmd[0] else "javascript",
            )
        except asyncio.TimeoutError:
            elapsed = (time.time() - start) * 1000
            return CodeRunResult(
                success=False,
                error=f"执行超时 ({self.timeout}秒)",
                execution_time_ms=round(elapsed, 1),
            )
        except FileNotFoundError:
            return CodeRunResult(
                success=False,
                error=f"未找到运行时: {cmd[0]}",
            )
        except Exception as e:
            return CodeRunResult(
                success=False,
                error=str(e),
            )

    async def _run_docker(self, runtime: str, file_path: str, stdin: str, start: float) -> CodeRunResult:
        import time

        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "run", "--rm", "--network=none",
                "--memory=256m", "--cpus=0.5",
                "-v", f"{file_path}:/code/{os.path.basename(file_path)}:ro",
                f"{runtime}:slim",
                runtime, f"/code/{os.path.basename(file_path)}",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(stdin.encode("utf-8") if stdin else None),
                timeout=self.timeout,
            )
            elapsed = (time.time() - start) * 1000

            stdout = stdout_bytes.decode("utf-8", errors="replace")[:self.MAX_OUTPUT_BYTES]
            stderr = stderr_bytes.decode("utf-8", errors="replace")[:self.MAX_OUTPUT_BYTES]

            return CodeRunResult(
                success=proc.returncode == 0,
                stdout=stdout,
                stderr=stderr,
                execution_time_ms=round(elapsed, 1),
                language=os.path.basename(runtime),
            )
        except asyncio.TimeoutError:
            elapsed = (time.time() - start) * 1000
            return CodeRunResult(
                success=False,
                error=f"执行超时 ({self.timeout}秒)",
                execution_time_ms=round(elapsed, 1),
            )
        except Exception as e:
            return CodeRunResult(success=False, error=str(e))

    @staticmethod
    def _docker_available() -> bool:
        import shutil
        return shutil.which("docker") is not None

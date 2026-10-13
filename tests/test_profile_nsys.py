"""Run the actual PowerShell profiling wrapper against an argument-recording stub."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PWSH = shutil.which("pwsh")
CSC = Path(os.environ.get("WINDIR", "C:/Windows")) / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.fixture = tempfile.TemporaryDirectory(prefix="niyah-profile-test-")
        self.root = Path(self.fixture.name)
        self.stub = self.root / "nsys stub.ps1"
        self.stub.write_text(r'''$received = @($args)
ConvertTo-Json -InputObject $received -Compress | Add-Content -LiteralPath $env:NIYAH_STUB_LOG
if ($env:NIYAH_STUB_FAIL -eq $received[0]) { exit 9 }
if ($received[0] -eq 'profile' -and $env:NIYAH_STUB_FAIL -ne 'missing') {
    $position = [Array]::IndexOf($received, '-o')
    [System.IO.File]::WriteAllText(($received[$position + 1] + '.nsys-rep'), 'stub report')
}
exit 0
''', encoding="utf-8")
        if os.name == "nt" and CSC.is_file():
            source = self.root / "stub.cs"
            source.write_text(r'''using System;
using System.IO;
using System.Text;
class Stub {
    static string Quote(string value) {
        return "\"" + value.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";
    }
    static int Main(string[] args) {
        string[] quoted = Array.ConvertAll(args, Quote);
        File.AppendAllText(Environment.GetEnvironmentVariable("NIYAH_STUB_LOG"), "[" + string.Join(",", quoted) + "]\n", new UTF8Encoding(false));
        string failure = Environment.GetEnvironmentVariable("NIYAH_STUB_FAIL");
        if (failure == args[0]) return 9;
        if (args[0] == "profile" && failure != "missing") {
            File.WriteAllText(args[Array.IndexOf(args, "-o") + 1] + ".nsys-rep", "stub report");
        }
        return 0;
    }
}''', encoding="utf-8")
            self.stub = self.root / "native nsys stub.exe"
            compilation = subprocess.run([str(CSC), "/nologo", "/out:" + str(self.stub), str(source)],
                                         capture_output=True, text=True)
            self.assertEqual(compilation.returncode, 0, compilation.stdout + compilation.stderr)
        self.trainer = self.root / "trainer with space.exe"
        self.trainer.write_bytes(b"fixture, not executed")
        self.log = self.root / "calls.jsonl"
        self.runner = self.root / "run.ps1"
        def literal(value):
            return "'" + str(value).replace("'", "''") + "'"
        self.runner.write_text("& " + literal(ROOT / "tools" / "profile_nsys.ps1") +
                               " -Nsys " + literal(self.stub) + " -Exe " + literal(self.trainer) +
                               " -OutputDirectory " + literal(self.root / "reports with space") +
                               " -TrainerArguments @('--prompt', 'مرحبا world', '--empty', '', '--value', 'a;b&c')\n",
                               encoding="utf-8")

    def tearDown(self):
        self.fixture.cleanup()

    def run_wrapper(self, failure=""):
        env = dict(os.environ, NIYAH_STUB_LOG=str(self.log), NIYAH_STUB_FAIL=failure)
        result = subprocess.run([PWSH, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(self.runner)], env=env,
                                capture_output=True, text=True, encoding="utf-8")
        self.assertTrue(self.log.exists(), result.stdout + result.stderr)
        calls = [json.loads(line) for line in self.log.read_text(encoding="utf-8-sig").splitlines()]
        return result, calls

    def test_forwarding_unique_paths_and_checked_reports(self):
        result, calls = self.run_wrapper()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(calls), 3)
        profile = calls[0]
        self.assertEqual(profile[profile.index("--force-overwrite") + 1], "false")
        self.assertEqual(profile[-7:], [str(self.trainer), "--prompt", "مرحبا world", "--empty", "", "--value", "a;b&c"])
        first_report = profile[profile.index("-o") + 1]
        self.assertEqual(calls[1], ["stats", "--report", "cuda_gpu_kern_sum", first_report + ".nsys-rep"])
        self.assertEqual(calls[2], ["stats", "--report", "cuda_gpu_mem_time_sum", first_report + ".nsys-rep"])
        result, all_calls = self.run_wrapper()
        self.assertEqual(result.returncode, 0, result.stderr)
        second_profile = all_calls[3]
        self.assertNotEqual(first_report, second_profile[second_profile.index("-o") + 1])

    def test_profile_failure_stops_before_stats(self):
        result, calls = self.run_wrapper("profile")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("NSYS_PROFILE_FAILED: 9", result.stderr)
        self.assertEqual(len(calls), 1)

    def test_stats_failure_stops_at_first_failure(self):
        result, calls = self.run_wrapper("stats")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("NSYS_STATS_FAILED", result.stderr)
        self.assertEqual(len(calls), 2)

    def test_missing_report_rejected(self):
        result, calls = self.run_wrapper("missing")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("NSYS_REPORT_MISSING", result.stderr)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    if not PWSH:
        print("UNAVAILABLE: PowerShell pwsh is required for wrapper execution", file=sys.stderr)
        sys.exit(77)
    unittest.main(verbosity=2)

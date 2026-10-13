"""Exercise delayed stdio failures using an inherited read-only output handle."""
import pathlib
import subprocess
import sys
import tempfile
import re
import ast


def run(command, **kwargs):
    return subprocess.run(command, stderr=subprocess.PIPE, **kwargs)


def main():
    cli, probe, fixture, train = sys.argv[1:]
    with tempfile.TemporaryDirectory(prefix="niyah-stdout-") as temporary:
        d = pathlib.Path(temporary)
        tok, shard, alternate, ckpt, cursor = [str(d / name) for name in
            ("tok.bin", "shard.bin", "alt.bin", "model.ckpt", "state.cur")]
        result = run([fixture, tok, shard, alternate], stdout=subprocess.PIPE)
        assert result.returncode == 0, result.stderr
        result = run([train, "new", "--tokenizer", tok, "--shard", shard,
            "--checkpoint-out", ckpt, "--cursor-out", cursor,
            "--updates", "1", "--batch-size", "1", "--accumulation-steps", "1",
            "--model-seed", "42", "--data-seed", "7", "--context-length", "4",
            "--embedding-dim", "8", "--layers", "1", "--heads", "2", "--kv-heads", "1",
            "--ffn-hidden-dim", "16", "--rms-norm-eps", "0.00001", "--tie-word-embeddings", "1",
            "--learning-rate", "0.001", "--beta1", "0.9", "--beta2", "0.999", "--epsilon", "0.00000001",
            "--weight-decay", "0", "--max-grad-norm", "1"], stdout=subprocess.PIPE)
        assert result.returncode == 0, result.stderr
        sink = d / "read-only-output.bin"
        sink.write_bytes(b"preserve")
        commands = [
            [cli, "run", "--tokenizer", tok, "--checkpoint", ckpt,
             "--prompt", "a", "--max-new-tokens", "1", "--temperature", "0"],
            [probe, "--tokenizer", tok, "--checkpoint", ckpt, "--prompt", "a", "--topk", "1"],
        ]
        # Probe uses native tokenizer decoding with escaped bytes. A single
        # greedy generation must emit exactly those bytes plus one LF trailer,
        # including on Windows where default CRT output expands LF to CRLF.
        generated = run(commands[0], stdout=subprocess.PIPE)
        decoded = run(commands[1], stdout=subprocess.PIPE)
        assert generated.returncode == decoded.returncode == 0
        match = re.search(rb'rank=1[^\r\n]* bytes=("[^\r\n]*")', decoded.stdout)
        assert match, decoded.stdout
        token_bytes = ast.literal_eval("b" + match.group(1).decode("ascii"))
        assert generated.stdout == token_bytes + b"\n", (generated.stdout, token_bytes)
        for command in commands:
            with sink.open("rb") as output:
                result = run(command, stdout=output)
            assert result.returncode != 0, "read-only stdout incorrectly succeeded"
            assert b"stdout_write" in result.stderr, result.stderr
            assert sink.read_bytes() == b"preserve"
        print("STDOUT_FAILURE=PASS")


if __name__ == "__main__":
    main()

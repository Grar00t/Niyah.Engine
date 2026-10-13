"""Native UTF-8 paths: prepare, read, publication, templates and toy resume."""
import hashlib
import pathlib
import subprocess
import sys
import tempfile


def invoke(command, success=True):
    result = subprocess.run([str(x) for x in command], capture_output=True)
    assert (result.returncode == 0) == success, (command, result.returncode, result.stderr)
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cli, train, probe = sys.argv[1:]
    with tempfile.TemporaryDirectory(prefix="niyah-utf8-") as temporary:
        d = pathlib.Path(temporary) / "\u062a\u062c\u0631\u0628\u0629-\u03a9"
        d.mkdir()
        corpus = d / "\u0646\u0635.txt"
        corpus.write_bytes("\u0627\u0628\u062c abc abc".encode("utf8"))
        tok, shard, ckpt, cursor = [d / name for name in
            ("\u0631\u0645\u0648\u0632.bin", "\u0628\u064a\u0627\u0646.bin", "\u0646\u0645\u0648\u0630\u062c.ckpt", "\u0645\u0624\u0634\u0631.cur")]
        prepare = [cli, "prepare", "--corpus", corpus, "--tokenizer-out", tok,
            "--shard-out", shard, "--target-vocab", "258", "--min-pair-frequency", "2", "--sequence-length", "4"]
        invoke(prepare)
        before = {str(p): digest(p) for p in (tok, shard)}
        invoke(prepare, success=False)
        alias = prepare.copy()
        alias[alias.index("--tokenizer-out") + 1] = str(d) + "/./" + tok.name
        invoke(alias, success=False)
        assert all(digest(pathlib.Path(p)) == h for p, h in before.items())
        invoke([train, "new", "--tokenizer", tok, "--shard", shard,
            "--checkpoint-out", ckpt, "--cursor-out", cursor,
            "--updates", "1", "--batch-size", "1", "--accumulation-steps", "1",
            "--model-seed", "42", "--data-seed", "7", "--context-length", "4",
            "--embedding-dim", "8", "--layers", "1", "--heads", "2", "--kv-heads", "1",
            "--ffn-hidden-dim", "16", "--rms-norm-eps", "0.00001", "--tie-word-embeddings", "1",
            "--learning-rate", "0.001", "--beta1", "0.9", "--beta2", "0.999", "--epsilon", "0.00000001",
            "--weight-decay", "0", "--max-grad-norm", "1"])
        assert pathlib.Path(str(ckpt) + ".complete").is_file()
        for suffix in (".prompt-prefix", ".prompt-suffix"):
            pathlib.Path(str(ckpt) + suffix).write_bytes(b"")
        empty = hashlib.sha256(b"").hexdigest()
        pathlib.Path(str(ckpt) + ".prompt-sha256").write_text(
            f"NIYAH_PROMPT_V2\n{digest(ckpt)}\n{empty}\n{empty}\n", encoding="ascii")
        invoke([cli, "run", "--tokenizer", tok, "--checkpoint", ckpt,
            "--prompt", "a", "--max-new-tokens", "1", "--temperature", "0"])
        invoke([probe, "--tokenizer", tok, "--shard", shard, "--checkpoint", ckpt,
            "--prompt", "a", "--topk", "1", "--trace-steps", "1"])
        invoke([cli, "eval", "--tokenizer", tok, "--checkpoint", ckpt,
            "--heldout", shard, "--format", "shard"])
        pair_before = [digest(ckpt), digest(cursor)]
        result = invoke([train, "resume", "--tokenizer", tok, "--shard", shard,
            "--checkpoint-in", ckpt, "--cursor-in", cursor,
            "--checkpoint-out", d / "next.ckpt", "--cursor-out", d / "next.cur",
            "--updates", "1", "--batch-size", "1", "--accumulation-steps", "1"])
        assert b"optimizer_step=2" in result.stdout
        assert [digest(ckpt), digest(cursor)] == pair_before
        assert not list(d.glob("*.niyah-stage-*"))
        print("UNICODE_PATHS=PASS")


if __name__ == "__main__":
    main()

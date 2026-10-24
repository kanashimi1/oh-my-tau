"""Type text into a client and print the screen (autocomplete popup checks)."""
import shutil, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from driver import Term
import mock_provider, run

binary, text = sys.argv[1], sys.argv[2]
srv = mock_provider.serve(0)
root = Path(tempfile.mkdtemp(prefix="e2e-home-"))
env = run.make_home(root, srv.server_address[1])
(Path(env["E2E_WORK"]) / "src").mkdir()
(Path(env["E2E_WORK"]) / "src" / "main.rs").write_text("fn main() {}\n")
t = Term([binary, "--model", "mock/text"], env=env, cwd=env["E2E_WORK"])
try:
    run.ready(t)
    t.type(text, per_char=0.03)
    t.settle(1.2, 10)
    print("\n".join(l for l in t.screen() if l.strip()))
finally:
    t.close()
    run.reap_home_processes(root)
    shutil.rmtree(root, ignore_errors=True)
    srv.shutdown()

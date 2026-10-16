"""Debug helper: start a client in the e2e home and dump raw output + screen."""
import shutil, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from driver import Term
import mock_provider, run

binary = sys.argv[1] if len(sys.argv) > 1 else shutil.which("omp")
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 20
srv = mock_provider.serve(0)
root = Path(tempfile.mkdtemp(prefix="e2e-home-"))
env = run.make_home(root, srv.server_address[1])
t = Term([binary, "--model", "mock/text"], env=env, cwd=env["E2E_WORK"])
start = time.time()
t.pump(secs)
print(f"bytes={len(t.raw)} alive={t.proc.poll() is None} t={time.time()-start:.1f}s")
print("RAW tail:", bytes(t.raw[-1500:]))
print("=== screen ===")
print(t.text())
t.close()
shutil.rmtree(root, ignore_errors=True)
srv.shutdown()

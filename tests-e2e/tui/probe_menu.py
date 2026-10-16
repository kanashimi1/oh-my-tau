"""Open a slash menu in a client and report its screen and terminal modes
(alternate screen, mouse tracking) — debug helper for overlay parity."""
import shutil, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from driver import Term
import mock_provider, run

binary, command = sys.argv[1], sys.argv[2]
srv = mock_provider.serve(0)
root = Path(tempfile.mkdtemp(prefix="e2e-home-"))
env = run.make_home(root, srv.server_address[1])
t = Term([binary, "--model", "mock/text"], env=env, cwd=env["E2E_WORK"])
try:
    run.ready(t)
    mark = len(t.raw)
    run.submit(t, command)
    t.settle(1.5, 15)
    raw = bytes(t.raw[mark:])
    modes = {name: seq in raw for name, seq in {
        "alt_screen_on(1049h)": b"\x1b[?1049h",
        "mouse_click(1000h)": b"\x1b[?1000h",
        "mouse_motion(1003h)": b"\x1b[?1003h",
        "mouse_sgr(1006h)": b"\x1b[?1006h",
    }.items()}
    print("MODES", modes)
    nonblank = [l for l in t.screen() if l.strip()]
    print(f"SCREEN rows used: {len(nonblank)} of {t.rows}")
    print(t.text())
finally:
    t.close()
    run.reap_home_processes(root)
    shutil.rmtree(root, ignore_errors=True)
    srv.shutdown()

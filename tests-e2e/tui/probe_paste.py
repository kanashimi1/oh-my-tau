"""Send paste chords/bytes and print the composer (paste wiring probe)."""
import shutil, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from driver import Term
import mock_provider, run

binary, mode = sys.argv[1], sys.argv[2]
srv = mock_provider.serve(0)
root = Path(tempfile.mkdtemp(prefix="e2e-home-"))
env = run.make_home(root, srv.server_address[1])
env["OMP_CLIPBOARD_TEXT_GET"] = "printf stub-clipboard-text"
env["OMP_CLIPBOARD_IMAGE_GET"] = "printf PNGSTUB"
t = Term([binary, "--model", "mock/text"], env=env, cwd=env["E2E_WORK"])
try:
    run.ready(t)
    if mode == "csi-u":
        t.send(b"\x1b[118;6u")
    elif mode == "legacy-ctrlv":
        t.send(b"\x16")
    elif mode == "csi-u2":
        t.send(b"\x1b[118;6:1u")   # kitty with event-type field
    elif mode == "csi-u3":
        t.send(b"\x1b[27;6;118~")  # xterm modifyOtherKeys
    elif mode == "ctrl-a":
        t.send(b"\x01")
    elif mode == "bracketed":
        t.send("\x1b[200~pasted text\x1b[201~")
    t.pump(2.0)
    if mode.endswith("-wake"):
        mode = mode[:-5]
        t.type("x")
        t.pump(1.0)
    print("\n".join(l for l in t.screen() if l.strip()))
    print("\n".join(l for l in t.screen() if l.strip()))
finally:
    t.close()
    run.reap_home_processes(root)
    shutil.rmtree(root, ignore_errors=True)
    srv.shutdown()

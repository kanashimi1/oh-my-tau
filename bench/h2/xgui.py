"""Simulated Minecraft GUI behind the mcbridge HTTP contract + the `xeval.py gui` episode.

The simulator answers /screen, /click, /key, /state exactly in the shape of
mcp-mc/bridge CoreEndpoints (screen class + title + AbstractWidget list; click by x,y or text;
errors as HTTP 500 {"ok":false,"error"}). Screens model 1.20.1 menus with two real-world
dynamics: delayed screens (resource reload, world loading) and a pop-up that replaces the
screen after an option change. The agent reaches it only through the real mcp_mc.py MCP server.
"""
import json, os, re, shutil, sys, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
MCP_MC = os.path.expanduser("~/fleet-split/mcp-mc/server/mcp_mc.py")
SKILL_PATH = os.path.expanduser("~/fleet-split/mcp-mc/skills/mc-gui/SKILL.md")
B = lambda text, x, y, w=200, h=20, t="Button", active=True: {"type": t, "text": text, "x": x, "y": y, "w": w, "h": h, "active": active, "visible": True}
FOVS = [70, 90, 110, 30, 50]
RDS = [8, 12, 16, 24, 32, 2, 4]
DEFAULT_BINDINGS = {"key.jump": "Space", "key.sprint": "Left Control", "key.inventory": "E"}
LANGS = [("English (US)", "en_us"), ("Deutsch", "de_de"), ("Fran\u00e7ais", "fr_fr"), ("Espa\u00f1ol", "es_es"), ("\u65e5\u672c\u8a9e", "ja_jp")]


class Sim:
    def __init__(self):
        self.lock = threading.Lock()
        self.graphics, self.clouds, self.rd, self.fov = "Fancy", "Fancy", 8, 70
        self.packs_selected, self.packs_pending, self.highlight = [], [], None
        self.bindings = dict(DEFAULT_BINDINGS)
        self.listening = None  # key-bind row waiting for the next key press
        self.language, self.lang_pending = "en_us", None
        self.in_world, self.screen, self.stack = False, "title", []
        self.pending = None  # (deadline, fn): delayed screen change
        self.clicks = []     # every click: {screen, x, y, hit, transient}
        self.t0 = time.monotonic()

    # ---- dynamics
    def _tick(self):
        if self.pending and time.monotonic() >= self.pending[0]:
            fn = self.pending[1]; self.pending = None; fn()

    def _later(self, secs, fn):
        self.pending = (time.monotonic() + secs, fn)

    def _open(self, name):
        self.stack.append(self.screen); self.screen = name

    def _back(self):
        self.screen = self.stack.pop() if self.stack else (None if self.in_world else "title")

    # ---- views (GUI-scaled 427x240, like a 1280x720 window at scale 3)
    def widgets(self):
        s = self.screen
        if s == "title":
            return ("net.minecraft.client.gui.screens.TitleScreen", "Minecraft",
                    [B("Singleplayer", 113, 90), B("Multiplayer", 113, 114), B("Minecraft Realms", 113, 138),
                     B("Options...", 113, 168, 98), B("Quit Game", 215, 168, 98)])
        if s == "options":
            fov = "Quake Pro" if self.fov == 110 else ("Normal" if self.fov == 70 else str(self.fov))
            return ("net.minecraft.client.gui.screens.OptionsScreen", "Options",
                    [B(f"FOV: {fov}", 60, 40, 150, t="OptionSlider"), B("Online...", 217, 40, 150),
                     B("Video Settings...", 60, 88, 150), B("Controls...", 217, 88, 150),
                     B("Resource Packs...", 60, 112, 150), B("Language...", 217, 112, 150), B("Done", 113, 200)])
        if s == "video":
            return ("net.minecraft.client.gui.screens.VideoSettingsScreen", "Video Settings",
                    [B(f"Graphics: {self.graphics}", 60, 40, 150, t="CycleButton"), B(f"Render Distance: {self.rd} chunks", 217, 40, 150, t="OptionSlider"),
                     B(f"Clouds: {self.clouds}", 60, 64, 150, t="CycleButton"), B("Max Framerate: 120 fps", 217, 64, 150, t="OptionSlider"),
                     B("Done", 113, 200)])
        if s == "gpu_warning":
            return ("net.minecraft.client.gui.screens.PopupScreen", "Graphics Warning",
                    [B("Cancel", 60, 64, 150), B("I Understand", 217, 64, 150)])
        if s == "packs":
            avail = [p for p in ["Programmer Art", "High Contrast"] if p not in self.packs_pending]
            ws = [B(("> " if self.highlight == p else "") + p, 20, 40 + 24 * i, 180, t="PackEntry") for i, p in enumerate(avail)]
            ws += [B(p, 227, 40 + 24 * i, 180, t="PackEntry") for i, p in enumerate(self.packs_pending)]
            ws += [B("Move to Selected >", 113, 160, active=self.highlight is not None), B("Done", 113, 200)]
            return ("net.minecraft.client.gui.screens.packs.PackSelectionScreen", "Select Resource Packs", ws)
        if s == "controls":
            return ("net.minecraft.client.gui.screens.options.controls.ControlsScreen", "Controls",
                    [B("Key Binds...", 60, 40, 150), B("Auto-Jump: OFF", 217, 40, 150, t="CycleButton"), B("Done", 113, 200)])
        if s == "keybinds":
            ws = [B(f"{cat.split('.')[1].capitalize()}: {'> ? <' if self.listening == cat else self.bindings[cat]}",
                    20, 40 + 24 * i, 240, t="KeyBind") for i, cat in enumerate(sorted(self.bindings))]
            ws += [B("Reset", 113, 180, 150), B("Done", 113, 200)]
            return ("net.minecraft.client.gui.screens.options.controls.KeyBindsScreen", "Key Binds", ws)
        if s == "language":
            cur = self.lang_pending or self.language
            ws = [B(("> " if code == cur else "") + name, 60, 40 + 22 * i, 220, t="LanguageEntry") for i, (name, code) in enumerate(LANGS)]
            ws += [B("Done", 113, 200)]
            return ("net.minecraft.client.gui.screens.LanguageScreen", "Language", ws)
        if s == "reload":
            return ("net.minecraft.client.gui.screens.LoadingOverlay", "", [])
        if s == "worlds":
            return ("net.minecraft.client.gui.screens.worldselection.SelectWorldScreen", "Select World",
                    [B(("> " if self.highlight == "Test World" else "") + "Test World", 63, 40, 300, 36, t="WorldEntry"),
                     B("Play Selected World", 63, 180, 150, active=self.highlight == "Test World"), B("Create New World", 215, 180, 150),
                     B("Delete", 63, 204, 72), B("Cancel", 215, 204, 150)])
        if s == "loading":
            return ("net.minecraft.client.gui.screens.ReceivingLevelScreen", "Loading terrain...", [])
        if s == "pause":
            return ("net.minecraft.client.gui.screens.PauseScreen", "Game Menu",
                    [B("Back to Game", 113, 60), B("Advancements", 113, 84, 98), B("Statistics", 215, 84, 98),
                     B("Options...", 113, 132, 98), B("Open to LAN", 215, 132, 98), B("Save and Quit to Title", 113, 156)])
        return (None, None, [])

    def screen_json(self):
        with self.lock:
            self._tick(); cls, title, ws = self.widgets()
            if cls is None:
                return {"screen": None}
            return {"screen": cls, "title": title, "widgets": ws}

    def click(self, x, y, text=None, button=0):
        with self.lock:
            self._tick(); cls, title, ws = self.widgets()
            if cls is None:
                raise ValueError("no screen open")
            hit = None
            if text is not None:
                hit = next((w for w in ws if text.lower() in w["text"].lower()), None)
                if hit is None:
                    raise ValueError(f"no widget with text '{text}'")
                x, y = hit["x"] + hit["w"] / 2, hit["y"] + hit["h"] / 2
            else:
                hit = next((w for w in ws if w["x"] <= x < w["x"] + w["w"] and w["y"] <= y < w["y"] + w["h"]), None)
            handled = bool(hit and hit["active"])
            self.clicks.append({"t": round(time.monotonic() - self.t0, 2), "screen": title, "x": x, "y": y, "hit": hit["text"] if hit else None, "handled": handled})
            if handled:
                self._press(hit["text"])
            return {"handled": handled, "x": round(x, 1), "y": round(y, 1)}

    def _press(self, t):
        s = self.screen
        if s == "title":
            if t == "Options...": self._open("options")
            elif t == "Singleplayer": self.highlight = None; self._open("worlds")
        elif s == "options":
            if t.startswith("FOV"): self.fov = FOVS[(FOVS.index(self.fov) + 1) % len(FOVS)]
            elif t == "Video Settings...": self._open("video")
            elif t == "Resource Packs...": self.highlight = None; self.packs_pending = list(self.packs_selected); self._open("packs")
            elif t == "Controls...": self._open("controls")
            elif t == "Language...": self.lang_pending = None; self._open("language")
            elif t == "Done": self._back()
        elif s == "video":
            if t.startswith("Graphics"):
                order = ["Fast", "Fancy", "Fabulous!"]; self.graphics = order[(order.index(self.graphics) + 1) % 3]
                if self.graphics == "Fabulous!":
                    self._later(0.8, lambda: self._open("gpu_warning") if self.screen == "video" and self.graphics == "Fabulous!" else None)
            elif t.startswith("Clouds"):
                order = ["Fancy", "Fast", "OFF"]; self.clouds = order[(order.index(self.clouds) + 1) % 3]
            elif t.startswith("Render Distance"): self.rd = RDS[(RDS.index(self.rd) + 1) % len(RDS)]
            elif t == "Done": self.pending = None; self._back()
        elif s == "gpu_warning":
            if t == "Cancel": self.graphics = "Fancy"
            self._back()
        elif s == "packs":
            name = t[2:] if t.startswith("> ") else t
            if name in ("Programmer Art", "High Contrast") and name not in self.packs_pending: self.highlight = name
            elif t == "Move to Selected >" and self.highlight: self.packs_pending.append(self.highlight); self.highlight = None
            elif t == "Done":
                changed = self.packs_pending != self.packs_selected
                self._back()
                if changed:
                    back_to = self.screen; self.screen = "reload"
                    def done(): self.packs_selected = list(self.packs_pending); self.screen = back_to
                    self._later(2.5, done)
        elif s == "controls":
            if t == "Key Binds...": self.listening = None; self._open("keybinds")
            elif t == "Done": self._back()
        elif s == "keybinds":
            name = t.split(":")[0].strip()
            cat = {"Jump": "key.jump", "Sprint": "key.sprint", "Inventory": "key.inventory"}.get(name)
            if cat: self.listening = cat
            elif t == "Reset": self.bindings = dict(DEFAULT_BINDINGS); self.listening = None
            elif t == "Done": self.listening = None; self._back()
        elif s == "language":
            name = t[2:] if t.startswith("> ") else t
            code = next((c for n, c in LANGS if n == name), None)
            if code: self.lang_pending = code
            elif t == "Done":
                if self.lang_pending: self.language = self.lang_pending
                self.lang_pending = None; self._back()
        elif s == "worlds":
            name = t[2:] if t.startswith("> ") else t
            if name == "Test World": self.highlight = "Test World"
            elif t == "Play Selected World":
                self.stack = []; self.screen = "loading"
                def enter(): self.in_world = True; self.screen = None
                self._later(3.0, enter)
            elif t == "Cancel": self._back()
        elif s == "pause":
            if t == "Back to Game": self.stack = []; self.screen = None
            elif t == "Options...": self._open("options")

    def key(self, name):
        with self.lock:
            self._tick()
            if self.listening and name.lower() not in ("escape", "esc"):
                k = str(name).strip()
                self.bindings[self.listening] = k.upper() if len(k) == 1 else k
                self.listening = None
                return {"key": name, "action": "tap"}
            if name.lower() in ("escape", "esc"):
                if self.listening: self.listening = None  # real MC: esc cancels listen-in, screen stays
                elif self.screen in ("reload", "loading"): pass
                elif self.screen is None and self.in_world: self.screen = "pause"; self.stack = []
                elif self.screen == "pause": self.screen = None
                elif self.screen not in (None, "title"): self._back()
            return {"key": name, "action": "tap"}

    def state(self):
        with self.lock:
            self._tick(); cls, title, _ = self.widgets()
            return {"inWorld": self.in_world, "screen": cls}


def serve(sim):
    class Hd(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0); body = json.loads(self.rfile.read(n) or b"{}")
            try:
                if self.path == "/screen": out = sim.screen_json()
                elif self.path == "/click": out = sim.click(body.get("x", 0), body.get("y", 0), body.get("text"), body.get("button", 0))
                elif self.path == "/key": out = sim.key(body.get("key", ""))
                elif self.path == "/state": out = sim.state()
                else: raise ValueError(f"unknown route {self.path}")
                out = {"ok": True, **out}; code = 200
            except Exception as e:  # noqa: BLE001  the bridge answers 500 + error text
                out = {"ok": False, "error": str(e)}; code = 500
            data = json.dumps(out).encode(); self.send_response(code)
            self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        do_GET = do_POST
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Hd)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


TASKS = {
    "g_video": ("The Minecraft client is on the title screen. Using the mc tools only, set Graphics to Fabulous!, "
                "Clouds to OFF and Render Distance to 16 chunks, then return to the title screen. Verify the final values before you finish.",
                lambda s: {"graphics": s.graphics == "Fabulous!", "clouds": s.clouds == "OFF", "rd16": s.rd == 16, "title": s.screen == "title"}),
    "g_packs": ("The Minecraft client is on the title screen. Using the mc tools only, enable the 'Programmer Art' resource pack "
                "(it must end up selected and applied) and nothing else, then return to the title screen. Verify before you finish.",
                lambda s: {"applied": s.packs_selected == ["Programmer Art"], "title": s.screen == "title"}),
    "g_keys": ("The Minecraft client is on the title screen. Using the mc tools only, rebind the 'Jump' key to B "
               "in the controls options, then return to the title screen. Verify the new binding before you finish.",
               lambda s: {"bound": s.bindings["key.jump"] == "B", "title": s.screen == "title"}),
    "g_lang": ("The Minecraft client is on the title screen. Using the mc tools only, switch the game language to Deutsch "
               "and return to the title screen. Verify the selected language before you finish.",
               lambda s: {"lang": s.language == "de_de", "title": s.screen == "title"}),
    "g_fov": ("The Minecraft client is on the title screen. Using the mc tools only, load the singleplayer world 'Test World', "
              "then in game set FOV to Quake Pro and return to the game with no menu open. Verify before you finish.",
              lambda s: {"in_world": s.in_world, "fov": s.fov == 110, "no_menu": s.screen is None and s.in_world}),
}
ARM_TOOLS = {"base": ["mc_screen", "mc_click", "mc_key", "mc_state"], "hcu": ["mc_ui", "mc_act", "mc_state"],
             "hcu_skill": ["mc_ui", "mc_act", "mc_state"]}


def gui_one(arm, tid, rep, a):
    import xeval as X, h2
    sim = Sim(); httpd = serve(sim); ws = tempfile.mkdtemp(prefix=f"xg-{tid}-")
    proj = os.path.join(ws, ".projects.json")
    json.dump({"sim": {"dir": ws, "rcon_port": 1, "world": "w", "bridge_port": httpd.server_address[1]}}, open(proj, "w"))
    # the session HOME is a temp dir: point the MCP server back at the real user site-packages (fastmcp)
    json.dump({"mcpServers": {"sim": {"command": sys.executable, "args": [MCP_MC], "env": {"MCP_MC_PROJECTS": proj, "PYTHONUSERBASE": os.path.expanduser("~/.local")}}}}, open(os.path.join(ws, "mcp.json"), "w"))
    tools = ",".join(a.mcp_prefix + t for t in ARM_TOOLS[arm])
    extra = ["--tools", tools, "--max-time", "8m"]
    if arm == "hcu_skill":
        raw = open(SKILL_PATH).read()
        m = re.match(r"\A---\n.*?\n---\n", raw, re.S)
        extra += ["--append-system-prompt", "You are driving a Minecraft GUI with the mc tools. Skill mc-gui (follow strictly):\n" + (raw[m.end():] if m else raw).strip()]
    s = X.Sess(ws, a.model, extra)
    try:
        r = s.prompt(TASKS[tid][0], timeout=500)
    finally:
        s.close(); httpd.shutdown()
    checks = TASKS[tid][1](sim); shutil.rmtree(ws, ignore_errors=True)
    tdir = os.path.join(HERE, "results", "x-gui-traces"); os.makedirs(tdir, exist_ok=True)
    json.dump({"steps": r["steps"], "final": r["final"], "clicks": sim.clicks},
              open(os.path.join(tdir, f"{a.label}-{arm}-{tid}-{rep}.json"), "w"), ensure_ascii=False, indent=0)
    used = [x["tool"] for x in r["steps"]]
    return {"arm": arm, "task": tid, "rep": rep, "model": a.model, "status": r["status"], "score": round(sum(checks.values()) / len(checks), 3),
            "success": all(checks.values()), "checks": checks, "calls": len(used), "tools": {t: used.count(t) for t in set(used)},
            "errors": sum(x["error"] for x in r["steps"]), "clicks": len(sim.clicks), "dead_clicks": sum(1 for c in sim.clicks if not c["handled"]),
            "popup_cancel": sum(1 for c in sim.clicks if c["hit"] == "Cancel" and c["screen"] == "Graphics Warning"), "tokens": X.tok(r["usage"])}

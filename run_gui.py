"""Local browser GUI for the lab incident model.

    python run_gui.py            # opens http://127.0.0.1:8090
    python run_gui.py 8091 --no-browser

The browser only sends commands and draws: every rule runs in the Python/Mesa
model, exactly as in the batch experiments.
"""

import argparse
import json
import secrets
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agents import Occupant, Responder
from environment import FLOOR_PLAN
from model import LabIncidentModel
from options import OPTION_NAMES
from simulation_config import DEFAULT_SEED, PRESETS

ROOT = Path(__file__).resolve().parent
STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/style.css": ("style.css", "text/css; charset=utf-8")}
MAX_SESSIONS, IDLE_SECONDS = 16, 3600


def snapshot(m):
    env = m.env
    people = []
    for a in m.agents:
        if isinstance(a, Responder) and a.pos is not None:
            people.append({"t": "R", "x": a.pos[0], "y": a.pos[1], "id": a.unique_id,
                           "charges": a.charges, "task": a.task,
                           "option": "refilling" if a.refilling else "to station" if a.charges == 0 else
                           OPTION_NAMES[a.option] if a.option is not None else "idle"})
        elif isinstance(a, Occupant) and a.pos is not None:
            people.append({"t": "O", "x": a.pos[0], "y": a.pos[1], "s": a.state[0]})
    hist = m.datacollector.get_model_vars_dataframe()
    return {
        "step": m.steps, "running": m.running, "alarm": m.alarm_on,
        "floorPlan": env.floor_plan,
        "fire": [[int(v * 100) for v in col] for col in env.fire],
        "smoke": [[int(v * 100) for v in col] for col in env.smoke],
        "burned": [[int(v) for v in col] for col in env.ever_burned],
        "closedCells": m.closed_cells,
        "queueWait": m.queue_wait.tolist(),
        "robotGuidance": m.robot_guidance,
        "robots": [{"x": r.location[0], "y": r.location[1],
                    "west": r.assigned["west"], "east": r.assigned["east"]} for r in m.robots],
        "people": people,
        "detectors": [{"x": d.location[0], "y": d.location[1], "on": d.triggered} for d in m.detectors],
        "messages": [{"step": s, "p": p, "from": a, "to": b} for s, p, a, b in m.bus.log[-14:]],
        "contracts": {str(r.unique_id): r.task for r in m.responder_list if r.task is not None},
        "history": {k: hist[k].tolist() for k in ("Burning", "BurnedArea", "Evacuated", "Exposure", "Messages")},
        "summary": m.summary(),
        "coordination": m.coordination,
    }


class Store:
    def __init__(self):
        self.lock = threading.Lock()
        self.sessions = {}

    def reset(self, preset, seed):
        if preset not in PRESETS:
            raise ValueError("unknown preset")
        model = LabIncidentModel(seed=int(seed), **PRESETS[preset]["params"])
        with self.lock:
            now = time.time()
            for k in [k for k, (_, t) in self.sessions.items() if now - t > IDLE_SECONDS]:
                del self.sessions[k]
            while len(self.sessions) >= MAX_SESSIONS:
                del self.sessions[min(self.sessions, key=lambda k: self.sessions[k][1])]
            sid = secrets.token_urlsafe(16)
            self.sessions[sid] = (model, now)
        return sid, model

    def step(self, sid, n):
        with self.lock:
            if sid not in self.sessions:
                raise KeyError
            model, _ = self.sessions[sid]
            self.sessions[sid] = (model, time.time())
            for _ in range(max(1, min(int(n), 20))):
                if model.running:
                    model.step()
            return model


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, status, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/config":
            self.reply(200, {"defaultSeed": DEFAULT_SEED, "floorPlan": FLOOR_PLAN,
                             "presets": [{"id": k, "name": v["name"]} for k, v in PRESETS.items()]})
        elif path in STATIC:
            name, ctype = STATIC[path]
            self.reply(200, (ROOT / name).read_bytes(), ctype)
        else:
            self.reply(404, {"error": "not found"})

    def do_POST(self):
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 2048:
                raise ValueError("bad request size")
            body = json.loads(self.rfile.read(size))
            if self.path == "/api/reset":
                sid, model = self.server.store.reset(body.get("preset", "dispatcher"),
                                                     body.get("seed", DEFAULT_SEED))
            elif self.path == "/api/step":
                sid = body["session"]
                model = self.server.store.step(sid, body.get("n", 1))
            else:
                return self.reply(404, {"error": "unknown command"})
            state = snapshot(model)
            state["session"] = sid
            self.reply(200, state)
        except KeyError:
            self.reply(410, {"error": "Session expired: press Reset."})
        except (ValueError, TypeError) as e:
            self.reply(400, {"error": str(e)})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("port", nargs="?", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.store = Store()
    url = f"http://{args.host}:{args.port}"
    print(f"Lab Incident simulator on {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

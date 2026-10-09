"""Edit decision list. Every cut is expressed in BEATS (x = beat index, .5 = half-beat)
and converted to time with the fitted beat grid from beats.json, then snapped to the
nearest 30 fps frame. Run directly to write shotlist.md."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BEATS = json.load(open(os.path.join(HERE, "beats.json")))
FPS = 30
DUR = 16.0
NFRAMES = int(round(DUR * FPS))
T0 = BEATS["beats"][0]
P = BEATS["beat_period_s"]

END_HANDLE = "@yourhandle"     # <- end-card name/handle
TITLE = "KARAN AUJLA"


def bt(x):
    """beat index -> seconds on the fitted grid"""
    return T0 + x * P


def bf(x):
    """beat index -> output frame (nearest)"""
    return int(round(bt(x) * FPS))


# speed: float (constant) or list of (u, speed) keyframes over the shot, u in 0..1
# fx_in / fx_out: transition applied on the first / last frames of the shot
# fx: in-shot emphasis effects
S = []


def shot(a, b, src, t_in, speed=1.0, flip=False, drift="in", fx_in=None, fx_out=None, fx=(), note=""):
    S.append(dict(a=a, b=b, src=src, t_in=t_in, speed=speed, flip=flip, drift=drift,
                  fx_in=fx_in, fx_out=fx_out, fx=list(fx), note=note))


START = None  # shot 1 starts at frame 0, not on a beat
#     beats       source      in      speed / options
shot(START, 4,  "TEA",       1.62,   0.42, drift="hero", fx=["fade_from_black", "text_title"],
     note="slow-mo hero, slow push-in, fade up from black on beat 0, title on downbeat")
shot(4, 6,      "CANDLES",   6.95,   0.85, drift="out", fx_in="light_leak")
shot(6, 8,      "MANOR",     2.62,   0.92, drift="in", fx_in="whip_in")
shot(8, 10,     "SUNSET",    4.42,   0.58, drift="in", fx_out="zoom_through_out", note="golden-hour slow-mo")
shot(10, 12,    "HORSES",    5.10,   1.0, drift="out", fx_in="zoom_through_in")
shot(12, 14,    "KTABLE",    10.75,  1.0, drift="in")
shot(14, 15,    "HORSE_MED", 3.53,   0.95, drift="none", fx=["zoom_punch"])
shot(15, 16,    "LOWCU",     14.05,  1.0, drift="in", fx_in="glitch")
shot(16, 17,    "MANOR2",    4.08,   0.48, drift="out")
shot(17, 18,    "TABLE",     8.55,   1.0, drift="none", fx=["zoom_punch"])
shot(18, 19,    "TEA",       0.30,   1.0, flip=True, drift="in", fx_in="rgb_split")
shot(19, 20,    "CANDLES",   7.75,   1.0, drift="out")
shot(20, 21,    "SUNSET",    4.40,   1.0, flip=True, drift="in", fx=["zoom_punch"])
shot(21, 22,    "HORSES",    6.10,   1.0, flip=True, drift="out")
shot(22, 23,    "KTABLE",    11.70,  1.0, drift="in", fx_in="whip_in_l")
shot(23, 24,    "TABLE",     9.60,   1.0, flip=True, drift="out")
shot(24, 24.5,  "LOWCU",     15.00,  1.0, drift="in", fx=["zoom_punch"])
shot(24.5, 25,  "TEA",       1.00,   1.0, drift="out")
shot(25, 25.5,  "MANOR",     3.00,   1.0, flip=True, drift="in", fx_in="rgb_split")
shot(25.5, 26,  "HORSE_MED", 3.70,   1.0, flip=True, drift="out")
shot(26, 29,    "KTABLE",    12.30,  [(0, 1.0), (1, 0.3)], drift="breakdown",
     fx=["breakdown_mood"], note="breakdown: speed ramp 100%->30%, slow push, desaturate")
shot(29, 29.5,  "LOWCU",     13.85,  1.0, drift="in", fx=["zoom_punch"])
shot(29.5, 30,  "SUNSET",    4.70,   1.0, drift="accel", fx_out="zoom_through_out", note="accelerating push into the drop")
# ---- DROP ---------------------------------------------------------------------------
shot(30, 30.5,  "LOWCU",     14.55,  1.0, drift="none", fx=["flash3", "shake", "drop", "text_drop"])
shot(30.5, 31,  "HORSES",    5.60,   1.0, drift="none", fx=["zoom_punch", "drop"])
shot(31, 32,    "SUNSET",    4.40,   [(0, 1.0), (0.3, 0.3), (0.62, 0.3), (1.0, 2.0)],
     drift="none", fx=["flash2", "shake", "drop"], note="speed ramp 100% -> 30% -> 200%")
shot(32, 32.5,  "MANOR",     2.75,   1.0, drift="none", fx_in="whip_in", fx=["drop"])
shot(32.5, 33,  "KTABLE",    12.75,  1.0, drift="none", fx=["zoom_punch", "drop"])
shot(33, 33.5,  "LOWCU",     15.70,  1.0, drift="none", fx=["flash2", "shake", "drop"])
shot(33.5, 34,  "TEA",       2.05,   1.0, flip=True, drift="none", fx_in="glitch", fx=["drop"])
# ---- END CARD (hard cut on the last downbeat) ---------------------------------------
shot(34, None,  "ENDCARD",   0.0,    1.0, drift="none", fx=["end_card"])

# frames ----------------------------------------------------------------------------------
for s in S:
    s["f0"] = 0 if s["a"] is None else bf(s["a"])
    s["f1"] = NFRAMES if s["b"] is None else bf(s["b"])
    s["t0"] = s["f0"] / FPS
    s["t1"] = s["f1"] / FPS
for p, n in zip(S, S[1:]):
    assert p["f1"] == n["f0"], (p, n)

CUTS = [s["f0"] for s in S[1:]]

# emphasis hits on the drop that are NOT cuts (bass hits inside shots) -> none needed;
# each drop bass hit (13.608, 13.833, 14.058, 14.957) is a cut with flash/shake/punch.


def speed_at(s, u):
    sp = s["speed"]
    if isinstance(sp, (int, float)):
        return float(sp)
    for (u0, v0), (u1, v1) in zip(sp, sp[1:]):
        if u <= u1:
            k = (u - u0) / max(u1 - u0, 1e-9)
            k = k * k * (3 - 2 * k)          # smoothstep between keyframes
            return v0 + (v1 - v0) * k
    return sp[-1][1]


def src_time(s, t):
    """source time for output time t inside shot s (numerical integral of speed)"""
    dur = s["t1"] - s["t0"]
    if isinstance(s["speed"], (int, float)):
        return s["t_in"] + s["speed"] * (t - s["t0"])
    n = 200
    acc, prev = 0.0, s["t0"]
    for i in range(1, n + 1):
        tt = s["t0"] + (t - s["t0"]) * i / n
        u = (tt - s["t0"]) / dur
        acc += speed_at(s, u) * (tt - prev)
        prev = tt
    return s["t_in"] + acc


def write_shotlist(path):
    from sources import SHOTS
    L = []
    L.append("# Shot list — Karan Aujla beat-synced edit (16.000 s)\n")
    L.append(f"- Audio: song 2:50–3:06 (recording audio, 0.000–16.000 s), BPM **{BEATS['bpm']:.2f}** "
             f"(beat = {P*1000:.1f} ms), grid origin {T0:.4f} s")
    L.append(f"- Downbeats (bar lines): {', '.join(f'{t:.3f}' for t in BEATS['downbeats'])}")
    L.append("- Sections: " + "; ".join(f"{x['name']} {x['start']:.2f}–{x['end']:.2f}s" for x in BEATS["sections"]))
    L.append(f"- Output 1080x1920 @ {FPS} fps = {NFRAMES} frames. Cut frame = round(beat_time × 30); "
             "max quantisation error ±16.7 ms (< ½ frame).")
    L.append(f"- {len(S)} shots / **{len(CUTS)} cuts**, every cut on a beat (B) or half-beat (½).\n")
    L.append("| # | start–end (s) | frames | beat | source (framing) | in-point → out (src s) | speed | effects |")
    L.append("|---|---|---|---|---|---|---|---|")
    for i, s in enumerate(S, 1):
        a = "0 (start)" if s["a"] is None else (f"B{s['a']:g}" if float(s["a"]).is_integer() else f"½ B{s['a']:g}")
        if s["src"] == "ENDCARD":
            src, io, spd = "end card", "—", "—"
        else:
            src = f"{s['src']} ({SHOTS[s['src']][2]})" + (" ⇋flip" if s["flip"] else "")
            io = f"{s['t_in']:.2f} → {src_time(s, s['t1']):.2f}"
            spd = (f"{int(s['speed']*100)}%" if isinstance(s["speed"], (int, float))
                   else "→".join(f"{int(v*100)}%" for _, v in s["speed"]))
        fx = [x for x in [s["fx_in"] and f"in:{s['fx_in']}", s["fx_out"] and f"out:{s['fx_out']}"] if x]
        fx += s["fx"] + ([f"drift:{s['drift']}"] if s["drift"] not in ("none",) else [])
        L.append(f"| {i} | {s['t0']:.3f}–{s['t1']:.3f} | {s['f0']}–{s['f1']-1} | {a} | {src} | {io} | {spd} | "
                 f"{', '.join(fx)}{(' — ' + s['note']) if s['note'] else ''} |")
    L.append("\n## Text hits (max 3)\n")
    L.append(f"1. `{TITLE}` — tracked serif, B2 downbeat (1.019 s) → B4, letter-spacing breathes open.")
    L.append("2. `AUJLA` — heavy sans slam on the drop (B30, 13.608 s, vocal punch at 13.616 s) for 2 beats, RGB split.")
    L.append(f"3. End card `{END_HANDLE}` — glow-and-fade, pulse on the final bass hit (B35, 15.856 s).")
    L.append("\n## Section → effect map\n")
    L.append("- **Intro (0–1.92 s)**: 42% optical-flow slow-mo, slow push-in 100→110%, fade from black, title.")
    L.append("- **Build-up (1.92–11.81 s)**: cuts every 2 beats → every beat (5.97 s) → every half-beat (10.91 s); "
             "transitions light leak, whip pan, zoom-through, glitch, RGB split; zoom punches on bass hits.")
    L.append("- **Breakdown (11.81–13.61 s)**: one held shot, speed ramp 100→30%, slow push, desaturated; "
             "half-beat stutter + accelerating push into the drop.")
    L.append("- **Drop (13.61–15.41 s)**: half-beat rapid fire, white flash frames on the hardest bass hits, "
             "camera shake + chromatic aberration (drop only), zoom punches, speed ramp 100→30→200%, text slam.")
    L.append("- **Outro (15.41–16.00 s)**: hard cut on the downbeat to a glow-and-fade end card.")
    L.append("\n## Look\n")
    L.append("Warm-gold grade (suits the golden-hour + candle-lit footage): per-shot exposure / white-balance "
             "match, lifted blacks, warm highlights with slightly cool shadows, highlight bloom, soft vignette, "
             "fine film grain. Same pipeline on every frame for a consistent look.")
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    write_shotlist(os.path.join(HERE, "shotlist.md"))
    print(open(os.path.join(HERE, "shotlist.md")).read())

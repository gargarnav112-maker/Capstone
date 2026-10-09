"""CRAZY cut: same audio, beat grid and footage as edl.py, but cut on beats, half-beats and
quarter-beat bursts (54 cuts), with velocity ramps, bounce/shake on every half-beat,
mirror/kaleido shots, stutter holds, invert hits, spins and a word slam on every drop beat.
Flashes + inverts are kept to <= 3 per second (photosensitivity guideline).
Run directly to write shotlist_crazy.md."""
import os

import edl as base
from edl import BEATS, FPS, NFRAMES, P, T0, bf, bt, src_time  # noqa: F401  (re-exported for render.py)

TITLE = base.TITLE
END_HANDLE = base.END_HANDLE
TITLE_BEAT = 1
TEXT_HITS = [(30, 31, "KARAN"), (31, 32, "AUJLA"), (32, 33, "KARAN"), (33, 34, "AUJLA")]

VEL = [(0, 2.0), (0.4, 0.3), (0.75, 0.3), (1.0, 2.0)]        # velocity edit: fast-slow-fast
S = []


def shot(a, b, src, t_in, speed=1.0, flip=False, drift="none", fx_in=None, fx_out=None, fx=(), hold=0, note=""):
    S.append(dict(a=a, b=b, src=src, t_in=t_in, speed=speed, flip=flip, drift=drift, fx_in=fx_in,
                  fx_out=fx_out, fx=list(fx), hold=hold, note=note))


# ---- INTRO: calm before the storm ---------------------------------------------------
shot(None, 2,   "TEA",       1.75, 0.45, drift="hero", fx=["fade_from_black", "text_title"])
# ---- BUILD 1: every beat, velocity ramps + bounce --------------------------------------
shot(2, 3,      "LOWCU",     14.00, VEL, fx_in="spin_in", fx=["flash2", "bounce"])
shot(3, 4,      "MANOR",     2.62, VEL, fx_in="whip_in", fx=["bounce"])
shot(4, 5,      "SUNSET",    4.42, VEL, flip=True, fx=["mirror", "bounce"])
shot(5, 6,      "HORSES",    5.15, VEL, fx_in="rgb_split", fx=["bounce"])
shot(6, 7,      "KTABLE",    10.80, VEL, fx=["zoom_punch", "flash2"])
shot(7, 8,      "CANDLES",   6.95, 1.4, fx_in="whip_in_l", fx=["bounce"])
shot(8, 9,      "TEA",       0.20, VEL, fx_in="glitch", fx=["bounce"])
shot(9, 10,     "HORSE_MED", 3.55, 0.85, fx=["zoom_punch"], fx_out="zoom_through_out")
# ---- BUILD 2: half-beats on the bass groove, shakes --------------------------------------
shot(10, 10.5,  "LOWCU",     14.60, fx_in="zoom_through_in", fx=["flash2", "shake", "ca"])
shot(10.5, 11,  "TABLE",     8.60, fx_in="rgb_split")
shot(11, 11.5,  "SUNSET",    4.70, fx=["zoom_punch", "invert1"])
shot(11.5, 12,  "MANOR",     3.00, flip=True, fx_in="whip_in")
shot(12, 12.5,  "KTABLE",    11.50, fx=["shake", "flash2", "ca"])
shot(12.5, 13,  "HORSES",    5.90, flip=True, fx=["zoom_punch"])
shot(13, 13.5,  "TEA",       0.90, flip=True, fx_in="glitch", fx=["bounce"])
shot(13.5, 14,  "CANDLES",   7.50, flip=True, fx_in="spin_in")
shot(14, 15,    "LOWCU",     15.00, hold=3, fx=["bounce", "shake_beats", "ca"], note="stutter x3")
shot(15, 16,    "HORSES",    6.20, VEL, fx=["rgb_beats", "bounce"])
shot(16, 17,    "KTABLE",    12.00, hold=2, fx=["bounce", "flash2"], note="stutter x2")
shot(17, 18,    "SUNSET",    4.42, VEL, flip=True, fx=["mirror", "shake_beats"])
# ---- BUILD 3: chaos — half-beats + quarter-beat bursts -----------------------------------
shot(18, 18.5,  "TABLE",     9.40, fx=["flash2", "shake", "ca"])
shot(18.5, 19,  "TEA",       1.30, flip=True, fx_in="rgb_split")
shot(19, 19.5,  "LOWCU",     15.50, fx=["zoom_punch"])
shot(19.5, 20,  "MANOR",     2.70, fx_in="whip_in_l")
shot(20, 20.25, "KTABLE",    12.50, fx=["invert1", "ca"], note="quarter-beat burst")
shot(20.25, 20.5, "HORSES",  5.30, fx=["ca"])
shot(20.5, 20.75, "SUNSET",  4.60, fx=["ca"])
shot(20.75, 21, "CANDLES",   7.00, flip=True, fx=["ca"])
shot(21, 22,    "TEA",       0.40, VEL, fx=["rgb_beats", "bounce"])
shot(22, 22.5,  "LOWCU",     14.30, flip=True, fx=["flash2", "shake", "ca"])
shot(22.5, 23,  "HORSE_MED", 3.60, flip=True, fx_in="glitch")
shot(23, 23.5,  "TABLE",     10.00, fx=["zoom_punch"])
shot(23.5, 24,  "KTABLE",    11.20, flip=True, fx_in="rgb_split")
shot(24, 24.25, "SUNSET",    4.45, fx=["invert1", "ca"], note="quarter-beat burst")
shot(24.25, 24.5, "MANOR",   3.20, fx=["ca"])
shot(24.5, 24.75, "LOWCU",   15.20, flip=True, fx=["ca"])
shot(24.75, 25, "TEA",       2.10, fx=["ca"])
shot(25, 25.5,  "HORSES",    6.40, fx=["shake", "flash2", "ca"])
shot(25.5, 26,  "CANDLES",   7.90, fx_out="zoom_through_out")
# ---- BREAKDOWN / RISER ---------------------------------------------------------------------
shot(26, 28,    "KTABLE",    12.25, [(0, 1.0), (1, 0.25)], drift="breakdown", fx=["breakdown_mood"],
     note="ramp 100->25%")
shot(28, 29,    "SUNSET",    4.42, 0.5, flip=True, hold=2, drift="accel", fx=["mirror"], note="mirror + stutter riser")
shot(29, 29.5,  "LOWCU",     13.90, drift="accel", fx=["shake_beats"])
shot(29.5, 29.75, "TEA",     1.60, drift="accel", fx=["shake_beats", "ca"])
shot(29.75, 30, "HORSE_MED", 3.60, drift="accel", fx=["shake_beats", "ca"], fx_out="zoom_through_out")
# ---- DROP ----------------------------------------------------------------------------------
shot(30, 30.5,  "LOWCU",     14.60, fx=["flash3", "shake", "drop"])
shot(30.5, 31,  "HORSES",    5.50, fx=["zoom_punch", "shake_beats", "drop"])
shot(31, 32,    "SUNSET",    4.40, [(0, 1.0), (0.3, 0.3), (0.62, 0.3), (1.0, 2.0)],
     flip=True, fx=["flash2", "shake", "drop", "mirror"], note="ramp 100->30->200%")
shot(32, 32.25, "MANOR",     2.80, fx=["drop", "bounce"])
shot(32.25, 32.5, "KTABLE",  12.80, fx=["invert1", "drop"])
shot(32.5, 33,  "TEA",       1.90, flip=True, fx=["zoom_punch", "shake", "drop"])
shot(33, 33.5,  "LOWCU",     15.60, fx=["flash2", "shake", "drop"])
shot(33.5, 33.75, "TABLE",   9.00, fx_in="glitch", fx=["drop"])
shot(33.75, 34, "HORSE_MED", 3.70, flip=True, fx_in="spin_in", fx=["drop"])
# ---- END CARD ------------------------------------------------------------------------------
shot(34, None,  "ENDCARD",   0.0, fx=["end_card", "glitch_in"])

for s in S:
    s["f0"] = 0 if s["a"] is None else bf(s["a"])
    s["f1"] = NFRAMES if s["b"] is None else bf(s["b"])
    s["t0"], s["t1"] = s["f0"] / FPS, s["f1"] / FPS
    assert s["f1"] > s["f0"], s
for p_, n_ in zip(S, S[1:]):
    assert p_["f1"] == n_["f0"]
CUTS = [s["f0"] for s in S[1:]]

# photosensitivity guard: flash/invert frames <= 3 in any 30-frame window
_fl = sorted(s["f0"] for s in S if any(f in s["fx"] for f in ("flash2", "flash3", "invert1")))
assert all(sum(1 for x in _fl if a <= x < a + FPS) <= 3 for a in range(NFRAMES)), _fl


def write_shotlist(path):
    from sources import SHOTS
    L = ["# Shot list — CRAZY cut (16.000 s)\n",
         f"Same audio / beat grid as the cinematic cut: {BEATS['bpm']:.2f} BPM, beat {P*1000:.1f} ms. "
         f"**{len(S)} shots / {len(CUTS)} cuts** on beat (B), half-beat (½) and quarter-beat (¼) frames.\n",
         "| # | start–end (s) | frames | beat | source | in → out (src s) | speed | effects |",
         "|---|---|---|---|---|---|---|---|"]
    for i, s in enumerate(S, 1):
        a = "0" if s["a"] is None else f"B{s['a']:g}"
        if s["src"] == "ENDCARD":
            src, io, spd = "end card", "—", "—"
        else:
            src = f"{s['src']} ({SHOTS[s['src']][2]})" + (" ⇋" if s["flip"] else "")
            io = f"{s['t_in']:.2f} → {src_time(s, s['t1']):.2f}"
            spd = (f"{int(s['speed']*100)}%" if isinstance(s["speed"], (int, float))
                   else "velocity " + "→".join(f"{int(v*100)}" for _, v in s["speed"]))
            if s["hold"]:
                spd += f", stutter×{s['hold']}"
        fx = [x for x in [s["fx_in"] and f"in:{s['fx_in']}", s["fx_out"] and f"out:{s['fx_out']}"] if x] + s["fx"]
        L.append(f"| {i} | {s['t0']:.3f}–{s['t1']:.3f} | {s['f0']}–{s['f1']-1} | {a} | {src} | {io} | {spd} | "
                 f"{', '.join(fx)}{(' — ' + s['note']) if s['note'] else ''} |")
    L.append("\nText: title `KARAN AUJLA` B1→B2; drop words `KARAN` / `AUJLA` / `KARAN` / `AUJLA` one per beat "
             f"(B30–B34); end card `{END_HANDLE}`.")
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    write_shotlist(os.path.join(os.path.dirname(os.path.abspath(__file__)), "shotlist_crazy.md"))
    print(len(S), "shots", len(CUTS), "cuts")

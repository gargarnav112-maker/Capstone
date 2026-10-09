"""Beat-locked cut grid -> work/shotlist.md (source columns filled once footage exists)."""
import json
d = json.load(open("work/beats.json"))
B = d["beats"]; P = d["beat_period"]
h = lambda i: B[i] + P / 2          # half-beat after beat i
FPS = 60
# (start_time, section, timing basis, effect, framing)
plan = [
    (0.0,   "intro",     "fade-in from black lands on b0 (0.208)", "slow-mo 40% + slow push-in 100->108%, fade up from black", "wide hero"),
    (B[3],  "build",     "b3 downbeat - 808 entry",               "hard cut",                                  "medium"),
    (B[5],  "build",     "b5 (2-beat)",                           "whip pan L->R into shot",                   "close"),
    (B[7],  "build",     "b7 downbeat (2-beat)",                  "hard cut",                                  "wide"),
    (B[9],  "build",     "b9 (2-beat)",                           "light leak sweep",                          "medium"),
    (B[11], "build",     "b11 downbeat (1-beat)",                 "hard cut",                                  "close"),
    (B[12], "build",     "b12 (1-beat)",                          "zoom punch 100->115% / 4f",                 "wide"),
    (B[13], "build",     "b13 (1-beat)",                          "hard cut",                                  "medium"),
    (B[14], "build",     "b14 (half-beat run)",                   "glitch 3f",                                 "close"),
    (h(14), "build",     "b14.5 half-beat",                       "hard cut",                                  "medium"),
    (B[15], "drop",      "b15 downbeat - DROP",                   "white flash 3f + zoom punch + shake",       "wide"),
    (B[16], "drop",      "b16",                                   "hard cut",                                  "close"),
    (h(16), "drop",      "b16.5 bass hit",                        "RGB split 4f",                              "medium"),
    (B[17], "drop",      "b17",                                   "hard cut",                                  "wide"),
    (B[18], "drop",      "b18 - loudest vocal",                   "white flash 2f + speed ramp 100>30>200% (2 beats)", "close"),
    (B[20], "drop",      "b20",                                   "whip pan R->L",                             "medium"),
    (B[21], "drop",      "b21",                                   "hard cut + shake",                          "close"),
    (B[22], "drop",      "b22 bass hit",                          "zoom punch",                                "wide"),
    (h(22), "drop",      "b22.5 bass hit",                        "hard cut",                                  "medium"),
    (B[23], "peak",      "b23 downbeat",                          "white flash 3f + shake",                    "close"),
    (B[24], "peak",      "b24 vocal punch",                       "hard cut",                                  "wide"),
    (B[25], "peak",      "b25",                                   "zoom-through transition",                   "medium"),
    (B[26], "peak",      "b26",                                   "hard cut",                                  "close"),
    (B[27], "peak",      "b27 downbeat",                          "RGB split + shake",                         "wide"),
    (B[28], "peak",      "b28",                                   "hard cut",                                  "medium"),
    (B[29], "peak",      "b29",                                   "glitch 3f",                                 "close"),
    (B[30], "peak",      "b30 vocal punch",                       "zoom punch",                                "medium"),
    (B[31], "breakdown", "b31 downbeat - bass drop-out",          "hard cut; slow-mo 50%, bloom up",           "wide"),
    (B[33], "end card",  "b33 downbeat - HARD CUT on last full-bar beat", "end card: glow-in handle, glow burst on final hit b35 (15.881), fade to black by 16.000", "card"),
]
rows = []
for k, (t, sec, basis, fx, frame) in enumerate(plan):
    t1 = plan[k + 1][0] if k + 1 < len(plan) else 16.0
    f0, f1 = round(t * FPS), round(t1 * FPS)
    rows.append(f"| {k+1:>2} | {t:6.3f} | {t1:6.3f} | {f0:>3}-{f1:>3} | {(t1-t):.3f} | {sec} | {basis} | TBD | TBD | {frame} | {fx} |")
hdr = f"""# Shot list - Karan Aujla, "Ashke" (2:50-3:06)

BPM {d['bpm']} (half-time feel {d['bpm']/2:.1f}), beat period {P*1000:.1f} ms. Frames at {FPS} fps; worst-case cut
quantisation = 1/{2*FPS} s = {1000/(2*FPS):.1f} ms (< 1 frame). {len(plan)} shots / {len(plan)-1} cuts.

Sections: """ + "; ".join(f"{s['name']} {s['start']:.3f}-{s['end']:.3f}" for s in d["sections"]) + """

| # | start s | end s | frames@60 | dur | section | sync basis | source clip | in-point | framing | effect |
|---|---|---|---|---|---|---|---|---|---|---|
"""
open("work/shotlist.md", "w").write(hdr + "\n".join(rows) + "\n")
print(hdr + "\n".join(rows))

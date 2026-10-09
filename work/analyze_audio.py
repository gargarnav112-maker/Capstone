"""Beat / onset / energy analysis of the 16 s segment (2:50-3:06).

Analyses an 18 s window (1 s of context each side) so the tracker is stable
at the edges, then shifts all times back into segment time (0..16 s).
"""
import json
import numpy as np
import librosa

SR = 44100
HOP = 256
CTX = 1.0           # seconds of context before segment start
SEG = 16.0
FPS = 30

y, sr = librosa.load("work/analysis_ctx.wav", sr=SR, mono=True)

# --- tempo + beats (percussive component tracks drums more cleanly) ---
y_h, y_p = librosa.effects.hpss(y)
onset_env = librosa.onset.onset_strength(y=y_p, sr=sr, hop_length=HOP, aggregate=np.median)
tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=HOP, tightness=120)
tempo = float(np.atleast_1d(tempo)[0])
beat_t = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP) - CTX

# Fit a constant-tempo grid: joint BPM/phase search maximising the sum of the
# full-mix onset envelope and a low-band (kick/808) flux envelope at grid points.
# (A plain polyfit on tracker beats is skewed by the tracker's unstable first beats.)
times_env = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=HOP) - CTX
_S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
_f = librosa.fft_frequencies(sr=sr, n_fft=2048)
_low = librosa.amplitude_to_db(_S[(_f >= 35) & (_f <= 120)], ref=np.max)
kick_env = np.concatenate([[0], np.maximum(0, np.diff(_low, axis=1)).mean(0)])[: len(times_env)]
full_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)[: len(times_env)]
fit_env = full_env / full_env.max() + kick_env / kick_env.max()
best = (-1, None, None)
for bpm_c in np.arange(128.0, 140.0, 0.01):
    p = 60.0 / bpm_c
    for ph in np.arange(0.0, p, 0.002):
        g = ph + np.arange(0, SEG / p + 1) * p
        sc = np.interp(g[g < SEG], times_env, fit_env).mean()
        if sc > best[0]:
            best = (sc, bpm_c, ph)
_, grid_bpm, phase = best
period = 60.0 / grid_bpm
grid = phase + np.arange(0, SEG / period + 1) * period
grid = grid[grid < SEG]
idx = np.arange(len(beat_t))

# --- downbeats: choose the bar phase (of 4) with the most low-end energy ---
S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
bass = S[(freqs >= 30) & (freqs <= 150)].sum(axis=0)
bass = bass / (bass.max() + 1e-9)
def at(sig, t):
    return float(np.interp(t, times_env, sig[: len(times_env)]))
bar_scores = [np.mean([at(bass, t) + at(onset_env / onset_env.max(), t) for t in grid[k::4]]) for k in range(4)]
db_phase = int(np.argmax(bar_scores))
# Bass-sum scores across bar phases are near-equal on this 808-heavy section, so anchor
# the bar to the structural changes instead: the 808 entry, the bass drop-out and the
# final hit all land on the same beat phase.
def _bass_jump(t):
    return at(bass, t + 0.1) - at(bass, t - 0.25)
jumps = np.array([abs(_bass_jump(t)) for t in grid])
db_phase_struct = int(np.argmax([jumps[k::4].max() for k in range(4)]))
db_phase = db_phase_struct
downbeats = grid[db_phase::4]

# --- onsets (full mix) ---
onset_full = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
on_frames = librosa.onset.onset_detect(onset_envelope=onset_full, sr=sr, hop_length=HOP, backtrack=False, delta=0.25)
on_t = librosa.frames_to_time(on_frames, sr=sr, hop_length=HOP) - CTX
on_str = onset_full[on_frames] / onset_full.max()
keep = (on_t >= 0) & (on_t < SEG)
onsets = [{"t": round(float(t), 4), "strength": round(float(s), 3)} for t, s in zip(on_t[keep], on_str[keep])]

# --- energy curves ---
rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
rms_db = librosa.amplitude_to_db(rms, ref=np.max)
voc_band = S[(freqs >= 300) & (freqs <= 3500)].sum(axis=0)
y_h_S = np.abs(librosa.stft(y_h, n_fft=2048, hop_length=HOP))
voc = y_h_S[(freqs >= 300) & (freqs <= 3500)].sum(axis=0)
voc = voc / (voc.max() + 1e-9)
voc_flux = np.maximum(0, np.diff(voc, prepend=voc[0]))
bass_flux = np.maximum(0, np.diff(bass, prepend=bass[0]))

# High-energy moments: bass hits and vocal punches = peaks of flux, snapped to nearest beat/half-beat.
half_grid = np.sort(np.concatenate([grid, grid + period / 2]))
half_grid = half_grid[half_grid < SEG]
def peaks(sig, thr_pct, min_gap):
    fr = librosa.util.peak_pick(sig, pre_max=8, post_max=8, pre_avg=20, post_avg=20, delta=0.0, wait=int(min_gap * sr / HOP))
    t = librosa.frames_to_time(fr, sr=sr, hop_length=HOP) - CTX
    v = sig[fr]
    thr = np.percentile(v, thr_pct) if len(v) else 0
    out = [(float(a), float(b)) for a, b in zip(t, v) if b >= thr and 0 <= a < SEG]
    return out
def snap(t):
    return float(half_grid[np.argmin(np.abs(half_grid - t))])
bass_hits = [{"t": round(snap(t), 4), "raw": round(t, 4), "strength": round(v / bass_flux.max(), 3)} for t, v in peaks(bass_flux, 60, 0.2)]
vocal_punches = [{"t": round(snap(t), 4), "raw": round(t, 4), "strength": round(v / voc_flux.max(), 3)} for t, v in peaks(voc_flux, 70, 0.25)]

# Per-beat energy table
rows = []
for i, t in enumerate(grid):
    t2 = t + period
    m = (times_env >= t) & (times_env < t2)
    rows.append({
        "beat": i,
        "t": round(float(t), 4),
        "frame30": int(round(t * FPS)),
        "downbeat": bool(np.any(np.isclose(downbeats, t))),
        "rms_db": round(float(rms_db[: len(times_env)][m].mean()), 2),
        "bass": round(float(bass[: len(times_env)][m].mean()), 3),
        "onset": round(float(onset_env[m].max() / onset_env.max()), 3),
        "vocal": round(float(voc[: len(times_env)][m].mean()), 3),
    })

# Smoothed energy per second for section detection
sec = []
for s in range(int(SEG)):
    m = (times_env >= s) & (times_env < s + 1)
    sec.append({"sec": s,
                "rms_db": round(float(rms_db[: len(times_env)][m].mean()), 2),
                "bass": round(float(bass[: len(times_env)][m].mean()), 3),
                "onset_density": round(float(onset_env[m].mean() / onset_env.max()), 3),
                "vocal": round(float(voc[: len(times_env)][m].mean()), 3)})

# Section boundaries, all snapped to downbeats:
#  entry     = first beat whose RMS stays above -9 dB (808 comes in)
#  drop      = downbeat at/just before the first cluster of >=5 bass hits within 1 s
#  peak      = next downbeat pair after the drop (vocal-dense half)
#  drop_out  = first beat after 60% where RMS falls below -10.5 dB for 2 beats
#  final_hit = first beat after drop_out back above -9 dB
rd = np.array([r["rms_db"] for r in rows])
b = lambda i: round(float(grid[i]), 4)
entry = next(i for i in range(len(rd) - 1) if rd[i] > -9 and rd[i + 1] > -9)
drop_out = next(i for i in range(len(rd) - 1) if grid[i] > SEG * 0.6 and rd[i] < -10.5 and rd[i + 1] < -10.5)
final_hit = next(i for i in range(drop_out, len(rd)) if rd[i] > -9)
bh = np.array([h["t"] for h in bass_hits])
cluster = next(t for t in bh if ((bh >= t) & (bh < t + 1.0)).sum() >= 5 and t > grid[entry])
db_idx = [i for i in range(db_phase, len(grid), 4)]
drop = max(i for i in db_idx if grid[i] <= cluster + 1e-6)
peak = drop + 8
sections = [
    {"name": "intro / hero", "start": 0.0, "end": b(entry), "note": "filtered, vocal-led, almost no sub-bass"},
    {"name": "build-up", "start": b(entry), "end": b(drop), "note": "808 enters on downbeat; steady groove"},
    {"name": "drop", "start": b(drop), "end": b(peak), "note": "densest run of bass hits; loudest vocal at 8.27 s"},
    {"name": "peak", "start": b(peak), "end": b(drop_out), "note": "vocal-dense, rapid-fire phrase"},
    {"name": "breakdown / outro", "start": b(drop_out), "end": b(final_hit), "note": "bass drops out (-14 dB)"},
    {"name": "final hit / end card", "start": b(final_hit), "end": SEG, "note": "strongest bass hit in the segment"},
]

out = {
    "segment": {"source_start": 170.0, "source_end": 186.0, "duration": SEG},
    "bpm_tracker": round(tempo, 2),
    "bpm": round(grid_bpm, 2),
    "beat_period": round(float(period), 5),
    "beats": [round(float(t), 4) for t in grid],
    "half_beats": [round(float(t), 4) for t in half_grid],
    "downbeats": [round(float(t), 4) for t in downbeats],
    "raw_tracker_beats": [round(float(t), 4) for t in beat_t if 0 <= t < SEG],
    "onsets": onsets,
    "bass_hits": bass_hits,
    "vocal_punches": vocal_punches,
    "per_beat": rows,
    "per_second": sec,
    "sections": sections,
}
json.dump(out, open("work/beats.json", "w"), indent=1, default=float)

print(f"BPM (tracker) {tempo:.2f} | BPM (grid fit) {grid_bpm:.2f} | period {period*1000:.1f} ms | beats {len(grid)} | downbeat phase {db_phase}")
_bt = beat_t[(beat_t >= 0) & (beat_t < SEG)]
_dev = np.array([t - grid[np.argmin(np.abs(grid - t))] for t in _bt]) * 1000
print(f"tracker->grid deviation: median {np.median(np.abs(_dev)):.1f} ms, max {np.max(np.abs(_dev)):.1f} ms (beats with |dev|>34ms: {[round(float(t),3) for t,d in zip(_bt,_dev) if abs(d)>34]})")
print("\n beat   time   f@30  DB  rms_dB  bass   onset  vocal  energy")
for r in rows:
    bar = "#" * int(max(0, (r["rms_db"] + 30)) * 1.2)
    print(f" {r['beat']:>3}  {r['t']:6.3f}  {r['frame30']:>4}  {'*' if r['downbeat'] else ' '}  {r['rms_db']:6.2f}  {r['bass']:.3f}  {r['onset']:.3f}  {r['vocal']:.3f}  {bar}")
print("\nper second:")
for s in sec:
    print(f" {s['sec']:>2}s  rms {s['rms_db']:6.2f}  bass {s['bass']:.3f}  onset {s['onset_density']:.3f}  vocal {s['vocal']:.3f}  " + "#" * int((s['rms_db'] + 30) * 1.5))
print("\nsections:")
for x in sections:
    print(f"  {x['start']:6.3f} - {x['end']:6.3f}  {x['name']:<22} {x['note']}")
print("\nbass hits:", [(h['t'], h['strength']) for h in bass_hits])
print("vocal punches:", [(h['t'], h['strength']) for h in vocal_punches])
print("onsets:", len(onsets), "strong(>0.4):", [o['t'] for o in onsets if o['strength'] > 0.4])

"""Step 1: trim the song segment and analyse beats / energy.

Input : work/full_audio.wav  (audio track of the screen recording = song 2:50-3:06)
Output: work/segment.wav, work/beats.json, printed timeline table
"""
import json
import subprocess
import sys

import librosa
import numpy as np

SR = 48000
HOP = 256
SEG_START = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0  # offset inside the recording
SEG_LEN = 16.0

# --- trim with fades (sample-accurate, done by ffmpeg) -----------------------
subprocess.run([
    "ffmpeg", "-v", "error", "-y", "-i", "full_audio.wav",
    "-af", f"atrim=start={SEG_START}:duration={SEG_LEN},asetpts=N/SR/TB,"
           f"afade=t=in:st=0:d=0.15:curve=qsin,afade=t=out:st={SEG_LEN-0.4}:d=0.4:curve=qsin",
    "-ar", str(SR), "-c:a", "pcm_s24le", "segment.wav"], check=True)

# analyse the un-faded audio so the first/last beats are not masked by the fades
y, _ = librosa.load("full_audio.wav", sr=SR, mono=True, offset=SEG_START, duration=SEG_LEN)
dur = len(y) / SR

# --- onset envelopes -----------------------------------------------------------
oenv = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
bass = S[freqs < 150].sum(0)
bass_flux = np.maximum(0, np.diff(bass, prepend=bass[0]))
y_h, y_p = librosa.effects.hpss(y)
voc_env = librosa.onset.onset_strength(y=y_h, sr=SR, hop_length=HOP, fmin=300, fmax=4000)
times = librosa.times_like(oenv, sr=SR, hop_length=HOP)

# --- beats: librosa tracker, then fit a perfectly regular grid (music is quantised)
tempo, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=SR, hop_length=HOP, units="time")
tempo = float(np.atleast_1d(tempo)[0])
period = 60.0 / tempo
idx = np.round((beats - beats[0]) / period)
P, t0 = np.polyfit(idx, beats, 1)
# refine: snap each grid beat to the strongest high-resolution onset peak within +-70 ms,
# then robust (outlier-rejecting) linear fit -> tempo + phase from the real transients
hr_hop = 48
oenv_hr = librosa.onset.onset_strength(y=y, sr=SR, hop_length=hr_hop, n_fft=1024)
t_hr = librosa.times_like(oenv_hr, sr=SR, hop_length=hr_hop, n_fft=1024)
for _ in range(2):
    g = t0 + P * np.arange(int(np.ceil(-t0 / P)), 200)
    g = g[(g >= 0) & (g < dur)]
    ii, pk = [], []
    for b in g:
        m = (t_hr > b - 0.07) & (t_hr < b + 0.07)
        if m.any():
            pk.append(t_hr[m][np.argmax(oenv_hr[m])])
            ii.append(round((b - t0) / P))
    ii, pk = np.array(ii), np.array(pk)
    keep = np.ones(len(ii), bool)
    for _ in range(3):
        P, t0 = np.polyfit(ii[keep], pk[keep], 1)
        res = pk - (t0 + P * ii)
        keep = np.abs(res) < max(0.02, 2.5 * np.median(np.abs(res)))
grid_residual_ms = np.round(res * 1000).astype(int).tolist()
n0 = int(np.ceil(-t0 / P))
grid = t0 + P * np.arange(n0, 200)
grid = grid[(grid >= 0) & (grid < dur)]
bpm = 60.0 / P
max_dev = float(np.max(np.abs(beats[:, None] - grid[None, :]).min(1)))

# --- downbeats: phase (0..3) with the strongest summed bass flux ----------------
bf = np.interp(grid, times, bass_flux)
phase = int(np.argmax([bf[p::4].sum() for p in range(4)]))
downbeats = grid[phase::4]

# --- onset peaks & high-energy hits ---------------------------------------------
onsets = librosa.onset.onset_detect(onset_envelope=oenv, sr=SR, hop_length=HOP, units="time")
o_str = np.interp(onsets, times, oenv)
strong_onsets = onsets[o_str > np.percentile(o_str, 60)]
halfbeats = np.sort(np.concatenate([grid, grid + P / 2]))
halfbeats = halfbeats[halfbeats < dur]

def peaks(env, pct):
    pk = librosa.util.peak_pick(env, pre_max=8, post_max=8, pre_avg=20, post_avg=20, delta=0, wait=20)
    v = env[pk]
    return times[pk[v > np.percentile(v, pct)]], v

bass_hits, _ = peaks(bass_flux / bass_flux.max(), 70)
vocal_punch, _ = peaks(voc_env / voc_env.max(), 75)

def snap(ts, to, tol=0.06):
    out = []
    for t in ts:
        j = np.argmin(np.abs(to - t))
        if abs(to[j] - t) <= tol:
            out.append(round(float(to[j]), 4))
    return sorted(set(out))

bass_hits_on_grid = snap(bass_hits, halfbeats)

# --- energy per beat ---------------------------------------------------------------
rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
low = librosa.feature.rms(S=S[freqs < 200], frame_length=2048)[0] if False else None
edges = np.concatenate([[0], grid, [dur]])
rows = []
for a, b in zip(edges[:-1], edges[1:]):
    m = (times >= a) & (times < b)
    rows.append((a, b, float(rms[m].mean()) if m.any() else 0.0,
                 float(np.log1p(bass[m]).mean()) if m.any() else 0.0,
                 float(oenv[m].max()) if m.any() else 0.0))
e = np.array([r[2] for r in rows]); e_n = (e - e.min()) / (np.ptp(e) + 1e-9)
bs = np.array([r[3] for r in rows]); b_n = (bs - bs.min()) / (np.ptp(bs) + 1e-9)
energy = 0.6 * e_n + 0.4 * b_n
# smooth over a 4-beat window for the section curve
k = np.ones(4) / 4
smooth = np.convolve(np.pad(energy, 2, mode="edge"), k, mode="same")[2:-2]

# --- sections from the smoothed energy curve ------------------------------------
# beat-level energy (rows[0] is the pickup before beat 0)
eb = energy[1:]
sm = np.convolve(np.pad(eb, 1, mode="edge"), np.ones(3) / 3, mode="valid")
low = sm < 0.35                                    # breakdown = sustained low energy
bd = [i for i in range(len(sm)) if low[i]]
bd_start = bd[0] if bd else None
bd_end = bd[-1] + 1 if bd else None                # first beat back at full energy = the drop
# structural downbeats: section changes land on bar lines, so the bar phase is the
# one that puts the breakdown start / drop on a downbeat (bass-flux phase as fallback)
phase_struct = (bd_end % 4) if bd_end is not None else phase
downbeats_struct = grid[phase_struct::4]
sections = [
    {"name": "intro / hero", "start": 0.0, "end": float(grid[4])},
    {"name": "build-up", "start": float(grid[4]), "end": float(grid[bd_start])},
    {"name": "breakdown (riser)", "start": float(grid[bd_start]), "end": float(grid[bd_end])},
    {"name": "drop / peak", "start": float(grid[bd_end]), "end": float(grid[min(bd_end + 4, len(grid) - 1)])},
    {"name": "outro / end card", "start": float(grid[min(bd_end + 4, len(grid) - 1)]), "end": dur},
]

print(f"segment start in recording: {SEG_START:.3f}s   duration: {dur:.3f}s")
print(f"BPM (librosa) {tempo:.2f}   BPM (grid fit) {bpm:.3f}   period {P*1000:.1f} ms   "
      f"max tracker deviation from grid {max_dev*1000:.0f} ms")
print("grid residuals vs real onset peaks (ms):", grid_residual_ms)
print(f"downbeat phase {phase}  ->  downbeats: {np.round(downbeats,3).tolist()}")
print()
print(f"{'beat':>4} {'time':>7} {'frame@30':>8} {'DB':>2} {'rms':>6} {'energy':>6}  curve")
for i, (a, b, r, lb, om) in enumerate(rows):
    lab = "pre" if i == 0 else f"{i-1:>3}"
    db = "*" if i > 0 and (i - 1 - phase_struct) % 4 == 0 else ""
    hit = " BASS" if any(abs(a - h) < 0.02 for h in bass_hits_on_grid) else ""
    print(f"{lab:>4} {a:7.3f} {a*30:8.2f} {db:>2} {r:6.3f} {energy[i]:6.2f}  "
          f"{'#'*int(round(energy[i]*30)):<30}{hit}")

json.dump({
    "source_offset_s": SEG_START,
    "duration_s": dur,
    "bpm": round(bpm, 3),
    "bpm_librosa": round(tempo, 3),
    "beat_period_s": P,
    "grid_fit_residuals_ms": grid_residual_ms,
    "beats": [round(float(t), 4) for t in grid],
    "half_beats": [round(float(t), 4) for t in halfbeats],
    "downbeat_phase": phase,
    "downbeats_bassflux": [round(float(t), 4) for t in downbeats],
    "downbeat_phase_structural": int(phase_struct),
    "downbeats": [round(float(t), 4) for t in downbeats_struct],
    "sections": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()} for s in sections],
    "onsets": [round(float(t), 4) for t in onsets],
    "strong_onsets": [round(float(t), 4) for t in strong_onsets],
    "bass_hits": bass_hits_on_grid,
    "vocal_punches": [round(float(t), 4) for t in vocal_punch],
    "energy_per_beat": [{"start": round(float(r[0]), 4), "end": round(float(r[1]), 4),
                         "rms": round(r[2], 4), "energy": round(float(x), 3),
                         "smooth": round(float(s), 3)}
                        for r, x, s in zip(rows, energy, smooth)],
}, open("beats.json", "w"), indent=1)
print("\nbass hits (snapped):", bass_hits_on_grid)
print(f"structural downbeat phase {phase_struct}: {np.round(downbeats_struct,3).tolist()}")
for sct in sections:
    print(f"  {sct['name']:<20} {sct['start']:6.3f} - {sct['end']:6.3f}")
print("vocal punches:", np.round(vocal_punch, 3).tolist())

"""Step 1.3/1.4: tempo, beats, downbeats, onsets, band envelopes -> beats.json;
snap word starts to nearest beat/onset within 80 ms -> words.json."""
import json, numpy as np, soundfile as sf, librosa, scipy.signal as ss

T0 = 172.0           # song time of segment t=0
FPS_ENV = 100        # envelope sample rate (Hz)

y, sr = sf.read('segment.wav'); y = y.mean(1)
yv, _ = sf.read('segment_vocals.wav'); yv = yv.mean(1)
dur = len(y) / sr
hop = sr // FPS_ENV

# tempo / beats
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop, units='time', tightness=200)
tempo = float(np.atleast_1d(tempo)[0])

# band split (zero-phase butterworth)
def band(lo, hi):
    if lo is None: b = ss.butter(4, hi, 'low', fs=sr, output='sos')
    elif hi is None: b = ss.butter(4, lo, 'high', fs=sr, output='sos')
    else: b = ss.butter(4, [lo, hi], 'band', fs=sr, output='sos')
    return ss.sosfiltfilt(b, y)
bands = {'bass': band(None, 150), 'mid': band(150, 4000), 'high': band(4000, None)}
env, ons = {}, {}
for k, x in bands.items():
    r = librosa.feature.rms(y=x, frame_length=hop * 4, hop_length=hop)[0]
    r = r / (np.percentile(r, 99) + 1e-9)
    env[k] = np.clip(r, 0, 1.5)
    o = librosa.onset.onset_strength(y=x, sr=sr, hop_length=hop)
    pk = librosa.onset.onset_detect(onset_envelope=o, sr=sr, hop_length=hop, units='time',
                                    backtrack=False, delta=0.25 if k != 'high' else 0.35)
    st = o[np.clip((pk * FPS_ENV).astype(int), 0, len(o) - 1)]
    ons[k] = [dict(t=round(float(t), 3), s=round(float(s / (o.max() + 1e-9)), 3)) for t, s in zip(pk, st)]
onsets_all = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, units='time', backtrack=True)

# downbeats: choose phase (0..3) whose beats carry most bass energy
bidx = (beats * FPS_ENV).astype(int)
bass_at = env['bass'][np.clip(bidx, 0, len(env['bass']) - 1)]
phase = int(np.argmax([bass_at[p::4].mean() for p in range(4)]))
downbeats = beats[phase::4]

# overall loudness envelope for the waveform ribbon
full = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
full = full / (full.max() + 1e-9)
vocal = librosa.feature.rms(y=yv, frame_length=hop * 2, hop_length=hop)[0]
vocal = vocal / (np.percentile(vocal, 99) + 1e-9)

out = dict(duration=dur, tempo=tempo, env_fps=FPS_ENV,
           beats=[round(float(b), 3) for b in beats],
           downbeats=[round(float(b), 3) for b in downbeats],
           onsets=[round(float(o), 3) for o in onsets_all],
           band_onsets=ons,
           env={k: [round(float(v), 3) for v in a] for k, a in env.items()},
           loud=[round(float(v), 3) for v in full],
           vocal=[round(float(v), 3) for v in vocal])
json.dump(out, open('beats.json', 'w'))
print(f'tempo {tempo:.1f} BPM, {len(beats)} beats, phase {phase}, onsets {len(onsets_all)}, '
      f'bass/mid/high onsets {[len(v) for v in ons.values()]}')
print('beats', ' '.join(f'{b:.2f}' for b in beats))

# ---- words: segment-relative times, snapping, display end ----
raw = json.load(open('align_raw.json'))
grid = np.array(sorted(set(list(beats) + list(onsets_all))))
words = []
for w in raw:
    s, e = w['start'] - T0, w['end'] - T0
    j = np.argmin(np.abs(grid - s)); d = grid[j] - s
    snapped = abs(d) <= 0.080
    words.append(dict(line=w['line'], idx=w['idx'], text=w['text'].upper(),
                      start_raw=round(s, 3), start=round(float(grid[j]) if snapped else s, 3),
                      end=round(e, 3), snapped=bool(snapped), snap_ms=round(d * 1000) if snapped else 0))
# sung end: extend aligned end while vocal energy stays up, never past next start
for i, w in enumerate(words):
    nxt = words[i + 1]['start'] if i + 1 < len(words) else dur
    e = w['end']; k = int(e * FPS_ENV)
    while k < len(vocal) - 1 and vocal[k] > 0.35 and (k + 1) / FPS_ENV < nxt - 0.03:
        k += 1
    w['end'] = round(max(e, k / FPS_ENV), 3)
    w['end'] = round(min(w['end'], nxt - 0.01) if i + 1 < len(words) else w['end'], 3)
    w['end'] = round(max(w['end'], w['start'] + 0.08), 3)
json.dump(words, open('words.json', 'w'), indent=1)
print(f"\n{'line':<5}{'word':<11}{'start':>8}{'end':>8}  snap")
for w in words:
    print(f"L{w['line']+1:<4}{w['text']:<11}{w['start']:8.3f}{w['end']:8.3f}  "
          f"{('%+dms' % w['snap_ms']) if w['snapped'] else '-'}")

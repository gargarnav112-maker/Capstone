"""Forced alignment of the user's exact lyrics to the separated vocal stem.

Acoustic model: Meta Omnilingual-ASR 300M CTC (ONNX, via sherpa-onnx release).
Text: per-word Gurmukhi transliteration of the user's lyrics (lyrics.py) so the
model's native script is used; display words are always the user's spelling.
Viterbi CTC alignment with a "wildcard" state before/after the text so audio
outside the lines (previous line tail, next repeat) is absorbed.
"""
import json, sys, numpy as np, soundfile as sf
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from ctc_common import load_tokens, logits
from lyrics import LINES, GURMUKHI

VOCALS = 'vocals_wide.wav'      # stem of song 166..198 s
STEM_T0 = 166.0
# (song_start, song_end, line indices) – windows overlap; each < 20 s (model is
# reliable on short inputs).
WINDOWS = [(171.0, 187.0, [0, 1, 2, 3]), (185.0, 196.0, [4, 5])]


def ctc_align(lp, ids):
    """Viterbi over blank-interleaved labels with free wildcard prefix/suffix.
    Returns per-label (first_frame, last_frame, mean_logprob)."""
    T, _ = lp.shape
    ext = [0]
    for i in ids:
        ext += [i, 0]
    S = len(ext)
    wild = lp[:, 1:].max(1)                    # best non-blank token per frame
    NEG = -1e9
    dp = np.full((T, S), NEG); bp = np.zeros((T, S), np.int32)
    pre = np.concatenate([[0], np.cumsum(wild)])  # cost of skipping frames 0..t-1
    for t in range(T):
        for s in range(S):
            e = lp[t, ext[s]]
            cands = [(NEG, 0)]
            if s <= 1:
                cands.append((pre[t], -1))      # start here after wildcard prefix
            if t > 0:
                cands.append((dp[t-1, s], s))
                if s >= 1: cands.append((dp[t-1, s-1], s-1))
                if s >= 2 and ext[s] != 0 and ext[s] != ext[s-2]:
                    cands.append((dp[t-1, s-2], s-2))
            v, b = max(cands)
            dp[t, s] = v + e; bp[t, s] = b
    suf = np.concatenate([np.cumsum(wild[::-1])[::-1], [0]])  # wildcard suffix
    end_scores = [(dp[t, s] + suf[t+1], t, s) for t in range(T) for s in (S-1, S-2)]
    _, t, s = max(end_scores)
    path = []
    while t >= 0 and s >= 0:
        path.append((t, s)); b = bp[t, s]
        if b == -1: break
        s = b; t -= 1
    path.reverse()
    out = {}
    for t, s in path:
        if ext[s] != 0:
            k = (s - 1) // 2
            f, l, acc = out.get(k, (t, t, []))
            out[k] = (min(f, t), max(l, t), acc + [lp[t, ext[s]]])
    return [(out[k][0], out[k][1], float(np.mean(out[k][2]))) for k in range(len(ids))]


def main():
    tok = load_tokens(); V = {v: k for k, v in tok.items()}
    x, sr = sf.read(VOCALS)
    words = []
    for w0, w1, lines in WINDOWS:
        a, b = int((w0 - STEM_T0) * sr), int((w1 - STEM_T0) * sr)
        sf.write('/tmp/claude-0/align_win.wav', x[a:b], sr)
        lp, dt = logits('/tmp/claude-0/align_win.wav')
        ids, owner = [], []
        for li in lines:
            for wi, gw in enumerate(GURMUKHI[li].split()):
                for c in gw:
                    ids.append(V[c]); owner.append((li, wi))
                if not (li == lines[-1] and wi == len(GURMUKHI[li].split()) - 1):
                    ids.append(V[' ']); owner.append(None)
        res = ctc_align(lp, ids)
        per = {}
        for (f, l, sc), o in zip(res, owner):
            if o is None: continue
            p = per.setdefault(o, [f, l, []]); p[0] = min(p[0], f); p[1] = max(p[1], l); p[2].append(sc)
        for (li, wi), (f, l, scs) in sorted(per.items()):
            words.append(dict(line=li, idx=wi, text=LINES[li].split()[wi],
                              gurmukhi=GURMUKHI[li].split()[wi],
                              start=round(w0 + f * dt, 3), end=round(w0 + (l + 1) * dt, 3),
                              score=round(float(np.mean(scs)), 3),
                              char_scores=[round(s, 2) for s in scs]))
    json.dump(words, open('align_raw.json', 'w'), ensure_ascii=False, indent=1)
    for w in words:
        flag = '  <-- LOW' if w['score'] < -2.5 else ''
        print(f"L{w['line']+1}  {w['text']:<10} {w['gurmukhi']:<8} {w['start']:8.3f} {w['end']:8.3f}  {w['score']:6.2f}{flag}")


if __name__ == '__main__':
    main()

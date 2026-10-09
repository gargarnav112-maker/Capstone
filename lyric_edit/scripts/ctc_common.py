import numpy as np, soundfile as sf, onnxruntime as ort, librosa
MDIR = '/tmp/claude-0/models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-2025-11-12'
def load_tokens(mdir=MDIR):
    toks = {}
    for line in open(f'{mdir}/tokens.txt', encoding='utf8'):
        line = line.rstrip('\n'); sym, idx = line.rsplit(' ', 1)
        toks[int(idx)] = sym if sym else ' '
    return toks
def logits(wav_path, mdir=MDIR, model='model.onnx'):
    x, sr = sf.read(wav_path, dtype='float32', always_2d=True)
    x = librosa.resample(x.mean(1), orig_sr=sr, target_sr=16000)
    x = (x - x.mean()) / (x.std() + 1e-7)
    s = ort.InferenceSession(f'{mdir}/{model}', providers=['CPUExecutionProvider'])
    lg = s.run(None, {'x': x[None].astype(np.float32)})[0][0]
    lg = lg - lg.max(1, keepdims=True); lp = lg - np.log(np.exp(lg).sum(1, keepdims=True))
    return lp, len(x) / 16000 / lp.shape[0]

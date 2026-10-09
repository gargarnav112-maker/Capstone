"""Vocal separation with UVR-MDX-NET-Voc_FT via sherpa-onnx."""
import sys, numpy as np, soundfile as sf, sherpa_onnx as so
model, wav_in, wav_out, inst_out = sys.argv[1:5]
cfg = so.OfflineSourceSeparationConfig(
    model=so.OfflineSourceSeparationModelConfig(
        uvr=so.OfflineSourceSeparationUvrModelConfig(model=model), num_threads=4))
sep = so.OfflineSourceSeparation(cfg)
x, sr = sf.read(wav_in, dtype='float32', always_2d=True)
out = sep.process(sample_rate=sr, samples=np.ascontiguousarray(x.T))
stems = [np.array(s.data) for s in out.stems]
print('stems', len(stems), [s.shape for s in stems], out.sample_rate)
sf.write(wav_out, stems[0].T, out.sample_rate)
sf.write(inst_out, stems[1].T, out.sample_rate)

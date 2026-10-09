import sherpa_onnx, soundfile as sf, sys
d='models/w/sherpa-onnx-whisper-large-v3/'
a,sr=sf.read(sys.argv[1],dtype='float32')
L=float(sys.argv[2]); H=float(sys.argv[3])
for lang in sys.argv[4:]:
    r=sherpa_onnx.OfflineRecognizer.from_whisper(encoder=d+'large-v3-encoder.int8.onnx',decoder=d+'large-v3-decoder.int8.onnx',tokens=d+'large-v3-tokens.txt',language=lang,task='transcribe',num_threads=4,tail_paddings=1000)
    t=0.0
    while t<len(a)/sr-0.5:
        s=r.create_stream(); s.accept_waveform(sr,a[int(t*sr):int((t+L)*sr)]); r.decode_stream(s)
        print(lang,f'{t:5.1f}-{t+L:5.1f}','|',s.result.text,flush=True); t+=H

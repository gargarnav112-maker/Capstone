import sherpa_onnx, soundfile as sf, time, sys
d='models/w/sherpa-onnx-whisper-large-v3/'
a,sr=sf.read(sys.argv[1],dtype='float32')
for spec in sys.argv[2:]:
    lang,task=spec.split(':')
    t=time.time()
    r=sherpa_onnx.OfflineRecognizer.from_whisper(encoder=d+'large-v3-encoder.int8.onnx',decoder=d+'large-v3-decoder.int8.onnx',tokens=d+'large-v3-tokens.txt',language=lang,task=task,num_threads=4,tail_paddings=1000)
    s=r.create_stream(); s.accept_waveform(sr,a); r.decode_stream(s)
    res=s.result
    print(lang,task,round(time.time()-t,1),'|',res.text, flush=True)
    ts=getattr(res,'timestamps',None)
    if ts: print('  ts',list(zip(res.tokens,[round(x,2) for x in ts])))

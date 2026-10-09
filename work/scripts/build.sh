#!/bin/bash
# Parallel render (4 chunks) -> concat -> final encodes. Env LYRICS/SCENES may override inputs. Arg1 = output prefix.
set -e
cd "$(dirname "$0")/.."
P=${1:-out/ashke_lyric_edit}
mkdir -p out/chunks qa
rm -f out/chunks/*.mkv out/chunks/*.json
N=480; K=4; step=$((N / K))
pids=()
for i in $(seq 0 $((K - 1))); do
  a=$((i * step)); b=$(((i + 1) * step)); [ $i -eq $((K - 1)) ] && b=$N
  python3 scripts/render.py --frames $a:$b --out out/chunks/c$i.mkv --log out/chunks/log$i.json > out/chunks/r$i.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
: > out/chunks/list.txt; for i in $(seq 0 $((K - 1))); do echo "file 'c$i.mkv'" >> out/chunks/list.txt; done
python3 -c "import json,glob;json.dump(sum([json.load(open(f)) for f in sorted(glob.glob('out/chunks/log*.json'))],[]),open('qa/text_log.json','w'))"
# master: 1080x1920 30fps H.264 ~20 Mbps, AAC 320k, exactly 16.000 s
ffmpeg -v error -y -f concat -safe 0 -i out/chunks/list.txt -i segment.wav -map 0:v -map 1:a \
  -c:v libx264 -preset slow -b:v 20M -maxrate 24M -bufsize 40M -pix_fmt yuv420p -profile:v high -r 30 \
  -c:a aac -b:a 320k -ar 48000 -t 16.000 -movflags +faststart ${P}.mp4
# 480p preview
ffmpeg -v error -y -i ${P}.mp4 -vf scale=480:854:flags=lanczos -c:v libx264 -crf 23 -preset medium -c:a aac -b:a 128k -movflags +faststart ${P}_480p.mp4
# < 25 MB version (two-pass target 11 Mbps total)
ffmpeg -v error -y -i ${P}.mp4 -c:v libx264 -preset slow -b:v 10.5M -pass 1 -passlogfile out/chunks/p2 -an -f mp4 /dev/null
ffmpeg -v error -y -i ${P}.mp4 -c:v libx264 -preset slow -b:v 10.5M -pass 2 -passlogfile out/chunks/p2 -c:a aac -b:a 256k -movflags +faststart ${P}_under25MB.mp4
ls -la ${P}*.mp4

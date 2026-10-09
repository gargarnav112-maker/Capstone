"""Reframe the music video to 9:16 (subject-centred crop), drop duplicate screen-capture
frames, then optical-flow interpolate to 120 fps so any speed (incl. 30% slow-mo) is smooth."""
import subprocess, sys
from sources import VIDEO_AREA, CROP_W, crop_x_expr
SRC = sys.argv[1]
w, h, x, y = VIDEO_AREA
vf = (f"crop={w}:{h}:{x}:{y},crop={CROP_W}:{h}:x='{crop_x_expr()}':y=0,"
      "mpdecimate=hi=64*6:lo=64*3:frac=0.5,"
      "minterpolate=fps=120:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd=fdiff:scd_threshold=8,"
      "format=yuv444p")
subprocess.run(["ffmpeg", "-v", "error", "-stats", "-y", "-i", SRC, "-an", "-vf", vf,
                "-c:v", "libx264", "-crf", "6", "-preset", "medium", "-g", "120",
                "-r", "120", "src120.mp4"], check=True)

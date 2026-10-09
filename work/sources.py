"""Source shot library for the screen-recorded music video (times in recording seconds).

Each entry: usable in/out range (kept clear of the shot boundaries so optical-flow
interpolation never blends two shots), framing, and the horizontal subject centre
(fraction of the 2134 px wide video area) at the start / end of the range, used for
the 9:16 smart crop.
"""

VIDEO_AREA = (2134, 1062, 244, 72)  # w, h, x, y of the music video inside the recording
CROP_W = 598                         # 1062 * 9/16, even

SHOTS = {
    #  name        t_in   t_out  framing   cx_in  cx_out  description
    "TEA":       (0.03,  2.50,  "medium",  0.55,  0.53,  "tea party, Karan seated, butler behind"),
    "MANOR":     (2.60,  3.47,  "wide",    0.47,  0.47,  "manor + vintage cars, Karan gesturing"),
    "HORSE_MED": (3.53,  3.97,  "medium",  0.70,  0.70,  "polo field, Karan in plaid jacket"),
    "MANOR2":    (4.08,  4.30,  "medium",  0.47,  0.47,  "manor, hand to face"),
    "SUNSET":    (4.40,  4.95,  "close",   0.55,  0.52,  "golden-hour close-up with sun flare"),
    "HORSES":    (5.08,  6.78,  "wide",    0.74,  0.80,  "riders behind Karan, wide"),
    "CANDLES":   (6.87,  8.30,  "detail",  0.45,  0.52,  "candlelit banquet table insert"),
    "TABLE":     (8.40, 10.60,  "wide",    0.52,  0.48,  "banquet wide, guests"),
    "KTABLE":    (10.70, 13.33, "medium",  0.48,  0.55,  "Karan at the table, flat-hand gesture"),
    "LOWCU":     (13.80, 16.40, "close",   0.58,  0.58,  "low-angle close-up, candle-lit"),
}


def crop_x_expr():
    """ffmpeg crop x expression: piecewise-linear subject centre per source shot."""
    W = VIDEO_AREA[0]
    names = sorted(SHOTS.values(), key=lambda s: s[0])
    # boundaries = real cuts in the source
    bounds = [0.0, 2.567, 3.5, 4.0, 4.367, 5.0, 6.833, 8.35, 10.65, 13.55, 99]
    expr = "0"
    for (a, b), s in reversed(list(zip(zip(bounds[:-1], bounds[1:]), names))):
        t0, t1, _, c0, c1 = s[0], s[1], s[2], s[3], s[4]
        cx = f"({c0}+({c1}-{c0})*clip((t-{t0})/{t1 - t0},0,1))"
        x = f"clip({cx}*{W}-{CROP_W // 2},0,{W - CROP_W})"
        expr = f"if(lt(t,{b}),{x},{expr})"
    return expr

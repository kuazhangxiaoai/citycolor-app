

def convert_rgb2hsv(r, g, b):
    rn, gn, bn = r /255.0, g / 255.0, b / 255.0
    _max, _min = max(rn, gn, bn), min(rn, gn, bn)
    delta = _max - _min
    v = 0.5 * (_max + _min)
    s = delta / (1 - abs(2*v - 1)) if delta != 0 else 0

    if delta == 0:
        h = 0
    elif _max == rn:
        h = 60 * (((gn - bn) / delta) % 6)
    elif _max == gn:
        h = 60 * (((bn - rn) / delta) + 2)
    else:
        h = 60 * (((rn - gn) / delta) + 4)

    if h < 0:
        h += 360

    return (h, s, v)

def convert_hsv2rgb(h, s, v):
    c = (1 - abs(2 * v - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c / 2

    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h <180:
        r, g, b = 0, c, x
    elif 180 <= h <240:
        r, g, b = 0, x, c
    elif 240 <= h <300:
        r, g, b =  x, 0, c
    else:
        r, g, b = c, 0, x

    r = int((r + m) * 255)
    g = int((g + m) * 255)
    b = int((b + m) * 255)

    return (r, g, b)

def rgb_to_hsv(R, G, B):
    r = R / 255.0
    g = G / 255.0
    b = B / 255.0

    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin

    # Hue
    if delta == 0:
        H = 0
    elif cmax == r:
        H = 60 * (((g - b) / delta) % 6)
    elif cmax == g:
        H = 60 * (((b - r) / delta) + 2)
    else:
        H = 60 * (((r - g) / delta) + 4)

    # Saturation（圆柱半径）
    if cmax == 0:
        S = 0
    else:
        S = delta / cmax

    # Value（高度）
    V = cmax

    return H, S, V

def hsv_to_rgb(H, S, V):
    H = H % 360  # 防止越界

    C = V * S
    X = C * (1 - abs((H / 60) % 2 - 1))
    m = V - C

    if H < 60:
        r, g, b = C, X, 0
    elif H < 120:
        r, g, b = X, C, 0
    elif H < 180:
        r, g, b = 0, C, X
    elif H < 240:
        r, g, b = 0, X, C
    elif H < 300:
        r, g, b = X, 0, C
    else:
        r, g, b = C, 0, X

    R = int((r + m) * 255)
    G = int((g + m) * 255)
    B = int((b + m) * 255)

    # 防止浮点误差
    R = max(0, min(255, R))
    G = max(0, min(255, G))
    B = max(0, min(255, B))

    return R, G, B
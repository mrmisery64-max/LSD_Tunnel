"""
L Tunnel - an endless, never-repeating tunnel.

Every parameter (speed, twist, symmetry, colour palette, wall patterns, camera wobble,
trails...) constantly glides toward new random targets, so the visuals never loop.

Install:   pip install pygame numpy
Run:       python lsd_tunnel.py

Controls:
  ESC / Q     quit
  F           toggle fullscreen
  SPACE       reshuffle everything (fast morph to a new look)
  S           save screenshot
  UP / DOWN   sharper / faster (render resolution)

WARNING: contains rapidly shifting colours and patterns. Not for people with
photosensitive epilepsy.
"""
# --- imports ---
import math    # sin/cos/floor for scalar (non-array) maths
import random  # random targets, durations and bursts
import time    # seeding the RNG and timestamping screenshots

import numpy as np  # fast per-pixel maths on whole arrays at once
import pygame       # window, input, drawing to the screen

# --- global settings ---
WIN_W, WIN_H = 1280, 720  # starting window size in pixels
FPS = 60                  # target frames per second
TAU = 2 * math.pi         # one full turn in radians (handy for sin/cos cycles)

# seed with the current time in nanoseconds so every run looks different
random.seed(time.time_ns())


class Drift:
    """A value that wanders smoothly between random targets forever,
      saw this on a instagram video and thought it was cool."""

    def __init__(self, lo, hi, tmin=4.0, tmax=12.0):
        # lo/hi = allowed range of the value
        # tmin/tmax = how long (seconds) each glide to a new target can take
        self.lo, self.hi, self.tmin, self.tmax = lo, hi, tmin, tmax
        self.v = random.uniform(lo, hi)  # start at a random value in range
        self.new_target()

    def new_target(self, duration=None):
        # glide FROM the current value (a) TO a fresh random value (b)
        self.a = self.v
        self.b = random.uniform(self.lo, self.hi)
        # use a forced duration if given (used for SPACE / bursts), else random
        self.T = duration if duration else random.uniform(self.tmin, self.tmax)
        self.t = 0.0  # time elapsed in this glide

    def update(self, dt):
        # advance the glide by dt seconds and return the current value
        self.t += dt
        if self.t >= self.T:
            # reached the target: snap to it and immediately pick another one
            self.v = self.b
            self.new_target()
        else:
            s = self.t / self.T  # progress 0..1 through the glide
            s = s * s * (3 - 2 * s)  # smoothstep easing
            self.v = self.a + (self.b - self.a) * s  # blend between a and b
        return self.v


# All the knobs that morph over time
PARAMS = {
    "speed":   Drift(0.4, 2.6),     # how fast we fly forward
    "rotspd":  Drift(-1.6, 1.6),    # tunnel spin
    "depth":   Drift(0.25, 0.8),    # perspective strength
    "twist":   Drift(-2.5, 2.5),    # corkscrew amount
    "ringf":   Drift(0.4, 3.0),     # ring frequency
    "sym":     Drift(1.0, 10.0),    # radial symmetry (spokes)
    "spiral":  Drift(-4.0, 4.0),    # spiral tightness
    "sym2":    Drift(1.0, 6.0),     # secondary symmetry
    "f3":      Drift(1.5, 9.0),     # plasma frequency
    "warp":    Drift(0.0, 3.5),     # wall warping
    "wobble":  Drift(0.0, 1.0),     # wall ripple
    "w1":      Drift(0.0, 1.0),     # pattern weights
    "w2":      Drift(0.0, 1.0),
    "w3":      Drift(0.0, 1.0),
    "w4":      Drift(0.0, 1.0),
    "hspread": Drift(0.15, 1.3),    # how much colour varies across pattern
    "hz":      Drift(-0.35, 0.35),  # colour change with depth
    "huespd":  Drift(-0.4, 0.4),    # global colour cycling
    "o1":      Drift(0.0, 1.0),     # palette offsets
    "o2":      Drift(0.0, 1.0),
    "o3":      Drift(0.0, 1.0),
    "contrast": Drift(0.7, 2.2),
    "fog":     Drift(1.5, 7.0),     # darkness at the far end
    "glow":    Drift(0.0, 1.2),     # light at the end of the tunnel
    "trail":   Drift(0.0, 0.65),    # feedback trails
    "camamp":  Drift(0.0, 0.45),    # camera sway amount
    "camfx":   Drift(0.2, 1.2),
    "camfy":   Drift(0.2, 1.2),
}
# notes on the unlabelled knobs above:
#   w2/w3/w4  - how strongly each of the 4 wall patterns shows (w1 = rings,
#               w2 = spokes/spirals, w3 = plasma, w4 = checkerboard)
#   o2/o3     - palette offsets for the green and blue channels (o1 is red);
#               shifting them changes which colours the palette produces
#   contrast  - how hard colours are pushed away from mid-grey
#   camfx/fy  - how fast the camera sways left/right (x) and up/down (y)


class Grid:
    """Precomputed pixel coordinates for the low-res render buffer."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        asp = w / h  # aspect ratio, so circles stay round on wide screens
        # x runs -asp..asp and y runs -1..1, with (0,0) at the screen centre
        xs = np.linspace(-asp, asp, w, dtype=np.float32)
        ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
        self.X, self.Y = np.meshgrid(xs, ys, indexing="ij")  # shape (w, h)
        # feedback: sample slightly toward the centre so trails rush outward
        zoom = 1.035
        # for every pixel, the index of the pixel slightly closer to the centre
        # in the previous frame (clipped so we never index off the edge)
        ix = ((np.arange(w) - w / 2) / zoom + w / 2).astype(np.int32).clip(0, w - 1)
        iy = ((np.arange(h) - h / 2) / zoom + h / 2).astype(np.int32).clip(0, h - 1)
        # shaped so they broadcast into a full (w, h) lookup
        self.fx, self.fy = ix[:, None], iy[None, :]
        # last rendered frame (starts black), used for the trail effect
        self.prev = np.zeros((w, h, 3), dtype=np.float32)


def sym_sin(ang, n, phase):
    """sin(ang*n + phase) that stays seamless for non-integer n by blending
    the two nearest integer symmetries."""
    n0 = math.floor(n)  # lower whole number of spokes
    f = n - n0          # fractional part = how far toward n0+1 we are
    return (1 - f) * np.sin(ang * n0 + phase) + f * np.sin(ang * (n0 + 1) + phase)


def render(g, P, t, travel, rot, hue_phase, cx, cy):
    # Draws one frame. g = Grid, P = current knob values, t = time,
    # travel/rot/hue_phase = accumulated motion, cx/cy = camera offset.
    # Returns a (w, h, 3) uint8 image.

    # position of each pixel relative to the (swaying) tunnel centre
    dx = g.X - cx
    dy = g.Y - cy
    # convert to polar coordinates: r = distance from centre, a = angle
    r = np.sqrt(dx * dx + dy * dy) + 1e-4  # tiny offset avoids divide-by-zero
    a = np.arctan2(dy, dx)

    depth = P["depth"] / r            # perspective: small radius = far away
    z = depth + travel                # distance along the tunnel
    ang = a + depth * P["twist"] + rot  # angle around the wall, plus twist and spin

    # ripple the walls
    z = z + P["wobble"] * 0.25 * np.sin(ang * 3 + t * 1.7)

    # wall patterns
    p1 = np.sin(z * P["ringf"] * TAU)                             # rings
    p2 = sym_sin(ang, P["sym"], z * P["spiral"])                  # spokes / spirals
    p3 = np.sin(z * P["f3"] + P["warp"] * sym_sin(ang, P["sym2"], z * 0.7 + t))  # plasma
    p4 = p1 * p2                                                  # checkerboard

    # mix the four patterns together using their drifting weights
    v = P["w1"] * p1 + P["w2"] * p2 + P["w3"] * p3 + P["w4"] * p4
    v = v * (1.0 - np.exp(-r * 10.0))  # soften aliasing at the vanishing point

    # cosine palette
    # turn the pattern value + depth + global cycle into a "hue" number,
    # then each colour channel is a cosine wave of that hue with its own offset
    hue = v * P["hspread"] + z * P["hz"] + hue_phase
    rgb = np.empty((g.w, g.h, 3), dtype=np.float32)
    rgb[..., 0] = 0.5 + 0.5 * np.cos(TAU * (hue + P["o1"]))         # red
    rgb[..., 1] = 0.5 + 0.5 * np.cos(TAU * (hue + P["o2"] + 0.33))  # green
    rgb[..., 2] = 0.5 + 0.5 * np.cos(TAU * (hue + P["o3"] + 0.67))  # blue

    # punch up contrast
    rgb = np.clip((rgb - 0.5) * P["contrast"] + 0.5, 0.0, 1.0)

    # depth fog + glowing core
    fog = 1.0 - np.exp(-r * P["fog"])      # 0 at the centre (dark), 1 at the edges
    glow = P["glow"] * np.exp(-r * 9.0)    # bright spot right at the centre
    rgb = rgb * fog[..., None] + glow[..., None]

    # feedback trails
    trail = P["trail"]
    if trail > 0.01:
        # blend in a slightly zoomed copy of the last frame -> streaks outward
        rgb = rgb * (1 - trail) + g.prev[g.fx, g.fy] * trail
    g.prev = rgb  # remember this frame for next time

    # convert 0..1 floats to 0..255 bytes for pygame
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def main():
    # --- window setup ---
    pygame.init()
    pygame.display.set_caption("LSD Tunnel")
    screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)
    clock = pygame.time.Clock()  # used to cap the frame rate and measure dt

    scale = 3  # render at 1/scale resolution, then upscale smoothly
    fullscreen = False

    def make_grid():
        # (re)build the render grid to match the current window size and scale
        sw, sh = screen.get_size()
        return Grid(max(sw // scale, 16), max(sh // scale, 16))

    g = make_grid()
    # t = total time, travel = distance flown, rot = total spin,
    # hue_phase = total colour cycling (all accumulate every frame)
    t = travel = rot = hue_phase = 0.0
    next_burst = random.uniform(15, 35)  # seconds until the next random "wave"

    # --- main loop: runs once per frame ---
    running = True
    while running:
        # seconds since last frame, capped at 0.1 so a hiccup doesn't cause a jump
        dt = min(clock.tick(FPS) / 1000.0, 0.1)
        t += dt

        # --- handle input / window events ---
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                # window close button
                running = False
            elif e.type == pygame.VIDEORESIZE and not fullscreen:
                # window was resized: recreate it and rebuild the grid
                screen = pygame.display.set_mode(e.size, pygame.RESIZABLE)
                g = make_grid()
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    # quit
                    running = False
                elif e.key == pygame.K_f:
                    # toggle fullscreen ((0, 0) = use the monitor's resolution)
                    fullscreen = not fullscreen
                    if fullscreen:
                        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                    else:
                        screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)
                    g = make_grid()
                elif e.key == pygame.K_SPACE:
                    # every knob heads to a new target quickly (1-2.5 s)
                    for d in PARAMS.values():
                        d.new_target(random.uniform(1.0, 2.5))
                elif e.key == pygame.K_s:
                    # save a screenshot named with the current unix time
                    name = f"tunnel_{int(time.time())}.png"
                    pygame.image.save(screen, name)
                    print("saved", name)
                elif e.key == pygame.K_UP and scale > 1:
                    # render at higher resolution (sharper, slower)
                    scale -= 1
                    g = make_grid()
                elif e.key == pygame.K_DOWN and scale < 8:
                    # render at lower resolution (blurrier, faster)
                    scale += 1
                    g = make_grid()

        # occasional spontaneous "wave" where a random handful of knobs jump quickly
        next_burst -= dt
        if next_burst <= 0:
            for d in random.sample(list(PARAMS.values()), k=random.randint(4, 10)):
                d.new_target(random.uniform(1.5, 4.0))
            next_burst = random.uniform(15, 35)

        # step every knob forward and grab its current value
        P = {k: d.update(dt) for k, d in PARAMS.items()}

        # accumulate motion (speed * time = distance, etc.)
        travel += P["speed"] * dt
        rot += P["rotspd"] * dt
        hue_phase += P["huespd"] * dt

        # camera sway: we swerve through the tunnel
        # big slow sway from camamp/camfx/camfy plus a small constant jiggle
        amp = P["camamp"]
        cx = amp * math.sin(t * P["camfx"]) + 0.08 * math.sin(t * 2.3)
        cy = amp * 0.6 * math.sin(t * P["camfy"] + 1.3) + 0.06 * math.cos(t * 1.9)

        # --- draw ---
        frame = render(g, P, t, travel, rot, hue_phase, cx, cy)
        surf = pygame.surfarray.make_surface(frame)  # numpy array -> pygame image
        # stretch the small render up to the full window with smoothing
        screen.blit(pygame.transform.smoothscale(surf, screen.get_size()), (0, 0))
        pygame.display.flip()  # show the finished frame

    pygame.quit()


# only run when executed directly (not when imported as a module)
if __name__ == "__main__":
    main()

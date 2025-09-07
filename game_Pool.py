from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import math, random
import traceback

WIN_W, WIN_H = 1000, 800

# Play area (felt) extents (centered at origin on z = 0)
HALF_X = 300.0
HALF_Y = 150.0
FELT_Z = 0.0

# Visual sizes
WALL_H = 20.0
WALL_T = 15.0
BALL_R = 10.0

# Pockets (disks, fully inside table so nothing sticks out)
POCKET_R = 18.0
POCKET_INSET_CORNER = 20
POCKET_INSET_SIDE   = 10

# Camera (orbit around Z with arrow keys)
cam_radius = 600.0
cam_height = 350.0
cam_yaw_deg = 0.0
FOVY = 60.0

# Balls: 8 colored + 1 cue (static)
balls = []

# Physics & collisions
FRICTION_COEFF          = 0.985     # per-frame damping
SLEEP_EPS               = 0.01      # speed below this -> clamp to 0
COLLISION_RESTITUTION   = .9      # ball-ball bounce (1.0 = perfectly elastic)
CUSHION_RESTITUTION     = .9      # ball-cushion bounce
POS_ITERATIONS          = 5         # resolve passes per frame
EPS                     = 1e-6
CONTACT_TOLERANCE       = 0.25      # small allowance beyond exact touch for impulse
MAX_IMPULSE             = 1000.0       # cap per-contact impulse (prevents spikes)
MAX_SPEED               = 41000.0      # optional: cap raw speed after collisions
BALL_RESTITUTION        = 0.94
PENETRATION_SLOP      = 0.05   # ignore tiny overlaps (units)
PENETRATION_PERCENT   = 0.8    # correct only 80% of the remaining overlap
MIN_REL_SPEED_FOR_HIT = 0.02   # skip impulse if approach is basically zero


# ====== Cue state (single source of truth) ======
cue_angle   = 0.0          # initial yaw (points into table)
cue_power   = 0.0            # charge amount (0..100)
charging    = False          # charging with space held
cue_velocity = [0.0, 0.0]    # cue ball XY velocity after strike

cue_length  = 255.0          # stick length
cue_radius  = 3.0            # stick thickness
cue_gap     = 1.0            # tip gap from ball to avoid z-fighting

# Butt (rear) vertical offset; start at lowest legal height (clear the cushion)
cue_butt_z  = max(0.0, WALL_H + BALL_R)

# Stroke dynamics
PULL_MAX     = 80.0          # max pull-back while charging (visual)
HIT_MAX      = 40.0          # forward push distance when striking (visual)
cue_stroked  = False         # set on release until balls stop
stroke_offset = 0.0          # forward offset after impact (visual)
impact_base   = None         # (x,y,z) base position at impact for stick rendering



# Potting Logic
POT_THRESHOLD = POCKET_R - BALL_R * 1.2   # tighten if potting feels too easy


# --- Game state ---
GAME_RUNNING = 0
GAME_WON     = 1
GAME_OVER    = 2
game_state   = GAME_RUNNING

# --- Lives (scratches) ---
LIVES_MAX = 3
lives     = LIVES_MAX


def seed_balls():
    """Populate balls: 7 colored, 1 black in row 2 middle, and cue ball."""
    random.seed(42)

    colors7 = [
        (1.0, 0.2, 0.2), (1.0, 0.6, 0.2), (1.0, 1.0, 0.2),
        (0.2, 1.0, 0.2), (0.2, 1.0, 1.0), (0.2, 0.4, 1.0),
        (0.8, 0.2, 1.0),
    ]
    BLACK = (0.02, 0.02, 0.02)

    start_x, start_y = 150.0, 0.0
    gap = 2.0
    dx = 2.0 * BALL_R + gap
    dy = dx * (3 ** 0.5) * 0.5

    balls.clear()

    slots = []
    row_counts = [1, 3, 4]
    for r, cnt in enumerate(row_counts):
        x = start_x + r * dy
        # center this row horizontally around start_x
        y0 = start_y - 0.5*(cnt - 1) * dx
        for c in range(cnt):
            y = y0 + c * dx
            slots.append((x, y))

    # Black ball in row 2, middle = slot[2]
    black_idx = 2
    bx, by = slots[black_idx]
    balls.append({
        "id": 8,
        "color": BLACK,
        "pos": [bx, by, FELT_Z + BALL_R],
        "vel": [0.0, 0.0],
        "alive": True,
    })

    # Fill other 7 slots with colored balls
    ci = 0
    for i, (px, py) in enumerate(slots):
        if i == black_idx:
            continue
        balls.append({
            "id": ci + 1,
            "color": colors7[ci],
            "pos": [px, py, FELT_Z + BALL_R],
            "vel": [0.0, 0.0],
            "alive": True,
        })
        ci += 1

    # Cue ball (white)
    balls.append({
        "id": 0,
        "color": (1.0, 1.0, 1.0),
        "pos": [-150.0, 0.0, FELT_Z + BALL_R],
        "vel": [0.0, 0.0],
        "alive": True,
    })



# (Debug logging removed)

def setup_camera():
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(FOVY, WIN_W/float(WIN_H), 0.1, 2000.0)    
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    theta = math.radians(cam_yaw_deg)
    x = cam_radius * math.cos(theta)
    y = cam_radius * math.sin(theta)
    z = cam_height
    gluLookAt(x, y, z, 0, 0, 0, 0, 0, 1)                      

def draw_felt():
    glColor3f(0.05, 0.45, 0.1)
    glBegin(GL_QUADS)
    glVertex3f(-HALF_X, -HALF_Y, FELT_Z)
    glVertex3f( HALF_X, -HALF_Y, FELT_Z)
    glVertex3f( HALF_X,  HALF_Y, FELT_Z)
    glVertex3f(-HALF_X,  HALF_Y, FELT_Z)
    glEnd()

def draw_walls():
    glColor3f(0.35, 0.18, 0.05)
    def wall(x, y, sx, sy, sz):
        glPushMatrix()
        glTranslatef(x, y, FELT_Z + sz*0.5)
        glScalef(sx, sy, sz)
        glutSolidCube(1.0)
        glPopMatrix()
    wall(0,  HALF_Y + WALL_T*0.5, 2*HALF_X + 2*WALL_T, WALL_T, WALL_H)
    wall(0, -HALF_Y - WALL_T*0.5, 2*HALF_X + 2*WALL_T, WALL_T, WALL_H)
    wall(-HALF_X - WALL_T*0.5, 0, WALL_T, 2*HALF_Y + 2*WALL_T, WALL_H)
    wall( HALF_X + WALL_T*0.5, 0, WALL_T, 2*HALF_Y + 2*WALL_T, WALL_H)

def pocket_centers():
    c, s = POCKET_INSET_CORNER, POCKET_INSET_SIDE
    return [
        (-HALF_X + c, -HALF_Y + c),
        ( 0.0,        -HALF_Y + s),
        ( HALF_X - c, -HALF_Y + c),
        (-HALF_X + c,  HALF_Y - c),
        ( 0.0,         HALF_Y - s),
        ( HALF_X - c,  HALF_Y - c),
    ]

def draw_pockets():
    
    for (px, py) in pocket_centers():
        glPushMatrix()
        glTranslatef(px, py, FELT_Z + 0.5)     
        glColor3f(0.0, 0.0, 0.0)              
        gluDisk(gluNewQuadric(), 0.0, POCKET_R, 64, 1)
        glPopMatrix()

def draw_balls():
    for b in balls:
        if not b["alive"]:
            continue
        glPushMatrix()
        glColor3f(*b["color"])
        glTranslatef(b["pos"][0], b["pos"][1], b["pos"][2])
        gluSphere(gluNewQuadric(), BALL_R, 20, 16)
        glPopMatrix()


def display():
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glViewport(0, 0, WIN_W, WIN_H)
    setup_camera()

    draw_felt()
    draw_pockets()
    draw_walls() 
    draw_balls()
    draw_cue()
    draw_hud()
    
    glutSwapBuffers()

def draw_hud():
    glDisable(GL_DEPTH_TEST)

    """Draw a simple screen-space HUD with the current power bar."""
    # save matrices
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, WIN_W, 0, WIN_H, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    # bar parameters
    pad = 10
    bar_w = 300
    bar_h = 24
    x = pad
    y = WIN_H - pad - bar_h

    # border
    glColor3f(0.0, 0.0, 0.0)
    glBegin(GL_QUADS)
    glVertex2f(x-2, y-2)
    glVertex2f(x+bar_w+2, y-2)
    glVertex2f(x+bar_w+2, y+bar_h+2)
    glVertex2f(x-2, y+bar_h+2)
    glEnd()

    # background
    glColor3f(0.2, 0.2, 0.2)
    glBegin(GL_QUADS)
    glVertex2f(x, y)
    glVertex2f(x+bar_w, y)
    glVertex2f(x+bar_w, y+bar_h)
    glVertex2f(x, y+bar_h)
    glEnd()

    # filled portion
    pct = max(0.0, min(1.0, cue_power / 100.0))
    glColor3f(0.1, 0.8, 0.2)
    glBegin(GL_QUADS)
    glVertex2f(x, y)
    glVertex2f(x+bar_w*pct, y)
    glVertex2f(x+bar_w*pct, y+bar_h)
    glVertex2f(x, y+bar_h)
    glEnd()

    # text (power percent) - use GLUT bitmap font
    glColor3f(1.0, 1.0, 1.0)
    glRasterPos2f(x + bar_w + 10, y + 6)
    txt = f"Power: {int(pct*100)}%"
    for ch in txt:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(ch))


    # status lines (lives + state)

    glColor3f(1.0, 1.0, 1.0)
    status = [f"Lives: {lives} / {LIVES_MAX}"]
    if game_state == GAME_WON:
        status.append("YOU WIN! Press R to restart.")
    elif game_state == GAME_OVER:
        status.append("GAME OVER. Press R to restart.")

    # --- add rules under Lives ---
    status.extend([
        "Rules:",
        "- Pot balls 1-7 first.",
        "- Pot the 8-ball last to win.",
        "- Potting 8-ball early = Game Over.",
        "- Cue ball in pocket = lose a life.",
        "- Lose 3 lives = Game Over.",
        "- Press R to restart."
    ])

    ty = y - 28
    for line in status:
        glRasterPos2f(pad, ty)
        for ch in line:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(ch))
        ty -= 20
    # restore matrices
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()
    glEnable(GL_DEPTH_TEST)

# ================= Cue Stick Mechanics =================

# cue_angle = 0.0        # angle around cue ball
# cue_power = 0.0        # current power
# charging = False       # are we charging power?
# cue_velocity = [0.0, 0.0]  # velocity of cue ball after hit
# cue_butt_z = 0.0       # vertical offset of the butt (end away from ball)
# PULL_MAX = 80.0       # maximum pull-back while charging (world units)
# HIT_MAX = 40.0        # maximum forward push on hit (how far cue moves forward when striking)
# cue_stroked = False   # set after releasing space until ball stops
# stroke_offset = 0.0   # how far forward the cue is after striking (positive)
# impact_base = None    # fixed world (x,y,z) position of cue base at impact

def cue_limits(cx, cy, angle_deg):
    dx = math.cos(math.radians(angle_deg))
    dy = math.sin(math.radians(angle_deg))

    # butt goes opposite aim
    ux, uy = -dx, -dy

    # distance to wall behind the ball
    t_candidates = []
    if abs(ux) > EPS:
        t_candidates += [( -HALF_X - cx) / ux, ( HALF_X - cx) / ux]
    if abs(uy) > EPS:
        t_candidates += [( -HALF_Y - cy) / uy, ( HALF_Y - cy) / uy]

    t_candidates = [t for t in t_candidates if t > 0]
    t_hit = min(t_candidates) if t_candidates else float('inf')

    # minimum butt height needed to clear wall
    if not math.isfinite(t_hit) or t_hit >= cue_length:
        min_z = 0.0
    else:
        need = WALL_H + BALL_R
        min_z = need * (cue_length / max(t_hit, EPS))

    # global max tilt = 10° above horizontal
    max_z = math.tan(math.radians(10.0)) * cue_length
    return min_z, max_z




def draw_cue():
    if not balls:
        return
    try:
        # current cue ball position and angle
        cx, cy, cz = balls[-1]["pos"]
        ang = cue_angle

        # dynamic limits (butt must clear the wall behind the ball along -angle)
        min_z, max_z = cue_limits(cx, cy, ang)

        # clamp global butt height
        global cue_butt_z
        if cue_butt_z < min_z: cue_butt_z = min_z
        if cue_butt_z > max_z: cue_butt_z = max_z

        # --- your existing draw code follows ---
        length = cue_length
        offset = BALL_R + cue_gap

        r_front = cue_radius * 0.7
        r_back  = cue_radius * 1.3
        L1 = length * 0.75
        L2 = length - L1
        r_mid = r_front + (r_back - r_front) * 0.75

        dx = math.cos(math.radians(ang))
        dy = math.sin(math.radians(ang))

        if charging:
            pull_back = (cue_power / 100.0) * PULL_MAX
            current_offset = offset + pull_back
        elif cue_stroked:
            if impact_base is not None:
                base_x, base_y, base_z = impact_base
            else:
                current_offset = max(1.0, offset - stroke_offset)
                base_x = cx - dx * current_offset
                base_y = cy - dy * current_offset
                base_z = cz
        else:
            current_offset = offset

        if not cue_stroked:
            base_x = cx - dx * current_offset
            base_y = cy - dy * current_offset
            base_z = cz

        # Orientation vector (butt offset is vertical component)
        raw_x = -dx * length
        raw_y = -dy * length
        raw_z = cue_butt_z
        mag = math.sqrt(raw_x*raw_x + raw_y*raw_y + raw_z*raw_z) or 1e-6
        v_x, v_y, v_z = raw_x/mag, raw_y/mag, raw_z/mag
        axis_x, axis_y, axis_z = -v_y, v_x, 0.0
        angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, v_z))))

        glPushMatrix()
        glTranslatef(base_x, base_y, base_z)
        if abs(axis_x) > 1e-6 or abs(axis_y) > 1e-6:
            glRotatef(angle_deg, axis_x, axis_y, axis_z)

        quad = gluNewQuadric()
        glColor3f(0.8, 0.6, 0.3); gluCylinder(quad, r_front, r_mid, L1, 12, 1)
        glTranslatef(0.0, 0.0, L1)
        glColor3f(0.35, 0.18, 0.05); gluCylinder(quad, r_mid, r_back, L2, 12, 1)
        glPopMatrix()
    except Exception:
        return


def resolve_ball_collisions():
    """Equal-mass 2D collisions with positional correction and restitution."""
    R  = BALL_R
    D  = 2.0 * R
    D2 = D * D

    n = len(balls)
    for _ in range(POS_ITERATIONS):
        for i in range(n):
            bi = balls[i]
            if not bi["alive"]: 
                continue
            pi, vi = bi["pos"], bi["vel"]

            for j in range(i + 1, n):
                bj = balls[j]
                if not bj["alive"]:
                    continue
                pj, vj = bj["pos"], bj["vel"]

                dx = pj[0] - pi[0]
                dy = pj[1] - pi[1]
                dist2 = dx*dx + dy*dy
                if dist2 >= D2:
                    continue

                dist = math.sqrt(dist2) if dist2 > EPS else 0.0
                if dist > EPS:
                    nx = dx / dist
                    ny = dy / dist
                    overlap = D - dist
                else:
                    nx, ny = 1.0, 0.0
                    overlap = D

                # Positional correction: split overlap
                half = 0.5 * overlap
                pi[0] -= nx * half; pi[1] -= ny * half
                pj[0] += nx * half; pj[1] += ny * half

                # Relative velocity along normal
                rvx = vi[0] - vj[0]
                rvy = vi[1] - vj[1]
                rel_n = rvx * nx + rvy * ny
                if rel_n > 0.0:
                    continue  # separating

                e = COLLISION_RESTITUTION
                jimp = -(1.0 + e) * rel_n / 2.0  # equal mass

                imp_x = jimp * nx
                imp_y = jimp * ny

                vi[0] -= imp_x; vi[1] -= imp_y
                vj[0] += imp_x; vj[1] += imp_y

def resolve_cushion_collisions():
    min_x = -HALF_X + BALL_R
    max_x =  HALF_X - BALL_R
    min_y = -HALF_Y + BALL_R
    max_y =  HALF_Y - BALL_R

    for b in balls:
        if not b["alive"]:
            continue
        x, y, _ = b["pos"]
        vx, vy  = b["vel"]

        # X walls
        if x < min_x:
            b["pos"][0] = min_x
            b["vel"][0] = -vx * CUSHION_RESTITUTION
        elif x > max_x:
            b["pos"][0] = max_x
            b["vel"][0] = -vx * CUSHION_RESTITUTION

        # Y walls
        if y < min_y:
            b["pos"][1] = min_y
            b["vel"][1] = -vy * CUSHION_RESTITUTION
        elif y > max_y:
            b["pos"][1] = max_y
            b["vel"][1] = -vy * CUSHION_RESTITUTION


def all_balls_sleeping():
    for b in balls:
        if not b["alive"]:
            continue
        if abs(b["vel"][0]) >= SLEEP_EPS or abs(b["vel"][1]) >= SLEEP_EPS:
            return False
    return True


def resolve_ball_overlaps():
    """Split penetration along the normal with slop & percent (no velocity changes)."""
    R  = BALL_R
    D  = 2.0 * R
    D2 = D * D

    n = len(balls)
    for _ in range(POS_ITERATIONS):
        for i in range(n):
            bi = balls[i]
            if not bi["alive"]:
                continue
            pi = bi["pos"]

            for j in range(i + 1, n):
                bj = balls[j]
                if not bj["alive"]:
                    continue
                pj = bj["pos"]

                dx = pj[0] - pi[0]
                dy = pj[1] - pi[1]
                dist2 = dx*dx + dy*dy
                if dist2 >= D2:
                    continue

                dist = math.sqrt(dist2) if dist2 > EPS else 0.0
                if dist > EPS:
                    nx = dx / dist
                    ny = dy / dist
                    overlap = D - dist
                else:
                    nx, ny = 1.0, 0.0
                    overlap = D

                # Apply slop & percent to avoid over-correction
                corr = max(overlap - PENETRATION_SLOP, 0.0) * PENETRATION_PERCENT
                if corr <= 0.0:
                    continue

                half = 0.5 * corr
                pi[0] -= nx * half; pi[1] -= ny * half
                pj[0] += nx * half; pj[1] += ny * half


def apply_ball_impulses_once():
    """One bounce impulse per touching pair, only if approaching and not resting."""
    R  = BALL_R
    D  = 2.0 * R
    # accept near-touch contacts (helps discrete timesteps)
    thresh = (D + CONTACT_TOLERANCE) * (D + CONTACT_TOLERANCE)

    n = len(balls)
    for i in range(n):
        bi = balls[i]
        if not bi["alive"]:
            continue
        pi, vi = bi["pos"], bi["vel"]

        for j in range(i + 1, n):
            bj = balls[j]
            if not bj["alive"]:
                continue
            pj, vj = bj["pos"], bj["vel"]

            dx = pj[0] - pi[0]
            dy = pj[1] - pi[1]
            dist2 = dx*dx + dy*dy
            if dist2 > thresh:
                continue

            dist = math.sqrt(dist2) if dist2 > EPS else D
            nx = dx / dist if dist > EPS else 1.0
            ny = dy / dist if dist > EPS else 0.0

            # relative velocity of i vs j along the normal (i -> j)
            rvx = vi[0] - vj[0]
            rvy = vi[1] - vj[1]
            rel_n = rvx * nx + rvy * ny

            # Only if clearly approaching, not resting
            if rel_n <= MIN_REL_SPEED_FOR_HIT:
                continue

            # Equal mass impulse with restitution
            e = BALL_RESTITUTION
            jimp = ((1.0 + e) * rel_n) / 2.0

            # Cap rare spikes
            if MAX_IMPULSE is not None and jimp > MAX_IMPULSE:
                jimp = MAX_IMPULSE

            imp_x = jimp * nx
            imp_y = jimp * ny

            vi[0] -= imp_x; vi[1] -= imp_y
            vj[0] += imp_x; vj[1] += imp_y


def respawn_cue_ball():
    """Place cue back on table safely and stop it."""
    # start near your usual cue spot and scan right until clear
    base_x, base_y = -150.0, 0.0
    step = 2 * BALL_R + 4.0
    max_steps = int((2 * HALF_X) / step)

    cue = balls[-1]  # by construction, last is cue (id=0)
    for s in range(max_steps):
        x = base_x + s * step
        y = base_y
        # keep within felt
        if x < (-HALF_X + BALL_R) or x > (HALF_X - BALL_R):
            continue
        # avoid overlaps with any alive object ball
        free = True
        for b in balls:
            if not b["alive"] or b["id"] == 0:
                continue
            dx = x - b["pos"][0]
            dy = y - b["pos"][1]
            if dx*dx + dy*dy < (2*BALL_R)*(2*BALL_R):
                free = False
                break
        if free:
            cue["pos"][0], cue["pos"][1], cue["pos"][2] = x, y, FELT_Z + BALL_R
            cue["vel"][0] = cue["vel"][1] = 0.0
            cue_velocity[0] = cue_velocity[1] = 0.0
            return
    # fallback (rare)
    cue["pos"][0], cue["pos"][1], cue["pos"][2] = -HALF_X * 0.5, 0.0, FELT_Z + BALL_R
    cue["vel"][0] = cue["vel"][1] = 0.0
    cue_velocity[0] = cue_velocity[1] = 0.0


def handle_cue_scratch():
    """Lose a life; if lives run out -> game over; otherwise respawn cue."""
    global lives, game_state
    lives -= 1
    if lives <= 0:
        game_state = GAME_OVER
        # hard stop cue ball too
        cue = balls[-1]
        cue["vel"][0] = cue["vel"][1] = 0.0
        cue_velocity[0] = cue_velocity[1] = 0.0
        return
    respawn_cue_ball()


def handle_object_potted(b):
    """Remove object ball; 8-ball logic determines win/lose."""
    global game_state
    if b["id"] == 8:
        # 8-ball early = lose; last = win
        # (remaining_object_balls counts all alive non-cue including 8; so check before killing 8)
        others_left = 0
        for bb in balls:
            if bb["alive"] and bb["id"] not in (0, 8):
                others_left += 1
        if others_left > 0:
            game_state = GAME_OVER   # potted 8 early
        else:
            game_state = GAME_WON    # cleared all, then 8 → win
        b["alive"] = False
    else:
        b["alive"] = False


def check_potting():
    """Detect balls that have fallen into any pocket and handle them."""
    if not balls:
        return
    centers = pocket_centers()
    thr2 = (POT_THRESHOLD) * (POT_THRESHOLD)

    for b in balls:
        if not b["alive"]:
            continue
        bx, by, _ = b["pos"]

        # inside any pocket?
        potted = False
        for (px, py) in centers:
            dx = bx - px
            dy = by - py
            if dx*dx + dy*dy <= thr2:
                potted = True
                break

        if not potted:
            continue

        if b["id"] == 0:
            handle_cue_scratch()
        else:
            handle_object_potted(b)

def reset_game():
    """Full reset: lives, state, balls, cue state."""
    global lives, game_state, cue_power, charging, cue_velocity
    global cue_stroked, stroke_offset, impact_base, cue_angle, cue_butt_z

    lives = LIVES_MAX
    game_state = GAME_RUNNING

    # reset cue/control state
    cue_power = 0.0
    charging = False
    cue_velocity[0] = cue_velocity[1] = 0.0
    cue_stroked = False
    stroke_offset = 0.0
    impact_base = None
    cue_angle = 0.0
    cue_butt_z = 0.0

    # re-seed balls
    seed_balls()


def update_physics():
    """Integrate → overlap separation → single impulse pass → cushions → friction → clamp → cue reset."""
    if game_state != GAME_RUNNING:
        return
    
    if not balls:
        return

    # Keep cue dict vel in sync with your separate array (legacy)
    cue_vel = balls[-1]["vel"]
    cue_vel[0], cue_vel[1] = cue_velocity[0], cue_velocity[1]

    # 1) Integrate all alive balls
    for b in balls:
        if not b["alive"]:
            continue
        b["pos"][0] += b["vel"][0]
        b["pos"][1] += b["vel"][1]

    # 2) Separate overlaps (positions only, multiple passes)
    resolve_ball_overlaps()

    # 3) Single impulse sweep (apply bounce once per pair)
    apply_ball_impulses_once()

    # 4) Cushions (reflect on rails)
    resolve_cushion_collisions()

    # 4.5) Pockets (pot/remove balls, scratch respawn)
    check_potting()

    # 5) Friction & speed clamps
    for b in balls:
        if not b["alive"]:
            continue
        # friction
        b["vel"][0] *= FRICTION_COEFF
        b["vel"][1] *= FRICTION_COEFF
        # tiny clamp (sleep)
        if abs(b["vel"][0]) < SLEEP_EPS: b["vel"][0] = 0.0
        if abs(b["vel"][1]) < SLEEP_EPS: b["vel"][1] = 0.0
        # hard cap (prevents runaway spikes)
        sp2 = b["vel"][0]*b["vel"][0] + b["vel"][1]*b["vel"][1]
        if sp2 > MAX_SPEED*MAX_SPEED:
            s = math.sqrt(sp2)
            if s > EPS:
                b["vel"][0] = b["vel"][0] * (MAX_SPEED / s)
                b["vel"][1] = b["vel"][1] * (MAX_SPEED / s)

    # Write cue dict velocity back to your array
    cue_velocity[0], cue_velocity[1] = cue_vel[0], cue_vel[1]

    # 6) If cue finished & balls stopped, reset stick pose
    if all_balls_sleeping():
        global cue_stroked, stroke_offset, impact_base
        if cue_stroked:
            cue_stroked = False
            stroke_offset = 0.0
            impact_base = None

def keyboardListener(key, x, y):
    global cue_angle, cue_power, charging, cue_velocity, cue_butt_z, game_state
    if key == b'r':
        reset_game(); return
    if game_state != GAME_RUNNING:
        return

    if key == b'a': cue_angle += 5.0
    if key == b'd': cue_angle -= 5.0

    if key in (b'w', b's'):
        cx, cy, _ = balls[-1]["pos"]
        ang = cue_angle
        min_z, max_z = cue_limits(cx, cy, ang)
        if key == b'w':
            cue_butt_z = min(max_z, cue_butt_z + 4.0)
        else:
            cue_butt_z = max(min_z, cue_butt_z - 4.0)


    if key == b' ' and not cue_stroked:
        charging = True


def keyboardUpListener(key, x, y):
    """Release spacebar to shoot."""
    global charging, cue_power, cue_velocity
    if key == b' ' and charging:
        # shoot cue ball
        dx = math.cos(math.radians(cue_angle))
        dy = math.sin(math.radians(cue_angle))
        # apply a gentler multiplier so the ball moves slower
        cue_velocity[0] = dx * cue_power * 0.15
        cue_velocity[1] = dy * cue_power * 0.15
        # set stroke offset proportional to power so cue appears forward
        global cue_stroked, stroke_offset
        stroke_offset = (cue_power / 100.0) * HIT_MAX
        cue_stroked = True
        # store the world-space base position where the cue impacts so
        # the cue remains fixed there while the ball moves
        global impact_base
        cue_ball = balls[-1]["pos"]
        cx, cy, cz = cue_ball

        impact_base = (cx - dx * (BALL_R + 5), cy - dy * (BALL_R + 5), cz)
        #print(f"released: power={cue_power}, velocity=({cue_velocity[0]:.3f},{cue_velocity[1]:.3f}), impact_base={impact_base}")
        cue_power = 0.0
        charging = False

def idle():
    """Update game loop: physics and power charging."""
    global cue_power

    if game_state != GAME_RUNNING:
        glutPostRedisplay()
        return
    
    if charging:
        cue_power = min(100.0, cue_power + 1.0)  # build power
    update_physics()
    glutPostRedisplay()
        

def special_keys(key, x, y):
    global cam_yaw_deg, cam_height
    if key == GLUT_KEY_LEFT:  cam_yaw_deg -= 3.0
    if key == GLUT_KEY_RIGHT: cam_yaw_deg += 3.0
    if key == GLUT_KEY_UP:    cam_height += 10.0
    if key == GLUT_KEY_DOWN:  cam_height = max(60.0, cam_height - 10.0)
    glutPostRedisplay()

def init():
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.0, 0.0, 0.0, 1.0)

def main():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)                
    glutInitWindowSize(WIN_W, WIN_H)
    glutInitWindowPosition(100, 50)
    glutCreateWindow(b"3D Pool")

    glutIgnoreKeyRepeat(1) 
    init()
    seed_balls()

    glutDisplayFunc(display)
    glutIdleFunc(idle)
    glutSpecialFunc(special_keys)
    glutKeyboardFunc(keyboardListener)
    glutKeyboardUpFunc(keyboardUpListener)

    glutMainLoop()

if __name__ == "__main__":
    main()

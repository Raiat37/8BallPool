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
N_BALLS = 9
balls_pos, balls_col = [], []

# ====== Cue state & shooting (add below your other globals) ======
cue_angle = 0.0        # yaw around Z (left/right), degrees
cue_height = 0.0       # vertical offset of cue (up/down along Z), relative to ball center
cue_length = 255.0     # stick length behind the ball (was 150 -> 150*1.7 => 255)
cue_radius = 3.0       # stick thickness
cue_gap = 1.0          # tiny gap from ball surface to avoid z-fighting

cue_power = 0.0        # power accumulator while holding space
charging = False
cue_velocity = [0.0, 0.0]  # cue ball velocity in XY after strike


def seed_balls():
    random.seed(42)
    colors = [
        (1.0, 0.2, 0.2), (1.0, 0.6, 0.2), (1.0, 1.0, 0.2), (0.2, 1.0, 0.2),
        (0.2, 1.0, 1.0), (0.2, 0.4, 1.0), (0.8, 0.2, 1.0), (1.0, 0.2, 0.6)
    ]
    start_x, start_y = 120, 60
    dx = 2 * BALL_R + 2
    idx = 0
    for r in range(1, 5):
        count = r if r < 4 else 2          # total 8
        y0 = start_y - (r-1) * (BALL_R*2+2)
        x0 = start_x + (r-1) * dx * 0.5
        for c in range(count):
            if idx < 8:
                px = x0 + c*dx
                py = y0
                balls_pos.append([px, py, FELT_Z + BALL_R])
                balls_col.append(colors[idx])
                idx += 1
    # cue ball (white)
    balls_pos.append([-150.0, 0.0, FELT_Z + BALL_R])
    balls_col.append((1.0, 1.0, 1.0))

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
    for i in range(N_BALLS):
        glPushMatrix()
        glColor3f(*balls_col[i])
        glTranslatef(balls_pos[i][0], balls_pos[i][1], balls_pos[i][2])
        gluSphere(gluNewQuadric(), BALL_R, 20, 16)
        glPopMatrix()

def display():
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
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

    # restore matrices
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()

# ================= Cue Stick Mechanics =================

cue_angle = 0.0        # angle around cue ball
cue_power = 0.0        # current power
charging = False       # are we charging power?
cue_velocity = [0.0, 0.0]  # velocity of cue ball after hit
cue_butt_z = 0.0       # vertical offset of the butt (end away from ball)
PULL_MAX = 80.0       # maximum pull-back while charging (world units)
HIT_MAX = 40.0        # maximum forward push on hit (how far cue moves forward when striking)
cue_stroked = False   # set after releasing space until ball stops
stroke_offset = 0.0   # how far forward the cue is after striking (positive)
impact_base = None    # fixed world (x,y,z) position of cue base at impact

def draw_cue():
    """Draw the cue stick as a two-segment tapered cylinder behind the cue ball."""
    if not balls_pos:
        return
    try:
        cue_ball = balls_pos[-1]
        cx, cy, cz = cue_ball

        # stick length and default offset behind cue ball
        length = cue_length
        offset = BALL_R + 5

        # front/back radii: front (touching cue) smaller, back slightly bigger
        r_front = cue_radius * 0.7
        r_back = cue_radius * 1.3
        # segment lengths: front 3/4 (current colour), back 1/4 (brown)
        L1 = length * 0.75
        L2 = length - L1
        # radius at segment boundary (interpolated)
        r_mid = r_front + (r_back - r_front) * 0.75

        # direction from angle
        dx = math.cos(math.radians(cue_angle))
        dy = math.sin(math.radians(cue_angle))

        # compute dynamic offset
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

        # Build local orientation
        raw_x = -dx * length
        raw_y = -dy * length
        raw_z = cue_butt_z
        mag = math.sqrt(raw_x*raw_x + raw_y*raw_y + raw_z*raw_z)
        if mag == 0.0:
            mag = 1e-6
        v_x = raw_x / mag
        v_y = raw_y / mag
        v_z = raw_z / mag
        axis_x = -v_y
        axis_y = v_x
        axis_z = 0.0
        v_z_clamped = max(-1.0, min(1.0, v_z))
        angle_deg = math.degrees(math.acos(v_z_clamped))

        glPushMatrix()
        glTranslatef(base_x, base_y, base_z)
        if abs(axis_x) > 1e-6 or abs(axis_y) > 1e-6:
            glRotatef(angle_deg, axis_x, axis_y, axis_z)

        quad = gluNewQuadric()
        # front segment (3/4) - wood color
        glColor3f(0.8, 0.6, 0.3)
        gluCylinder(quad, r_front, r_mid, L1, 12, 1)
        # move to back segment start
        glTranslatef(0.0, 0.0, L1)
        # back segment (1/4) - brown
        glColor3f(0.35, 0.18, 0.05)
        gluCylinder(quad, r_mid, r_back, L2, 12, 1)

        glPopMatrix()
    except Exception:
        return

def update_physics():
    """Move cue ball if it has velocity."""
    cue_ball = balls_pos[-1]
    was_moving = (abs(cue_velocity[0]) > 1e-6 or abs(cue_velocity[1]) > 1e-6)
    cue_ball[0] += cue_velocity[0]
    cue_ball[1] += cue_velocity[1]

    # simple friction
    cue_velocity[0] *= 0.98
    cue_velocity[1] *= 0.98
    if abs(cue_velocity[0]) < 0.01 and abs(cue_velocity[1]) < 0.01:
        cue_velocity[0] = cue_velocity[1] = 0.0
        # when ball stops, return cue to behind the ball
        global cue_stroked, stroke_offset, impact_base
        if cue_stroked:
            cue_stroked = False
            stroke_offset = 0.0
            impact_base = None
            # print("ball stopped; cue reset")
    else:
        if not was_moving:
            print(f"ball started moving vel=({cue_velocity[0]:.3f},{cue_velocity[1]:.3f})")

def keyboardListener(key, x, y):
    """Handle rotation and shooting power."""
    global cue_angle, cue_power, charging, cue_velocity, cue_butt_z

    if key == b'a':   # rotate left
        cue_angle += 5.0
    if key == b'd':   # rotate right
        cue_angle -= 5.0
    if key == b'w':   # raise butt end
        # compute allowed range: not below top of wall, and not above 30deg from lowest
        min_z = WALL_H - BALL_R
        # lowest angle
        min_angle = math.atan2(min_z, cue_length)
        max_angle = min_angle + math.radians(30.0)
        max_z = math.tan(max_angle) * cue_length
        cue_butt_z += 4.0
        cue_butt_z = min(cue_butt_z, max_z)
    if key == b's':   # lower butt end
        min_z = WALL_H - BALL_R
        cue_butt_z -= 4.0
        cue_butt_z = max(cue_butt_z, min_z)
    if key == b' ':   # spacebar to charge
        # only start charging if not already stroking (ball is moving)
        if not cue_stroked:
            charging = True

def keyboardUpListener(key, x, y):
    """Release spacebar to shoot."""
    global charging, cue_power, cue_velocity
    if key == b' ' and charging:
        # shoot cue ball
        dx = math.cos(math.radians(cue_angle))
        dy = math.sin(math.radians(cue_angle))
        # apply a gentler multiplier so the ball moves slower
        cue_velocity[0] = dx * cue_power * 0.06
        cue_velocity[1] = dy * cue_power * 0.06
        # set stroke offset proportional to power so cue appears forward
        global cue_stroked, stroke_offset
        stroke_offset = (cue_power / 100.0) * HIT_MAX
        cue_stroked = True
        # store the world-space base position where the cue impacts so
        # the cue remains fixed there while the ball moves
        global impact_base
        cue_ball = balls_pos[-1]
        cx, cy, cz = cue_ball
        impact_base = (cx - dx * (BALL_R + 5), cy - dy * (BALL_R + 5), cz)
        #print(f"released: power={cue_power}, velocity=({cue_velocity[0]:.3f},{cue_velocity[1]:.3f}), impact_base={impact_base}")
        cue_power = 0.0
        charging = False

def idle():
    """Update game loop: physics and power charging."""
    global cue_power
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
    glClearColor(0.0, 0.0, 0.0, 1.0)

def main():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)                
    glutInitWindowSize(WIN_W, WIN_H)
    glutInitWindowPosition(100, 50)
    glutCreateWindow(b"3D Pool (no depth test, solid black pockets)")

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

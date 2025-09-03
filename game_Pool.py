from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import math, random

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

    glutSwapBuffers()

def idle():
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

    glutMainLoop()

if __name__ == "__main__":
    main()

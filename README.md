# 8BallPool — Rules & Controls

A lightweight 3D pool demo built with PyOpenGL + GLUT. Single table, basic collisions and friction, 8‑ball rules, and simple keyboard controls (hold **Space** to charge, release to shoot).

## Rules
- **1–7 first** — pot balls 1–7 in any order.
- **8‑ball last** — sink it to **win**.
- **Early 8‑ball** → **Game Over**.
- **Cue scratch** (cue ball pocketed) → **lose 1 life**.
- **Lives**: 3 total; lose all → **Game Over**.
- **Restart**: press **R**.


> Note: Cue tilt is capped at **10°** and auto‑clamps so the butt clears the cushion behind the cue ball.


## Controls


### Cue & Shot
| Action | Key |
|---|---|
| Aim left | **A** |
| Aim right | **D** |
| Charge power (hold) | **Space** *(hold)* |
| Shoot | **Space** *(release)* |
| Raise cue | **W** |
| Lower cue | **S** |


### Camera
| Action | Key |
|---|---|
| Orbit left | **←** *(Arrow Left)* |
| Orbit right | **→** *(Arrow Right)* |
| Raise camera | **↑** *(Arrow Up)* |
| Lower camera | **↓** *(Arrow Down)* |


### System
| Action | Key |
|---|---|
| Restart | **R** |
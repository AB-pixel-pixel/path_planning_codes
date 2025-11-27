# File: manipulation_sorting_2dof_demo_save.py

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation, PillowWriter

# ----------------------------
# Data structures
# ----------------------------
@dataclass
class Obj:
    color: str
    pos: np.ndarray
    picked: bool = False
    done: bool = False

@dataclass
class Bin:
    color: str
    rect: Tuple[float, float, float, float]

    @property
    def center(self) -> np.ndarray:
        x, y, w, h = self.rect
        return np.array([x + w / 2.0, y + h / 2.0])

# ----------------------------
# Arm model
# ----------------------------
@dataclass
class Arm2DOF:
    base: np.ndarray
    l1: float
    l2: float
    q1: float = 0.0
    q2: float = 0.0

    def fk(self, q1: float = None, q2: float = None):
        q1 = self.q1 if q1 is None else q1
        q2 = self.q2 if q2 is None else q2
        shoulder = self.base
        elbow = shoulder + self.l1 * np.array([math.cos(q1), math.sin(q1)])
        ee = elbow + self.l2 * np.array([math.cos(q1 + q2), math.sin(q1 + q2)])
        return shoulder, elbow, ee

    def ik(self, tgt: np.ndarray, elbow: str = "down"):
        dx, dy = tgt - self.base
        r = math.sqrt(dx*dx + dy*dy)
        r = min(max(r, abs(self.l1 - self.l2)+1e-6), self.l1 + self.l2 - 1e-6)
        cos2 = (r*r - self.l1**2 - self.l2**2) / (2*self.l1*self.l2)
        cos2 = min(1, max(-1, cos2))
        q2_mag = math.acos(cos2)
        q2 = -q2_mag if elbow == "down" else q2_mag
        k1 = self.l1 + self.l2*math.cos(q2)
        k2 = self.l2*math.sin(q2)
        q1 = math.atan2(dy, dx) - math.atan2(k2, k1)
        return q1, q2

    def set_joints(self, q1: float, q2: float):
        self.q1, self.q2 = q1, q2

# ----------------------------
# Path utilities
# ----------------------------
def angle_interp(a0, a1, steps):
    da = (a1 - a0 + math.pi) % (2*math.pi) - math.pi
    return [a0 + da * t for t in np.linspace(0, 1, steps)]

def joints_path(q0, q1, steps):
    q1_path = angle_interp(q0[0], q1[0], steps)
    q2_path = angle_interp(q0[1], q1[1], steps)
    return list(zip(q1_path, q2_path))

# ----------------------------
# Planner
# ----------------------------
class SortingPlanner2DOF:
    FLOW = ["Spawn Objects", "MoveToObject", "Pick", "MoveToBin", "Place", "Done"]

    def __init__(self, arm, bins, spawn_rect, num_sets=4, move_steps=40, hold_steps=12, elbow="down"):
        self.arm = arm
        self.bins = {b.color: b for b in bins}
        self.spawn_rect = spawn_rect
        self.num_sets = num_sets
        self.move_steps = move_steps
        self.hold_steps = hold_steps
        self.elbow = elbow

        self.objects: List[Obj] = []
        self.carrying = None
        self.frames = []

    def spawn_objects(self):
        x, y, w, h = self.spawn_rect
        colors = ["red", "green", "blue"]
        objs = []
        for _ in range(self.num_sets):
            for c in colors:
                px = random.uniform(x+0.06, x+w-0.06)
                py = random.uniform(y+0.06, y+h-0.06)
                objs.append(Obj(c, np.array([px, py])))
        return objs

    def build(self):
        self.objects = self.spawn_objects()

        for _ in range(18):
            self.frames.append(self._snapshot("Spawn Objects", None, None))

        color_order = {"red":0, "green":1, "blue":2}
        self.objects.sort(key=lambda o: color_order[o.color])

        for obj in self.objects:
            self._append_move_to(obj.pos, "MoveToObject", obj, None)
            self._append_hold_pick(obj)

            bin_center = self.bins[obj.color].center
            self._append_move_to(bin_center, "MoveToBin", obj, self.bins[obj.color])
            self._append_hold_place(obj, self.bins[obj.color])

        for _ in range(25):
            self.frames.append(self._snapshot("Done", None, None))

    def _append_move_to(self, target, state, obj, b):
        q0 = (self.arm.q1, self.arm.q2)
        qg = self.arm.ik(target)
        for q1, q2 in joints_path(q0, qg, self.move_steps):
            self.arm.set_joints(q1, q2)
            _, _, ee = self.arm.fk()
            if self.carrying:
                self.carrying.pos = ee.copy()
            self.frames.append(self._snapshot(state, obj, b))

    def _append_hold_pick(self, obj):
        self.carrying = obj
        obj.picked = True
        for _ in range(self.hold_steps):
            _, _, ee = self.arm.fk()
            obj.pos = ee.copy()
            self.frames.append(self._snapshot("Pick", obj, None))

    def _append_hold_place(self, obj, b):
        # random placement inside bin
        x, y, w, h = b.rect
        obj.pos = np.array([
            random.uniform(x + 0.05, x + w - 0.05),
            random.uniform(y + 0.05, y + h - 0.05)
        ])
        obj.picked = False
        obj.done = True
        self.carrying = None
        for _ in range(self.hold_steps):
            self.frames.append(self._snapshot("Place", obj, b))

    def _snapshot(self, state, active_obj, b):
        sh, el, ee = self.arm.fk()
        return {
            "state": state,
            "arm_q1": self.arm.q1,
            "arm_q2": self.arm.q2,
            "shoulder": sh.copy(),
            "elbow": el.copy(),
            "ee": ee.copy(),
            "objects": [(o.color, o.pos.copy(), o.picked, o.done) for o in self.objects],
        }

# ----------------------------
# Visualization
# ----------------------------
def run_animation_and_save(save_prefix="sorting_demo", fps=25):
    plt.rcParams.update({"font.size": 18})

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("2-DOF Planar Arm Sorting Demo")

    spawn_rect = (0.05, 0.05, 0.45, 0.35)
    bins = [
        Bin("red",   (0.05, 0.75, 0.25, 0.20)),
        Bin("green", (0.375, 0.75, 0.25, 0.20)),
        Bin("blue",  (0.70, 0.75, 0.25, 0.20)),
    ]

    ax.add_patch(patches.Rectangle((spawn_rect[0], spawn_rect[1]),
                                   spawn_rect[2], spawn_rect[3],
                                   fill=False, linestyle="--", linewidth=3))
    ax.text(spawn_rect[0], spawn_rect[1]+spawn_rect[3]+0.03, "Spawn Zone")

    for b in bins:
        x, y, w, h = b.rect
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, linewidth=3))
        ax.text(x+0.02, y+h+0.02, f"{b.color.title()} Bin")

    arm = Arm2DOF(np.array([0.5, 0.10]), 0.42, 0.36, math.pi/2, -math.pi/3)

    planner = SortingPlanner2DOF(arm, bins, spawn_rect, num_sets=4)
    planner.build()
    frames = planner.frames

    link1, = ax.plot([], [], linewidth=5)
    link2, = ax.plot([], [], linewidth=5)
    ee_dot, = ax.plot([], [], marker="o")
    elbow_dot, = ax.plot([], [], marker="o")

    obj_dots = {
        "red": ax.plot([], [], marker="o", markersize=12)[0],
        "green": ax.plot([], [], marker="o", markersize=12)[0],
        "blue": ax.plot([], [], marker="o", markersize=12)[0]
    }

    state_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    def init():
        link1.set_data([], [])
        link2.set_data([], [])
        for d in obj_dots.values():
            d.set_data([], [])
        state_text.set_text("")
        return [link1, link2, ee_dot, elbow_dot, *obj_dots.values(), state_text]

    def update(i):
        f = frames[i]
        sh, el, ee = f["shoulder"], f["elbow"], f["ee"]

        link1.set_data([sh[0], el[0]], [sh[1], el[1]])
        link2.set_data([el[0], ee[0]], [el[1], ee[1]])
        elbow_dot.set_data([el[0]], [el[1]])
        ee_dot.set_data([ee[0]], [ee[1]])

        color_groups = {"red": [], "green": [], "blue": []}
        for c, pos, _, _ in f["objects"]:
            color_groups[c].append(pos)

        for c, pts in color_groups.items():
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                obj_dots[c].set_data(xs, ys)
                obj_dots[c].set_color(c)
            else:
                obj_dots[c].set_data([], [])

        state_text.set_text(f"Step: {f['state']}")
        return [link1, link2, ee_dot, elbow_dot, *obj_dots.values(), state_text]

    anim = FuncAnimation(fig, update, frames=len(frames), init_func=init,
                         blit=True, interval=1000//fps, repeat=False)

    try:
        anim.save(f"{save_prefix}.mp4", writer="ffmpeg", fps=fps)
        print("Saved MP4")
    except:
        anim.save(f"{save_prefix}.gif", writer=PillowWriter(fps=fps))
        print("Saved GIF")

    plt.show()

if __name__ == "__main__":
    run_animation_and_save()

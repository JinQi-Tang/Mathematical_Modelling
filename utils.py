import numpy as np

class Obj3D:
    def __init__(self, pos: np.ndarray, 
                 vel: np.ndarray = None, 
                 acc: np.ndarray = None):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float) if vel is not None else np.zeros(3)
        self.acc = np.array(acc, dtype=float) if acc is not None else np.zeros(3)

    def after(self, t: float):
        """
        计算t秒后物体的位置、速度，加速度（加速度不变）。
        """
        # 新速度
        nvel = self.vel + self.acc * t

        # 新位置
        npos = self.pos + self.vel * t + 0.5 * self.acc * t * t

        self.pos, self.vel = npos, nvel
        
        if self.pos[2] < 0.0:
            self.pos[2] = 0.0

#烟雾有效半径，时长
r_smoke = 10.0
t_smoke = 20.0

#假目标，真目标
fake_target = np.array([0, 0, 0], dtype = float)
true_target = np.array([0, 200, 0], dtype = float)

v_M = 300.0
r_target = 7.0
h_target = 10.0

#导弹
M1 = np.array([20000, 0, 2000], dtype = float)
M2 = np.array([19000, 600, 2100], dtype = float)
M3  = np.array([18000, -600, 1900], dtype = float)

# FY1-FY5
FY1 = np.array([17800, 0, 1800], dtype = float)
FY2 = np.array([12000, 1400, 1400], dtype = float)
FY3 = np.array([6000, -3000, 700], dtype = float)
FY4 = np.array([11000, 2000, 1800], dtype = float)
FY5 = np.array([13000, -2000, 1300], dtype = float)






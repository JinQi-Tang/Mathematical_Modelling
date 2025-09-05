from utils import *
import numpy as np
import copy
import logging
import time
from scipy.optimize import differential_evolution

def init_logging():
    logging.basicConfig(
        level = logging.INFO,
        format = '%(message)s',
        handlers=[
            logging.FileHandler(
                "log/" + time.strftime('%Y-%m-%d-%H-%M-%S') + ".log", 
                encoding='utf-8'
            )
        ]
    )

def is_on_opposite_sides_of_polar_plane(p_M, p_smoke, r, p_test):
    
    A = np.dot(p_M - p_smoke, p_M - p_smoke) - r ** 2
    B = np.dot(p_test - p_smoke, p_M - p_smoke) - r ** 2
    
    return (A <= 0 and B >= 0) or (A >= 0 and B <= 0)

def is_point_in_cone(p_M, p_smoke, r, p_test):
    """
    判断点p_test是否在以p_M为顶点，与球(p_smoke为球心，r为半径)相切的圆锥内部
    
    参数:
    p_M: 圆锥顶点，numpy数组 [x, y, z]
    p_smoke: 球心，numpy数组 [x, y, z]
    r: 球半径，float
    p_test: 测试点，numpy数组 [x, y, z]
    
    返回:
    bool: 如果点在圆锥内部返回True，否则返回False
    """
    # 计算从顶点到球心的向量
    M_to_smoke = p_smoke - p_M
    # 计算从顶点到测试点的向量
    M_to_test = p_test - p_M
    
    # 计算圆锥的半角（夹角）
    # 球心到顶点的距离
    d = np.linalg.norm(M_to_smoke)
    
    if d <= r:
        return True
    
    # 计算半角的正弦值
    sin_theta = r / d
    
    # 计算两个向量的夹角余弦值
    cos_alpha = np.dot(M_to_smoke, M_to_test) / (np.linalg.norm(M_to_smoke) * np.linalg.norm(M_to_test))
    
    # 计算夹角的正弦值
    sin_alpha = np.sqrt(1 - cos_alpha**2)
    
    # 判断点是否在圆锥内部
    # 条件1: 测试点与圆锥轴的夹角小于等于圆锥半角
    # 条件2: 测试点在圆锥的同一侧（即与圆锥轴的方向相同）
    return sin_alpha <= sin_theta and cos_alpha > 0

def check_point_block(p_M, p_smoke, r, p_test):

    # 判断测试点是否被遮挡
    
    if np.linalg.norm(p_smoke - p_M) <= r or np.linalg.norm(p_smoke - p_test) <= r:
        return True
    
    return is_point_in_cone(p_M, p_smoke, r, p_test) and is_on_opposite_sides_of_polar_plane(p_M, p_smoke, r, p_test)

def generate_circle_points(center, radius, n_points):
    """
    生成在xoy平面上以center为中心、radius为半径的圆周等分点
    
    参数:
    center: 圆心坐标，numpy数组 [x, y, z]
    radius: 圆半径，float
    n_points: 等分点数，int
    
    返回:
    numpy数组，形状为 (n_points, 3)
    """
    # 生成角度数组
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    
    # 计算x, y坐标（在xoy平面上）
    x_coords = center[0] + radius * np.cos(angles)
    y_coords = center[1] + radius * np.sin(angles)
    z_coords = np.full(n_points, center[2])  # z坐标保持不变
    
    # 组合成三维坐标点
    points = np.column_stack((x_coords, y_coords, z_coords))
    
    return points

def check_target_block(p_M, p_smoke, r = r_smoke, n_test = 8):
    '''
    判断真目标是否被遮挡
    
    参数：
    p_M：导弹坐标
    p_smoke:烟雾中心坐标
    r:烟雾半径
    n_test:在目标上取出多少离散的点
    
    true_target = np.array([0, 200, 0], dtype = float)

    r_target = 7.0
    h_target = 10.0
    
    '''
    
    circle_points = generate_circle_points(true_target, r_target, n_test)
    for i, point in enumerate(circle_points):
        if not check_point_block(p_M, p_smoke, r, point):
            return False
        if not check_point_block(p_M, p_smoke, r, point + np.array([0.0, 0.0, h_target])):
            return False
        
    return True

# 弃用
def tangent_cone_xoy_conic(p_smoke: np.ndarray, p_M: np.ndarray):
    """
    给定球心 p_smoke、圆锥顶点 p_M、球半径 r_smoke，
    返回圆锥与 xoy 平面的相交二次曲线系数 (a,b,c,d,e,f)，
    使得 a x^2 + b x y + c y^2 + d x + e y + f = 0。
    要求 ||p_smoke - p_M|| > r_smoke。
    """
    m = np.asarray(p_M, dtype=float)
    c = np.asarray(p_smoke, dtype=float)
    r = float(r_smoke)

    v = c - m
    d2 = float(np.dot(v, v))
    if d2 <= r * r:
        raise ValueError("无实切锥：需要 ||p_smoke - p_M|| > r_smoke。")

    A = d2 - r * r

    # kx, ky, kz are constants from (X - m_x), (Y - m_y), (0 - m_z)
    kx, ky, kz = -m[0], -m[1], -m[2]

    vx, vy, vz = v
    s0 = vx * kx + vy * ky + vz * kz

    a = vx * vx - A
    b = 2.0 * vx * vy
    ccoef = vy * vy - A  # 'c' as a variable name shadows Python's built-in; keep 'ccoef'
    d = 2.0 * vx * s0 - 2.0 * A * kx
    e = 2.0 * vy * s0 - 2.0 * A * ky
    f = s0 * s0 - A * (kx * kx + ky * ky + m[2] * m[2])

    return a, b, ccoef, d, e, f

def cal_block_time(M, FY, vx, vy, t1, t2, t_gap):
    '''
    计算有效遮蔽时长
    
    参数：
    M：导弹坐标
    FY：无人机坐标
    FY：无人机
    vx, vy:无人机速度
    间隔t1秒后投放
    再间隔t2秒后烟雾弹爆炸
    t_gap:时间刻度
    '''
    
    obj_M = Obj3D(M, -M / np.linalg.norm(M) * v_M)
    obj_FY = Obj3D(FY, np.array([vx, vy, 0.0]))
    
    obj_M.after(t1)
    obj_FY.after(t1)
    logging.info(f"t1 = {t1} 后导弹坐标{obj_M.pos}")
    logging.info(f"t1 = {t1} 后无人机坐标{obj_M.pos}")
    
    obj_smoke = Obj3D(copy.deepcopy(obj_FY.pos), copy.deepcopy(obj_FY.vel), np.array([0.0, 0.0, -9.8]))
    
    obj_M.after(t2)
    obj_FY.after(t2)
    obj_smoke.after(t2)
    logging.info(f"t2 = {t2} 后导弹坐标{obj_M.pos}")
    logging.info(f"t2 = {t2} 后无人机坐标{obj_FY.pos}")
    logging.info(f"烟雾弹起爆点{obj_smoke.pos}")

    obj_smoke_explode = Obj3D(copy.deepcopy(obj_smoke.pos), np.array([0.0, 0.0, -3.0]))
    
    t_now = 0.0
    cnt = 0
    tot = 0
    
    res = []
    
    while t_now <= 20.0:
        
        logging.info(f"烟雾弹起爆后 {t_now} 秒导弹坐标{obj_M.pos}")
        logging.info(f"烟雾弹起爆后 {t_now} 秒云团中心坐标{obj_smoke_explode.pos}")
        
        tot += 1
        if check_target_block(obj_M.pos, obj_smoke_explode.pos, r = r_smoke, n_test = 16):
            cnt += 1
            res.append(1)
        else:
            res.append(0)
        
        t_now += t_gap
        obj_M.after(t_gap)
        obj_FY.after(t_gap)
        obj_smoke_explode.after(t_gap)
    
    for i in range(len(res)):
        logging.info(f"{i * t_gap:.1f}:{res[i]}")
    
    return 20.0 * cnt / tot
        
def cal_block_time_without_info(M, FY, vx, vy, t1, t2, t_gap):
    '''
    计算有效遮蔽时长
    
    参数：
    M：导弹坐标
    FY：无人机坐标
    FY：无人机
    vx, vy:无人机速度
    间隔t1秒后投放
    再间隔t2秒后烟雾弹爆炸
    t_gap:时间刻度
    '''
    
    obj_M = Obj3D(M, -M / np.linalg.norm(M) * v_M)
    obj_FY = Obj3D(FY, np.array([vx, vy, 0.0]))
    
    obj_M.after(t1)
    obj_FY.after(t1)
    #logging.info(f"t1 = {t1} 后导弹坐标{obj_M.pos}")
    #logging.info(f"t1 = {t1} 后无人机坐标{obj_M.pos}")
    
    obj_smoke = Obj3D(copy.deepcopy(obj_FY.pos), copy.deepcopy(obj_FY.vel), np.array([0.0, 0.0, -9.8]))
    
    obj_M.after(t2)
    obj_FY.after(t2)
    obj_smoke.after(t2)
    #logging.info(f"t2 = {t2} 后导弹坐标{obj_M.pos}")
    #logging.info(f"t2 = {t2} 后无人机坐标{obj_FY.pos}")
    #logging.info(f"烟雾弹起爆点{obj_smoke.pos}")

    obj_smoke_explode = Obj3D(copy.deepcopy(obj_smoke.pos), np.array([0.0, 0.0, -3.0]))
    
    t_now = 0.0
    cnt = 0
    tot = 0
    
    res = []
    
    while t_now <= 20.0:
        
        #logging.info(f"烟雾弹起爆后 {t_now} 秒导弹坐标{obj_M.pos}")
        #logging.info(f"烟雾弹起爆后 {t_now} 秒云团中心坐标{obj_smoke_explode.pos}")
        
        tot += 1
        if check_target_block(obj_M.pos, obj_smoke_explode.pos, r = r_smoke, n_test = 16):
            cnt += 1
            res.append(1)
        else:
            res.append(0)
        
        t_now += t_gap
        obj_M.after(t_gap)
        obj_FY.after(t_gap)
        obj_smoke_explode.after(t_gap)
    
    return 20.0 * cnt / tot


def objective(params):
    vx, vy, t1, t2 = params
    # 检查速度大小约束
    speed = np.sqrt(vx**2 + vy**2)
    if speed < 70 or speed > 140:
        # 返回一个很差的函数值（因为要最大化，所以返回0相当于遮蔽时间为0）
        return 0.0   # 注意：由于我们要最大化，所以违反约束时返回最小值0（但实际最小可能为0）
    
    # 计算遮蔽时间（注意t_gap固定为0.1）
    block_time = cal_block_time_without_info(M1, FY1, vx, vy, t1, t2, t_gap=0.1)
    # 由于差分进化最小化目标，因此返回负值
    return -block_time

def test_with_info(vx, vy, t1, t2):
    init_logging()
    t_block = cal_block_time(M1, FY1, vx, vy, t1, t2, 0.1)
    print(t_block)
    
def optimization():
    bounds = [
        (-150, 0),   # vx（范围设大一些，但实际通过惩罚函数约束速度大小）
        (0, 150),   # vy
        (0, 10),       # t1
        (0, 10)        # t2
    ]
    
        # 直接指定你的初始值
    vx_initial = -123.0    # 你的vx初始值
    vy_initial = 1.5     # 你的vy初始值
    t1_initial = 0.8      # 你的t1初始值
    t2_initial = 3.8      # 你的t2初始值

    # 创建初始种群（第一个个体使用你的猜测，其余随机）
    init_population = []
    init_population.append([vx_initial, vy_initial, t1_initial, t2_initial])
    init_population.append([-77.4292686716235, 2.39709584886927, 0.4236289642793434, 2.7352032297591267])
    # 添加随机个体填充种群
    for i in range(2, 15):  # popsize=15
        vx = -80
        vy = 5
        t1 = np.random.uniform(0, 5)
        t2 = np.random.uniform(0, 5)
        init_population.append([vx, vy, t1, t2])

    init_population = np.array(init_population)

    # 运行优化
    result = differential_evolution(
        objective, 
        bounds, 
        strategy='best1bin', 
        maxiter=1000, 
        popsize=15,
        tol=1e-6, 
        mutation=(0.5, 1), 
        recombination=0.7,
        init=init_population
    )

    optimal_params = result.x
    vx_opt, vy_opt, t1_opt, t2_opt = optimal_params
    optimal_block_time = -result.fun

    print("最优解:")
    print(f"vx = {vx_opt}, vy = {vy_opt}, 速度大小 = {np.sqrt(vx_opt**2+vy_opt**2)}")
    print(f"t1 = {t1_opt}, t2 = {t2_opt}")
    print(f"最大遮蔽时间 = {optimal_block_time} 秒")

if __name__ == '__main__':

    test_with_info(vx = -120, vy = 0, t1 = 1.5, t2 = 3.6) # A1 题目 1.4s 3.0~4.4
    test_with_info(vx = -77, vy = 2.4, t1 = 0.4, t2 = 2.7) # A2 题目 4.6s 0.5~5.1
    optimization()
    
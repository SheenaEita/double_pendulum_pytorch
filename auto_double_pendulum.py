#-------------------------------------------------
# 本程式由Yuan-tao Huang製作，並基於以下專案開發而成：
#   https://github.com/rtqichen/torchdiffeq/tree/master
#   該專案主要提供完全運行於pytorch的積分器，但缺乏了自動產生ODEs的方法，需要人為手動輸入PyTorch版本的ODEs方程 q_ddot = f(q, q_dot)
#
# 本作品主要針對常見物理系統，透過歐拉-拉格朗日方程導出ODEs，過程使用SymPy套件進行符號運算，推導出運動方程式
# 再透過自動替換產生ODEs，並交付PyTorch上的數值積分器進行積分。
# 
# 最主要的繁重計算負擔（積分）運行於GPU，且機械系統固定，因此可以針對大量不同初始條件做批量平行積分，大幅提高計算效率。
# 由於完全基於PyTorch開發，且具有上述優勢，本方法可以用於穩定性分析、觀察系統混沌行為、暴力搜索簡單ODEs或是與機器學習/類神經網路結合。
# 
# 以下提供一個自動產生雙擺ODEs的方法，並進行大量（2^20）平行積分計算。
# Yuan-tao Huang, 2025.01.10
#-------------------------------------------------

import sympy
from sympy import symbols, Function, diff, sin, cos
import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import time

#-------------------------------------------------
# 1. 在 Sympy 中自動推導「雙擺」方程
#-------------------------------------------------

start_time = time.time()

#--- (A) 定義符號 ---
t_sym = sympy.Symbol('t', real=True)  # 連續時間 t
theta1_sym = Function('theta1')(t_sym)  # θ1(t)
theta2_sym = Function('theta2')(t_sym)  # θ2(t)

# 系統參數 (質量 m1, m2; 擺長 l1, l2; 重力加速度 g)
m1, m2, l1, l2, g = symbols('m1 m2 l1 l2 g', positive=True, real=True)

#--- (B) 定義動能 T 與位能 V ---
# 雙擺座標:
#   x1 = l1 * sin(θ1),  y1 = -l1 * cos(θ1)
#   x2 = x1 + l2 * sin(θ2), y2 = y1 - l2 * cos(θ2)
x1 = l1*sin(theta1_sym)
y1 = -l1*cos(theta1_sym)
x2 = x1 + l2*sin(theta2_sym)
y2 = y1 - l2*cos(theta2_sym)

# 對時間做一階微分 (速度)
x1_dot = diff(x1, t_sym)
y1_dot = diff(y1, t_sym)
x2_dot = diff(x2, t_sym)
y2_dot = diff(y2, t_sym)

# 動能: T = 1/2 m1 (x1_dot^2 + y1_dot^2) + 1/2 m2 (x2_dot^2 + y2_dot^2)
T = 0.5*m1*(x1_dot**2 + y1_dot**2) + 0.5*m2*(x2_dot**2 + y2_dot**2)

# 位能: V = m1*g*y1 + m2*g*y2
#  (y1, y2 可能為負值, 基準線自行決定)
V = m1*g*y1 + m2*g*y2

# 拉格朗日量 L = T - V
Lag = T - V

#--- (C) Euler-Lagrange 方程 ---
#   d/dt(∂L/∂θ1_dot) - ∂L/∂θ1 = 0
#   d/dt(∂L/∂θ2_dot) - ∂L/∂θ2 = 0
theta1_dot_sym = diff(theta1_sym, t_sym)
theta2_dot_sym = diff(theta2_sym, t_sym)

EL_eq1 = diff(Lag.diff(theta1_dot_sym), t_sym) - Lag.diff(theta1_sym)
EL_eq2 = diff(Lag.diff(theta2_dot_sym), t_sym) - Lag.diff(theta2_sym)

#--- (D) 簡化符號，簡化完可以提高計算效率 ---
EL_eq1 = sympy.simplify(EL_eq1)
EL_eq2 = sympy.simplify(EL_eq2)

#--- (E) 設定 θ1''、θ2'' 符號，並用 solve() 解出 ---
theta1_ddot_sym = diff(theta1_sym, (t_sym, 2))  # θ1''(t)
theta2_ddot_sym = diff(theta2_sym, (t_sym, 2))  # θ2''(t)

# sympy.solve: 取得 {θ1'': expr1, θ2'': expr2}
solutions = sympy.solve([EL_eq1, EL_eq2], [theta1_ddot_sym, theta2_ddot_sym], dict=True)
theta1_ddot_expr = solutions[0][theta1_ddot_sym]  # α1(θ1, θ2, θ1_dot, θ2_dot)
theta2_ddot_expr = solutions[0][theta2_ddot_sym]  # α2(θ1, θ2, θ1_dot, θ2_dot)

# 計算符號推導耗時
end_time = time.time()
print(f"Symbolic derivation time: {end_time - start_time:.4f} s")

print("\n========= ODEs =========")
print("theta_1_ddot =", theta1_ddot_expr)
print("theta_2_ddot =", theta2_ddot_expr)

#-------------------------------------------------
# 2. 將 Sympy expression 轉為 PyTorch 版 ODE 函式
#    (字串替換 + exec 動態生成)
#-------------------------------------------------

#--- (A) 定義新符號，用於替換 θ1(t), θ1'(t), θ2(t), θ2'(t) ---
theta1_ = sympy.Symbol('theta1', real=True)
omega1_ = sympy.Symbol('omega1', real=True)
theta2_ = sympy.Symbol('theta2', real=True)
omega2_ = sympy.Symbol('omega2', real=True)

# 把 θ1(t_sym), θ2(t_sym) 替換成 θ1_, θ2_；同理 θ1_dot_sym, θ2_dot_sym -> ω1_, ω2_
alpha1_subs = (
    theta1_ddot_expr
    .subs({theta1_sym: theta1_, theta2_sym: theta2_,
           theta1_dot_sym: omega1_, theta2_dot_sym: omega2_})
)
alpha2_subs = (
    theta2_ddot_expr
    .subs({theta1_sym: theta1_, theta2_sym: theta2_,
           theta1_dot_sym: omega1_, theta2_dot_sym: omega2_})
)

#--- (B) 用 pycode 輸出，再把 'math.' / 'numpy.' 換成 'torch.' ---
alpha1_code = sympy.printing.pycode(alpha1_subs).replace("math.", "torch.").replace("numpy.", "torch.")
alpha2_code = sympy.printing.pycode(alpha2_subs).replace("math.", "torch.").replace("numpy.", "torch.")

#--- (C) 組裝成可執行的 PyTorch 函式字串 ---
code_str = f"""
import torch

def alpha_func_torch(theta1, omega1, theta2, omega2, m1, m2, l1, l2, g):
    alpha1 = {alpha1_code}
    alpha2 = {alpha2_code}
    return alpha1, alpha2
"""

#--- (D) exec 在當前 namespace，動態生成 alpha_func_torch ---
ns = {}
exec(code_str, ns)
alpha_func_torch = ns["alpha_func_torch"]


print("\n========= Generated alpha_func_torch code_str =========")
print(code_str)

#-------------------------------------------------
# 3. 定義 PyTorch ODE 函式 (double_pendulum_ode)
#    在 GPU 上運算
#-------------------------------------------------
def double_pendulum_ode_torch(t, Y):
    """
    參數:
      t: 時間 (float, 但在 torchdiffeq 中不一定用)
      Y: shape=(batch_size,4), 包含 [θ1, ω1, θ2, ω2]
    回傳:
      dY/dt: shape=(batch_size,4)
    """
    # 取出狀態
    theta1_batch = Y[:, 0]
    omega1_batch = Y[:, 1]
    theta2_batch = Y[:, 2]
    omega2_batch = Y[:, 3]

    # 這裡可設定實際系統參數
    m1_ = 1.0
    m2_ = 1.0
    l1_ = 1.0
    l2_ = 1.0
    g_  = 9.8

    # 呼叫 alpha_func_torch 在 GPU 上運算
    alpha1, alpha2 = alpha_func_torch(
        theta1_batch, omega1_batch,
        theta2_batch, omega2_batch,
        m1_, m2_, l1_, l2_, g_
    )

    # dθ1/dt=ω1, dω1/dt=alpha1, dθ2/dt=ω2, dω2/dt=alpha2
    return torch.stack([omega1_batch, alpha1, omega2_batch, alpha2], dim=1)

#-------------------------------------------------
# 4. 進行大批量 (1048576) 積分測試
#-------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 1048576

#--- (A) 隨機生成初始條件 [θ1, ω1, θ2, ω2] ---
#   θ1, θ2 ∈ [0, 2π), ω1, ω2 為 N(0,1) 分佈
theta1_init = 2*torch.pi*torch.rand(batch_size)
theta2_init = 2*torch.pi*torch.rand(batch_size)
omega1_init = torch.randn(batch_size)
omega2_init = torch.randn(batch_size)

# 組合成 (batch_size, 4)，傳給GPU
y0 = torch.stack([theta1_init, omega1_init, theta2_init, omega2_init], dim=1).to(device)

# 時間軸: 0~3 秒, 301 個輸出點
t = torch.linspace(0, 3, 301, device=device)

print("\n========= Start Simulation =========")

start_time = time.time()
#--- (B) 使用 torchdiffeq 的 odeint 做數值積分 ---
with torch.no_grad():
    solution = odeint(double_pendulum_ode_torch, y0, t, method='rk4') #rk4 or dopri5
end_time = time.time()

print(f"Total GPU memory for simulate {batch_size} double pendulum systems {torch.cuda.memory_reserved(0)}.")

# 顯示解的形狀與執行時間
print("Solution shape:", solution.shape)  # (time, batch size, 2x order of ODEs)
print(f"Total simulation time: {end_time - start_time:.4f} s")

#-------------------------------------------------
# 5. 繪圖: 只畫第 1 條 (index=0) 與第 100 條 (index=99) 雙擺末端的 (x2,y2)
#-------------------------------------------------

# 對應第 i 條: solution[:, i, :]
# 定義一個函式來計算末端 x2, y2
def get_xy2(theta1, theta2, l1_val, l2_val):
    """
    根據 θ1(t), θ2(t), l1, l2
    回傳末端 (x2(t), y2(t))
    """
    x1_ = l1_val * torch.sin(theta1)
    y1_ = -l1_val * torch.cos(theta1)
    x2_ = x1_ + l2_val * torch.sin(theta2)
    y2_ = y1_ - l2_val * torch.cos(theta2)
    return x2_, y2_

# 取出第 0 條(第一條)與第 99 條(第一百條)的 θ1, θ2 時序
theta1_0 = solution[:, 0, 0]  # shape=(time,)
theta2_0 = solution[:, 0, 2]
theta1_99 = solution[:, 99, 0]
theta2_99 = solution[:, 99, 2]

# 假設 l1=1, l2=1
l1_val = 1.0
l2_val = 1.0

# 計算末端 (x2,y2)
x2_0, y2_0 = get_xy2(theta1_0, theta2_0, l1_val, l2_val)
x2_99, y2_99 = get_xy2(theta1_99, theta2_99, l1_val, l2_val)

# 畫軌跡
plt.figure(figsize=(6,6))
plt.plot(x2_0.cpu().numpy(),  y2_0.cpu().numpy(),  label='Pendulum #0')
plt.plot(x2_99.cpu().numpy(), y2_99.cpu().numpy(), label='Pendulum #99', alpha=0.7)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Double Pendulum Trajectory (Pendulum #0 & #99)')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

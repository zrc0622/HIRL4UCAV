import matplotlib.pyplot as plt
import numpy as np
import os
from pylab import mpl
from matplotlib.colors import Normalize
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False

def plot_3d_trajectories(self_pos, oppo_pos, fire, lock, dir, file_name):
    ax = plt.figure().add_subplot(projection='3d')
    self_pos = np.array(self_pos)
    oppo_pos = np.array(oppo_pos)

    num_points = len(self_pos)

    self_colors = plt.cm.Blues(np.linspace(0.3, 1, num_points))
    oppo_colors = plt.cm.Reds(np.linspace(0.3, 1, num_points))
    min_z = min(np.min(self_pos[:, 1]), np.min(oppo_pos[:, 1]))

    # Plot projections on XOY plane
    ax.plot(self_pos[:, 0], self_pos[:, 2], zs=min_z - 1000, zdir='z', color=self_colors[num_points//2], linestyle='--', linewidth=2, alpha=1)
    ax.plot(oppo_pos[:, 0], oppo_pos[:, 2], zs=min_z - 1000, zdir='z', color=oppo_colors[num_points//2], linestyle='--', linewidth=2, alpha=1)

    for i in range(num_points - 1):
        ax.plot(oppo_pos[i:i+2, 0], oppo_pos[i:i+2, 2], oppo_pos[i:i+2, 1], color=oppo_colors[i], label='oppo', zorder=2, linewidth=2)
        if i % 100 == 0:
            # Draw dashed line for opponent position projection
            ax.plot([oppo_pos[i, 0], oppo_pos[i, 0]], 
                    [oppo_pos[i, 2], oppo_pos[i, 2]], 
                    [oppo_pos[i, 1], min_z - 1000], 
                    color=oppo_colors[i], linestyle='--', linewidth=1, alpha=0.5, zorder=1)

    for i in range(num_points - 1):
        ax.plot(self_pos[i:i+2, 0], self_pos[i:i+2, 2], self_pos[i:i+2, 1], color=self_colors[i], label='self', zorder=2, linewidth=2)
        if i % 100 == 0:
            # Draw dashed line for self position projection
            ax.plot([self_pos[i, 0], self_pos[i, 0]], 
                    [self_pos[i, 2], self_pos[i, 2]], 
                    [self_pos[i, 1], min_z - 1000], 
                    color=self_colors[i], linestyle='--', linewidth=1, alpha=0.5, zorder=1)

    for i in lock:
        ax.plot(self_pos[i:i+2,0], self_pos[i:i+2,2], self_pos[i:i+2,1], color='#FFDFBF', zorder=3, linewidth=0.8, alpha=1)

    # 先创建一个Normalize对象，指定颜色映射的范围
    self_norm = Normalize(vmin=0, vmax=num_points)
    oppo_norm = Normalize(vmin=0, vmax=num_points)

    # 使用Normalize对象来创建ScalarMappable对象
    self_sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=self_norm)
    oppo_sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=oppo_norm)

    # 创建两个轴来放置颜色条，分别设置位置和大小
    self_cbar_ax = ax.inset_axes([1.15, 0.08, 0.03, 0.8])  # 调整位置和大小
    oppo_cbar_ax = ax.inset_axes([1.18, 0.08, 0.03, 0.8])  # 调整位置和大小

    # 添加颜色条到指定的轴上
    self_cbar = plt.colorbar(self_sm, cax=self_cbar_ax, label='Step number (agent)')
    oppo_cbar = plt.colorbar(oppo_sm, cax=oppo_cbar_ax, label='Step number (opponent)')

    # 设置颜色条的刻度
    # self_cbar.set_ticks(np.arange(0, num_points, 500))
    # 设置颜色条的标签位置
    self_cbar.ax.yaxis.set_label_coords(-2, 0.5)
    self_cbar.set_ticks([])
    oppo_cbar.set_ticks(np.arange(0, num_points, 500))

    for i in fire:
        ax.scatter(self_pos[i, 0], self_pos[i, 2], self_pos[i, 1], color='red', marker='^', s=8, zorder=5 if i == fire[0] else "")
        # ax.plot(self_pos[i:i+2,0], self_pos[i:i+2,2], self_pos[i:i+2,1], color='red', zorder=3, linewidth=3)

    for i in fire:
        ax.scatter(self_pos[i, 0], self_pos[i, 2], self_pos[i, 1], color='red', marker='^', s=8, zorder=5 if i == fire[0] else "")

    dir = os.path.join(dir, file_name)

    plt.title('3D trajectories', fontsize=14)

    plt.savefig(dir, bbox_inches='tight', dpi=300)
    plt.close()

def plot_distance(distance, lock, missile, fire, dir, file_name):
    # 创建一个新图形
    plt.figure(figsize=(10, 7))

    # 绘制 `distance` 曲线
    plt.plot(distance, color='blue', linewidth=3, marker='None', zorder=2)  # 用'None'禁用点的绘制


    # 着色 `lock` 区间为淡黄色
    for i in range(len(lock)):
        plt.axvspan(lock[i], lock[i]+1, facecolor='#FFDFBF', alpha=0.6, zorder=1)

    # 着色 `missile` 区间为淡蓝色
    for j in range(len(missile)):
        plt.axvspan(missile[j], missile[j]+1, facecolor='#ACD8E6', alpha=0.6, zorder=1)

    # 标记 `fire` 对应的点为红色并增大
    for fire_step in fire:
        # plt.scatter(fire_step, distance[fire_step], color='red', s=45, zorder=3)
        plt.scatter(fire_step, distance[fire_step], color='red', s=60, marker='^', zorder=3)

    plt.xticks(fontsize=18)  # 设置x轴标签的字体大小
    plt.yticks(fontsize=18)  # 设置y轴标签的字体大小     
    plt.xlabel('Step number', fontsize=25)
    plt.ylabel('Distance from opponent', fontsize=25)
    # plt.legend()
    plt.title('Distance', fontsize=28)
    plt.grid(True)
    dir = os.path.join(dir, file_name)
    plt.savefig(dir, dpi=300)
    plt.close()


def plot_2d_trajectories(ally_pos, enemy_pos, save_path=None):
    # 提取x和z坐标
    ally_x, ally_y, ally_z = zip(*ally_pos) if ally_pos else ([], [])
    enemy_x, enemy_y, enemy_z = zip(*enemy_pos) if enemy_pos else ([], [])

    # 创建图形和轴
    plt.figure(figsize=(10, 10))

    # 绘制友军和敌军轨迹
    plt.plot(ally_x, ally_z, 'b-o', label='Ally Trajectory', markersize=1)  # 蓝色线表示友军
    plt.plot(enemy_x, enemy_z, 'r-o', label='Enemy Trajectory', markersize=1)  # 红色线表示敌军

    # 设置图例、标题和坐标轴标签
    plt.legend()
    plt.title('Aircraft Trajectories')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')

    # 显示网格
    plt.grid(True)

    # 如果提供了保存路径，则保存图像
    if save_path:
        folder_path = os.path.dirname(save_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        plt.savefig(save_path)
        print(f"图像已保存至 {save_path}")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from citycolor.convert import hsv_to_rgb
matplotlib.use('Agg')

def get_hsv_colors(theta):
    """
    theta: 0~2pi
    返回 RGB 列表
    """
    hsv = np.zeros((len(theta),3))
    hsv[:,0] = theta / (2*np.pi)   # H 0~1
    hsv[:,1] = 1.0                 # 饱和度 100%
    hsv[:,2] = 1.0                 # 明亮度 100%
    rgb = mcolors.hsv_to_rgb(hsv)
    return rgb

def plot_hue(hist, save_path):
    hist = np.array(hist, dtype=float)
    N = len(hist)

    # ===== 角度 =====
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # 闭合
    theta = np.concatenate([theta, [theta[0]]])
    hist = np.concatenate([hist, [hist[0]]])

    # 归一化
    hist_norm = hist / (hist.max() + 1e-8)

    # ===== 创建极坐标 =====
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    # ===== 雷达图（多边形）=====
    ax.plot(theta, hist_norm, color='black', linewidth=2)
    ax.fill(theta, hist_norm, color='orange', alpha=0.3)

    # ===== 外圈 HSV 色环 =====
    r_outer = 1.1
    # 对应颜色
    colors = get_hsv_colors(theta[:-1])  # 不包含闭合的最后一个点
    ax.scatter(
        theta[:-1],
        np.ones_like(theta[:-1]) * r_outer,
        c=colors,
        s=60,  # 点大一些
        edgecolors='black',
        linewidths=0.5,
        zorder=10
    )

    # ===== 刻度线（保留）=====
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'])

    # 可选：角度刻度去掉（只保留网格线）
    ax.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False))
    ax.set_xticklabels([])

    ax.set_title("Hue Radar Histogram", pad=20)

    # 保存
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_saturation(hist, thelta, sats, save_path):
    plt.figure(figsize=(8, 6))
    colors = []
    for sat in sats:
        rgb = hsv_to_rgb(thelta, sat, 0.9)
        colors.append(tuple([v / 255 for v in rgb]))

    x=np.array([i for i in range(len(hist))])
    plt.bar(x=x, height=hist, color=colors, edgecolor='black', alpha=0.8)

    plt.xlabel("Saturation")
    plt.ylabel("Count")
    plt.title("Saturation Histogram")

    plt.grid(axis='y', alpha=0.3)

    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_hue_and_sat(hue_hist, sat_hist, save_path):
    hist = np.array(hue_hist, dtype=float)
    sat_hist = np.array(sat_hist, dtype=float)
    N = len(hist)

    # ===== 角度 =====
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # 闭合
    theta = np.concatenate([theta, [theta[0]]])
    hist = np.concatenate([hist, [hist[0]]])
    sats = np.concatenate([sat_hist, [sat_hist[0]]])

    # 归一化
    hist_norm = hist / (hist.max() + 1e-8)
    sats_norm = sats
    #sats_norm = sats / (sats.max() + 1e-8)

    # ===== 创建极坐标 =====
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    # ===== 雷达图（多边形）=====
    ax.plot(theta, hist_norm, color='black', linewidth=2)
    ax.fill(theta, hist_norm, color='orange', alpha=0.3)

    ax.plot(theta, sats_norm, color='black', linewidth=2)
    ax.fill(theta, sats_norm, color='green', alpha=0.3)

    # ===== 外圈 HSV 色环 =====
    r_outer = 1.1
    # 对应颜色
    colors = get_hsv_colors(theta[:-1])  # 不包含闭合的最后一个点
    ax.scatter(
        theta[:-1],
        np.ones_like(theta[:-1]) * r_outer,
        c=colors,
        s=60,  # 点大一些
        edgecolors='black',
        linewidths=0.5,
        zorder=10
    )

    # ===== 刻度线（保留）=====
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'])

    # 可选：角度刻度去掉（只保留网格线）
    ax.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False))
    ax.set_xticklabels([])

    ax.set_title("Hue Radar Histogram", pad=20)

    # 保存
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)




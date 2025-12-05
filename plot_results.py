import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_curve(log_file):
    # 1. 读取数据
    if not os.path.exists(log_file):
        print(f"错误: 找不到文件 {log_file}，请检查路径。")
        return

    try:
        data = pd.read_csv(log_file)
    except Exception as e:
        print(f"读取 CSV 出错: {e}")
        return

    # 2. 设置画图风格
    plt.figure(figsize=(12, 5))
    
    # --- 子图 1: 奖励曲线 (最重要) ---
    plt.subplot(1, 2, 1)
    # 绘制原始数据（透明度低，作为背景）
    plt.plot(data['Episode'], data['Reward'], alpha=0.3, color='gray', label='Raw Reward')
    
    # 绘制平滑曲线 (移动平均，窗口大小为 50)
    # 这样能过滤掉抖动，看清趋势
    data['Reward_Smooth'] = data['Reward'].rolling(window=50).mean()
    plt.plot(data['Episode'], data['Reward_Smooth'], color='blue', linewidth=2, label='Smoothed (MA-50)')
    
    plt.title('Training Reward Curve')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- 子图 2: Epsilon 衰减曲线 ---
    plt.subplot(1, 2, 2)
    plt.plot(data['Episode'], data['Epsilon'], color='orange', linewidth=2)
    plt.title('Epsilon Decay (Exploration Rate)')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.grid(True, alpha=0.3)

    # 3. 保存图片
    output_file = 'result_chart.png'
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"✅ 图表已生成并保存为: {output_file}")
    print("请在左侧文件列表中找到该图片，右键下载查看。")

if __name__ == "__main__":
    # 指向你的日志文件路径
    LOG_PATH = "./logs/q_learning_trust.csv" 
    plot_training_curve(LOG_PATH)
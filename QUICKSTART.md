# ⚡ 快速开始指南

## 🎯 目标

5分钟内完成环境配置并运行第一个实验！

---

## 步骤 1：检查环境 (30秒)

```bash
# 检查Python版本 (需要 >= 3.8)
python --version

# 检查当前目录
pwd  # 应该在 homework-li/ 目录下
ls   # 应该看到 main.py, algorithm/, envs/ 等
```

---

## 步骤 2：安装依赖 (2-5分钟)

```bash
# 安装所有Python依赖
pip install -r requirements.txt

# 验证安装
python -c "import torch; import pettingzoo; print('✅ All packages installed!')"
```

**如果安装失败：**
```bash
# 方案1: 升级pip
pip install --upgrade pip setuptools wheel

# 方案2: 使用国内镜像（如果在中国）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 步骤 3：运行快速测试 (2-3分钟)

```bash
# 运行100回合的快速验证
bash test_main_quick.sh
```

**预期输出：**
```
Test 1: Q-Learning (100 episodes, no GIF)
Device: cuda
Episode 100 | Avg Reward (last 100): -XX.XX | Epsilon: 0.XXXX
✅ Q-Learning training finished!

Test 2: MAPPO (100 episodes, no GIF)
Device: cuda
Episode 100 | Avg Reward (last 100): -XX.XX | Epsilon: 0.0000
✅ MAPPO training finished!

Log files created:
-rw-r--r-- 1 user user 5.1K logs/q_learning_prob0.5_seed42.csv
-rw-r--r-- 1 user user 5.0K logs/mappo_prob0.5_seed42.csv

✅ Quick test completed!
```

---

## 步骤 4：查看结果 (30秒)

```bash
# 查看训练日志
head -20 logs/q_learning_prob0.5_seed42.csv

# 应该看到：
# Episode,Reward,Epsilon
# 1,-27.45,0.9995
# 2,-23.12,0.9990
# ...
```

---

## 🎉 成功！

如果以上步骤都通过，说明环境配置成功！

---

## 📊 下一步

### 选项A：运行单次完整训练 (10-20分钟)

```bash
# MAPPO, 1000 episodes, prob=0.5
python main.py --algo mappo --prob 0.5 --seed 42 --total_episodes 1000 --eval_freq 200
```

---

### 选项B：运行完整批量实验 (6-24小时)

```bash
# 18个实验配置 (2 algos × 3 probs × 3 seeds × 5000 episodes)
bash run_experiments.sh

# 实时监控进度
tail -f experiment_logs/batch_run_*.log
```

**推荐：** 使用 `screen` 或 `tmux` 在后台运行
```bash
# 创建screen会话
screen -S marl_exp

# 运行实验
bash run_experiments.sh

# 分离会话: Ctrl+A, D
# 重新连接: screen -r marl_exp
```

---

### 选项C：生成可视化（需要先有数据）

```bash
# 分析所有结果并生成图表
python plot_results.py

# 查看生成的图表
ls plots/
```

---

## 🧪 可选：运行单元测试

### 测试MAPPO实现
```bash
python test_mappo.py

# 预期输出：
# 🎉 ALL TESTS PASSED! 🎉
```

### 测试Q-Learning
```bash
python test_qlearning.py

# 预期输出：
# 🎉 TEST PASSED! 🎉
```

---

## 🚀 高级用法

### 自定义参数训练

```bash
# MAPPO, 8000 episodes, 80%欺骗概率, seed=2024
python main.py \
    --algo mappo \
    --prob 0.8 \
    --seed 2024 \
    --total_episodes 8000 \
    --eval_freq 400

# Q-Learning + GIF生成
python main.py \
    --algo q_learning \
    --prob 0.5 \
    --seed 100 \
    --total_episodes 3000 \
    --eval_freq 300 \
    --save_gif
```

### 查看所有参数
```bash
python main.py --help
```

---

## 📁 输出文件说明

```
homework-li/
├── logs/                    # CSV训练日志
│   ├── q_learning_prob0.5_seed42.csv
│   └── mappo_prob0.5_seed42.csv
├── checkpoints/             # 模型检查点
│   ├── q_learning_prob0.5_seed42_ep5000_speaker_0.pth
│   └── mappo_prob0.5_seed42_ep5000_listener_0.pth
├── gifs/                    # 评估动画（如果启用）
│   └── mappo_prob0.5_seed42_ep5000.gif
├── plots/                   # 分析图表
│   ├── training_curves_prob0.5.png
│   ├── robustness_comparison.png
│   └── statistics_summary.csv
└── experiment_logs/         # 批量实验日志
    └── batch_run_20241205_120000.log
```

---

## ❓ 常见问题速查

### Q1: 提示"No module named 'torch'"
**A:** 运行 `pip install -r requirements.txt`

### Q2: CUDA out of memory
**A:**
```bash
# 使用CPU
export CUDA_VISIBLE_DEVICES=""
python main.py ...
```

### Q3: 训练太慢
**A:** 减少episodes数量
```bash
python main.py --algo mappo --prob 0.5 --seed 42 --total_episodes 500
```

### Q4: GIF无法生成
**A:**
```bash
pip install imageio imageio-ffmpeg
```

### Q5: 想停止批量实验
**A:**
- 按 `Ctrl+C` 停止
- 已完成的实验结果会保留在 `logs/` 中

---

## 📖 详细文档

- **完整使用说明**: 查看 [README.md](README.md)
- **测试指南**: 查看 [TESTING_GUIDE.md](TESTING_GUIDE.md)
- **技术细节**: 查看 [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

---

## 💡 实验建议

### 第一次使用（学习阶段）
1. ✅ 运行快速测试（100 episodes）
2. ✅ 运行单次完整训练（1000 episodes）
3. ✅ 查看CSV日志，理解数据格式
4. ✅ 手动绘制简单曲线（使用Excel或Python）

### 正式实验（论文/报告）
1. ✅ 运行完整批量实验（18个配置）
2. ✅ 使用 `plot_results.py` 生成专业图表
3. ✅ 分析 `statistics_summary.csv`
4. ✅ 撰写实验报告（参考README中的框架）

### 调试阶段
1. ✅ 使用少量episodes测试（50-100）
2. ✅ 检查单个算法是否正常
3. ✅ 验证日志输出
4. ✅ 测试GIF生成

---

## 🎓 学习路径

```
Day 1: 环境配置 + 快速测试
Day 2: 理解代码结构（algorithm/, envs/）
Day 3: 运行单次完整训练，理解训练过程
Day 4: 启动批量实验（后台运行）
Day 5: 分析结果，生成图表
Day 6: 撰写实验报告
```

---

## ✅ 检查清单

使用前确认：
- [ ] Python 3.8+ 已安装
- [ ] pip 已升级到最新版本
- [ ] `requirements.txt` 中的包已安装
- [ ] `test_main_quick.sh` 运行成功
- [ ] 生成了CSV日志文件

准备实验：
- [ ] 确定要测试的欺骗概率
- [ ] 确定训练回合数（建议5000）
- [ ] 准备好足够的磁盘空间（约500MB per 18 configs）
- [ ] 如果使用GPU，确认CUDA可用

---

**开始您的MARL实验之旅！** 🚀

有问题？查看 [TESTING_GUIDE.md](TESTING_GUIDE.md) 获取详细帮助！

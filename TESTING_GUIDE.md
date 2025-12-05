# 🧪 测试指南

本文档提供了完整的测试步骤和验证方法。

---

## 📋 测试清单

### 阶段1：环境验证 ✅

```bash
# 1. 检查Python版本
python --version  # 应该 >= 3.8

# 2. 检查CUDA（可选）
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 3. 检查依赖包
pip list | grep -E "(torch|pettingzoo|imageio)"
```

**预期输出：**
```
torch                    2.x.x
pettingzoo               1.24.3
imageio                  2.x.x
```

---

### 阶段2：算法单元测试

#### Test 1: MAPPO实现测试

```bash
python test_mappo.py
```

**测试项目：**
1. ✅ Network Initialization
2. ✅ Agent Initialization
3. ✅ Action Selection (stochastic & deterministic)
4. ✅ Transition Storage
5. ✅ GAE Computation
6. ✅ Policy Update
7. ✅ Save/Load Model
8. ✅ Full Episode Simulation

**预期输出：**
```
==================================================
🎉 ALL TESTS PASSED! 🎉
==================================================
```

**如果失败：**
- 检查torch是否正确安装
- 检查pettingzoo版本是否为1.24.3
- 查看错误堆栈，定位问题

---

#### Test 2: Q-Learning保存/加载测试

```bash
python test_qlearning.py
```

**预期输出：**
```
🎉 TEST PASSED! 🎉
```

---

### 阶段3：快速训练验证

#### Test 3: 100回合快速测试

```bash
bash test_main_quick.sh
```

**测试内容：**
- Q-Learning: 100 episodes, prob=0.5, seed=42
- MAPPO: 100 episodes, prob=0.5, seed=42

**预期结果：**
```
logs/
├── q_learning_prob0.5_seed42.csv
└── mappo_prob0.5_seed42.csv
```

**验证方法：**
```bash
# 检查CSV文件
head logs/q_learning_prob0.5_seed42.csv
# 应该看到：
# Episode,Reward,Epsilon
# 1,-XX.XX,0.XXXX
```

**耗时：** 约2-5分钟

---

#### Test 4: 单算法完整训练（可选）

```bash
# 测试Q-Learning (1000 episodes)
python main.py --algo q_learning --prob 0.5 --seed 42 --total_episodes 1000 --eval_freq 200

# 测试MAPPO (1000 episodes)
python main.py --algo mappo --prob 0.5 --seed 42 --total_episodes 1000 --eval_freq 200
```

**预期输出：**
```
Episode 200 | Avg Reward (last 100): -15.23 | Epsilon: 0.8187
[Episode 200] Eval Reward: -14.56
...
✅ Q-Learning training finished!
```

**耗时：** 约10-20分钟/算法

---

### 阶段4：GIF生成测试

#### Test 5: GIF功能验证

```bash
python main.py \
    --algo mappo \
    --prob 0.5 \
    --seed 42 \
    --total_episodes 200 \
    --eval_freq 200 \
    --save_gif
```

**预期结果：**
```
gifs/
└── mappo_prob0.5_seed42_ep200.gif
```

**验证：**
```bash
ls -lh gifs/*.gif
# 应该看到文件大小 > 100KB
```

**如果失败：**
- 检查 `imageio` 是否安装
- 检查 `render_mode='rgb_array'` 是否支持
- 查看控制台警告信息

---

### 阶段5：批量实验（完整测试）

#### Test 6: 小规模批量测试

修改 `run_experiments.sh`，将配置改为：

```bash
SEEDS=(42)               # 只用1个seed
PROBS=(0.5)              # 只用1个概率
TOTAL_EPISODES=500       # 减少回合数
```

然后运行：
```bash
bash run_experiments.sh
```

**预期输出：**
```
[1 / 2] Running: q_learning | Prob=0.5 | Seed=42
✅ Completed: q_learning | Prob=0.5 | Seed=42
[2 / 2] Running: mappo | Prob=0.5 | Seed=42
✅ Completed: mappo | Prob=0.5 | Seed=42
🎉 ALL EXPERIMENTS COMPLETED! 🎉
```

**耗时：** 约10-15分钟

---

#### Test 7: 完整批量实验

恢复原始配置后运行：

```bash
bash run_experiments.sh
```

**配置：**
- 18个实验 (2 algos × 3 probs × 3 seeds)
- 每个5000 episodes

**预期耗时：**
- CPU: 12-24小时
- GPU (2080Ti): 6-12小时

**监控方法：**
```bash
# 实时查看日志
tail -f experiment_logs/batch_run_*.log

# 检查已完成的实验
ls logs/*.csv | wc -l  # 应该逐渐增加到18
```

---

### 阶段6：结果分析测试

#### Test 8: 可视化脚本测试

**前提：** 至少有3个相同prob的CSV文件（来自3个seed）

```bash
# 创建测试数据（如果实验未完成）
# 可以先运行小规模批量测试

python plot_results.py
```

**预期输出：**
```
Found 18 log files in './logs'

Generating plots...
Processing Prob=0.0...
✅ Saved: plots/training_curves_prob0.0.png
Processing Prob=0.5...
✅ Saved: plots/training_curves_prob0.5.png
Processing Prob=0.8...
✅ Saved: plots/training_curves_prob0.8.png
Generating robustness comparison...
✅ Saved: plots/robustness_comparison.png
Computing statistics...
✅ Saved: plots/statistics_summary.csv
🎉 ANALYSIS COMPLETE!
```

**验证：**
```bash
ls plots/
# 应该看到：
# training_curves_prob0.0.png
# training_curves_prob0.5.png
# training_curves_prob0.8.png
# robustness_comparison.png
# statistics_summary.csv
```

---

## 🐛 常见测试问题

### 问题1: ModuleNotFoundError

**错误信息：**
```
ModuleNotFoundError: No module named 'torch'
```

**解决：**
```bash
pip install -r requirements.txt
```

---

### 问题2: CUDA out of memory

**错误信息：**
```
RuntimeError: CUDA out of memory
```

**解决：**
```bash
# 方案1: 使用CPU
export CUDA_VISIBLE_DEVICES=""

# 方案2: 清理显存
python -c "import torch; torch.cuda.empty_cache()"

# 方案3: 串行运行实验（在run_experiments.sh中添加）
sleep 10  # 每个实验后等待10秒
```

---

### 问题3: PettingZoo版本不兼容

**错误信息：**
```
AttributeError: 'ParallelEnv' object has no attribute 'render'
```

**解决：**
```bash
pip install pettingzoo[mpe]==1.24.3 --force-reinstall
```

---

### 问题4: GIF无法播放

**问题：** GIF文件生成但无法打开

**解决：**
```bash
# 安装完整的imageio
pip install imageio[ffmpeg]

# 或手动安装ffmpeg
sudo apt-get install ffmpeg  # Ubuntu/Debian
```

---

### 问题5: 训练卡住不动

**症状：** Episode数长时间不更新

**诊断：**
```bash
# 检查进程
ps aux | grep python

# 检查GPU利用率
nvidia-smi

# 检查日志
tail -f logs/*.csv
```

**解决：**
- 检查是否死锁（Ctrl+C终止）
- 减少batch_size
- 检查环境是否正确reset

---

## ✅ 测试通过标准

### 最小验证（快速）

- [ ] `test_mappo.py` 全部通过
- [ ] `test_qlearning.py` 通过
- [ ] `test_main_quick.sh` 生成2个CSV
- [ ] CSV文件内容正确（有Episode, Reward, Epsilon列）

**耗时：** < 10分钟

---

### 完整验证（推荐）

- [ ] 所有单元测试通过
- [ ] 单算法训练1000 episodes正常
- [ ] GIF正常生成
- [ ] 批量实验至少完成2个配置
- [ ] `plot_results.py` 成功生成图表

**耗时：** 约1-2小时

---

### 生产级验证（完整）

- [ ] 18个实验全部完成
- [ ] 所有CSV文件完整（5000行）
- [ ] 所有图表生成
- [ ] 统计汇总无异常值
- [ ] GIF可正常播放

**耗时：** 12-24小时

---

## 📊 性能基准

### 参考性能指标（prob=0.5）

| 算法 | 收敛Episode | 最终奖励 | 训练时间/1000ep |
|------|-------------|----------|----------------|
| Q-Learning | ~2000 | -8 to -12 | 10-15分钟 (GPU) |
| MAPPO | ~1500 | -6 to -10 | 8-12分钟 (GPU) |

**注：** 实际结果可能因硬件、随机种子而异

---

## 🔍 调试技巧

### 1. 打印中间输出

在`main.py`中添加：
```python
# 在训练循环中
if episode % 10 == 0:
    print(f"Episode {episode}: Reward={episode_reward:.2f}")
```

### 2. 检查梯度

在`algorithm/mappo.py`中：
```python
# 在update()方法中
for name, param in self.actor.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
```

### 3. 可视化Q值

```python
# 在Q-Learning中
with torch.no_grad():
    q_values = agent.q_net(state_tensor)
    print(f"Q-values: {q_values}")
```

---

## 📝 测试报告模板

```markdown
# 测试报告

## 环境
- 系统: Ubuntu 20.04
- Python: 3.9.7
- PyTorch: 2.0.1
- CUDA: 11.8
- GPU: NVIDIA 2080Ti

## 测试结果

### 单元测试
- MAPPO测试: ✅ PASS
- Q-Learning测试: ✅ PASS

### 快速验证
- 100 episodes测试: ✅ PASS
- 耗时: 3分钟

### 完整训练
- Q-Learning (5000 ep): ✅ PASS
- MAPPO (5000 ep): ✅ PASS
- GIF生成: ✅ PASS

### 批量实验
- 完成进度: 18/18
- 总耗时: 8小时15分钟
- 失败次数: 0

### 可视化
- 训练曲线: ✅ 正常
- 鲁棒性对比: ✅ 正常
- 统计汇总: ✅ 正常

## 问题与解决
1. 问题：...
   解决：...

## 结论
所有测试通过，代码可正常运行。
```

---

**测试愉快！如有问题请查看README.md** 🚀

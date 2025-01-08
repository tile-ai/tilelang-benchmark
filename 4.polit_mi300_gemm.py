import matplotlib.pyplot as plt
import numpy as np

matmul_providers = [
    "M0",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
    "M10",
    "M11",
    "M12",
]

matmul_times_data = [
    (
        "TileLang-W$_{FP16}$A$_{FP16}$",
        [
            0.0906,
            0.2612,
            0.9501,
            0.8848,
            0.1537,
            0.5213,
            1.9331,
            1.8919,
            0.1697,
            1.0426,
            4.1409,
            4.3970,
            0.2981,
        ],
    ),
    (
        "Triton-W$_{FP16}$A$_{FP16}$",
        [
            0.1208,
            0.3604,
            1.1863,
            1.2769,
            0.1784,
            0.7149,
            2.3485,
            2.4977,
            0.2384,
            1.4390,
            4.7919,
            5.4937,
            0.3898,
        ],
    ),
    (
        "rocBLAS-W$_{FP16}$A$_{FP16}$",
        [
            0.1610,
            0.3030,
            0.9510,
            0.9220,
            0.2340,
            0.6320,
            1.7300,
            2.2960,
            0.3090,
            1.0800,
            3.5700,
            4.6810,
            0.3880,
        ],
    ),
]

times_data = matmul_times_data
providers = matmul_providers

# 只取前8个 provider 做演示
num_ops = 8
providers = providers[:num_ops]
for i in range(len(times_data)):
    times_data[i] = (times_data[i][0], times_data[i][1][:num_ops])

# 设定 rocBLAS 作为 baseline（参考）
_1x_baseline = "TileLang-W$_{FP16}$A$_{FP16}$"
_1x_baseline_times = dict(times_data)[_1x_baseline]

# 计算“相对于 rocBLAS 的归一化延迟”，即 normalized_latency = times / baseline_times
normalized_latency_data = []
for label, times in times_data:
    if label != _1x_baseline:
        normalized_latency = [
            (t / p_i if p_i != 0 else 0)
            for t, p_i in zip(times, _1x_baseline_times)
        ]
        normalized_latency_data.append((label, normalized_latency))
    else:
        # baseline 自身的 normalized latency 恒为 1
        normalized_latency_data.append((label, [1.0] * len(times)))

# 如果想手动指定颜色，可以将上面 colers_sets 替换为如下样式
colers_sets = [
    # nilu
    # (20 / 255, 54 / 255, 95 / 255),
    # (118 / 255, 162 / 255, 185 / 255),
    (243 / 255, 191 / 255, 202 / 255),
    (191 / 255, 217 / 255, 229 / 255),
    # (214 / 255, 79 / 255, 56 / 255),
    # (112 / 255, 89 / 255, 146 / 255),
    # dori
    # (169 / 255, 115 / 255, 153 / 255),
    #  (248 / 255, 242 / 255, 236 / 255),
    # (214 / 255, 130 / 255, 148 / 255),
    # coller
    # (124 / 255, 134 / 255, 65 / 255),
    # (185 / 255, 198 / 255, 122 / 255),
    (248 / 255, 231 / 255, 210 / 255),
    (182 / 255, 110 / 255, 151 / 255),
]
hatch_patterns = ["*", "-", "\\", ".", "o", "O", "+", "x",]

# Create an array for x-axis positions
x = np.arange(len(providers))

# Set the width of the bars
total_bars = len(normalized_latency_data)
available_width = 0.76
bar_width = available_width / total_bars

# Plotting
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(providers))

# baseline: 画一条 y=1 的水平虚线
ax.axhline(y=1, color="black", linestyle="dashed", label=_1x_baseline)

# 逐列叠加画 bar
for i, (label, latency) in enumerate(normalized_latency_data):
    ax.bar(
        x + i * bar_width,
        latency,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=hatch_patterns[i % len(hatch_patterns)],
        color=colers_sets[i % len(colers_sets)],
    )

# 设置 Y 轴范围
max_latency = max([max(latency) for _, latency in normalized_latency_data])
ax.set_ylim(0, max_latency * 1.2)

# 调整图例位置和大小
legend_fontsize = 15
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.66, 1.0),
    ncol=2,
    fontsize=legend_fontsize,
    frameon=False,
)

# 设置坐标轴
ax.set_ylabel(
    "Normalized Latency (vs TileLang)", fontsize=17
)
ax.set_xticks(x + len(normalized_latency_data) * bar_width / len(times_data))
ax.set_xticklabels(providers, fontsize=14)

# 去掉网格
ax.grid(False)

# 标题
plt.title(
    "Normalized Latency of GEMM W$_{FP16}$A$_{FP16}$ on MI300X", fontsize=18
)

# 保存结果
plt.savefig(
    "pdf/op_benchmark_mi300_fp16_gemm_normalized_latency.pdf",
    bbox_inches="tight",
)
plt.savefig(
    "png/op_benchmark_mi300_fp16_gemm_normalized_latency.png",
    bbox_inches="tight",
    transparent=False,
    dpi=255,
)

plt.show()

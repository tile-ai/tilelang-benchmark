import matplotlib.pyplot as plt
import numpy as np
from data.data_float16_gemv import matmul_times_data as times_data
from data.data_float16_gemv import matmul_providers as providers

num_ops = 8
providers = providers[:num_ops]
for i in range(len(times_data)):
    times_data[i] = (times_data[i][0], times_data[i][1][:num_ops])

_1x_baseline = "cuBLAS-W$_{FP16}$A$_{FP16}$"
_1x_baseline_times = dict(times_data)[_1x_baseline]

# 计算其他方法相对加速比
speed_up_data = []
for label, times in times_data:
    if label != _1x_baseline:
        speed_up = [
            p_i / t if t != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
        ]
        speed_up_data.append((label, speed_up))
    else:
        speed_up_data.append((label, [1.0] * len(times)))

cmap = plt.cm.get_cmap("prism_r")  # 选择 colormap，例如 'tab10', 'viridis', 'plasma', 等
colers_sets = [cmap(i) for i in range(len(speed_up_data))]  # 根据数据长度分配颜色


print(speed_up_data)
# Data
colers_sets = [
    # nilu
    # (20 / 255, 54 / 255, 95 / 255),
    # (118 / 255, 162 / 255, 185 / 255),
    (191 / 255, 217 / 255, 229 / 255),
    # (214 / 255, 79 / 255, 56 / 255),
    # (112 / 255, 89 / 255, 146 / 255),
    # dori
    # (169 / 255, 115 / 255, 153 / 255),
    (248 / 255, 242 / 255, 236 / 255),
    # (214 / 255, 130 / 255, 148 / 255),
    (243 / 255, 191 / 255, 202 / 255),
    # coller
    (124 / 255, 134 / 255, 65 / 255),
    (185 / 255, 198 / 255, 122 / 255),
    (248 / 255, 231 / 255, 210 / 255),
    (182 / 255, 110 / 255, 151 / 255),
]
hatch_patterns = ["x", "\\", "*", "o", "O", ".", "-", "+"]

# Create an array for x-axis positions
x = np.arange(len(providers))

# Set the width of the bars
total_bars = len(speed_up_data)
available_width = 0.76
bar_width = available_width / total_bars

# Plotting
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(providers))

# Draw cublas as a horizontal dashed line
ax.axhline(y=1, color="black", linestyle="dashed", label="cuBLAS-W$_{FP16}$A$_{FP16}$")


# Draw a vertical dashed line to separate the two data parts
def get_inverse(a):
    inverse = [1 / i for i in a]
    return inverse

# Create bars using a loop
for i, (label, speedup) in enumerate(speed_up_data):
    rec = ax.bar(
        x + i * bar_width,
        speedup,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=hatch_patterns[i % 8],
        color=colers_sets[i],
    )

# set y-limit
max_speedup = max([max(speedup) for _, speedup in speed_up_data])
ax.set_ylim(0, max_speedup * 1.2)
# 调整图例位置和大小
legend_fontsize = 15

handles, labels = ax.get_legend_handles_labels()

# 将图例放置在图表中间
ax.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.48, 1.02),
    ncol=3,
    fontsize=legend_fontsize,
    frameon=False,
)
# X-axis and labels
# ax.set_xlabel("Shapes from LLM", fontsize=12)
ax.set_ylabel("Speedup vs cuBLAS-W$_{FP16}$A$_{FP16}$", fontsize=20)
ax.set_xticks(x + len(speed_up_data) * bar_width / len(times_data))
ax.set_xticklabels(providers, fontsize=14)
ax.grid(axis="y", linestyle="--", linewidth=0.5)

# disable grid
ax.grid(False)

# add a title
plt.title("Speedup of GEMV W$_{FP16}$A$_{FP16}$ on A100", fontsize=20)

# Save the plot to a file
plt.savefig("pdf/op_benchmark_a100_fp16_gemv.pdf", bbox_inches='tight')
plt.savefig("png/op_benchmark_a100_fp16_gemv.png",  bbox_inches='tight', transparent=False, dpi=255)

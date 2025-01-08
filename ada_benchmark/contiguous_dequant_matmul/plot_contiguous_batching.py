import matplotlib.pyplot as plt
import numpy as np

# 从您给定的外部文件中导入数据
from data.n16384_k16384_data import matmul_times_data as matmul_times_data_n16384_k16384
from data.n16384_k16384_data import matmul_providers as matmul_providers_n16384_k16384
from data.n8192_k8192_data import matmul_times_data as matmul_times_data_n8192_k8192
from data.n8192_k8192_data import matmul_providers as matmul_providers_n8192_k8192
from data.n8192_k28672_data import matmul_times_data as matmul_times_data_n8192_k28672
from data.n8192_k28672_data import matmul_providers as matmul_providers_n8192_k28672
from data.n28672_k8192_data import matmul_times_data as matmul_times_data_n28672_k8192
from data.n28672_k8192_data import matmul_providers as matmul_providers_n28672_k8192

# 您提供的颜色集(足够区分多条曲线)
colers_sets = [
    # nilu
    (20 / 255, 54 / 255, 95 / 255),
    (118 / 255, 162 / 255, 185 / 255),
    (191 / 255, 217 / 255, 229 / 255),
    (214 / 255, 79 / 255, 56 / 255),
    (112 / 255, 89 / 255, 146 / 255),
    # dori
    (169 / 255, 115 / 255, 153 / 255),
    (248 / 255, 242 / 255, 236 / 255),
    (214 / 255, 130 / 255, 148 / 255),
    (243 / 255, 191 / 255, 202 / 255),
    # coller
    (124 / 255, 134 / 255, 65 / 255),
    (185 / 255, 198 / 255, 122 / 255),
    (248 / 255, 231 / 255, 210 / 255),
    (182 / 255, 110 / 255, 151 / 255),
]

# 四组数据场景
scenarios = [
    ("n16384k16384", matmul_providers_n16384_k16384, matmul_times_data_n16384_k16384),
    ("n8192k8192", matmul_providers_n8192_k8192, matmul_times_data_n8192_k8192),
    ("n8192k28672", matmul_providers_n8192_k28672, matmul_times_data_n8192_k28672),
    ("n28672k8192", matmul_providers_n28672_k8192, matmul_times_data_n28672_k8192),
]

# 假设所有场景下的method数量相同，以第一组为参考
all_methods = scenarios[0][2]
method_labels = [m[0] for m in all_methods]
num_methods = len(method_labels)

# 为不同的方法定义标记风格（marker）
markers = ['o', 's', '^', 'D', 'v', 'X', 'P', 'h', '<', '>']
markers = markers * ((num_methods // len(markers)) + 1) 
markers = markers[:num_methods]

fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=False)
fig.suptitle("Normalized Speedup vs cuBLAS-W$_{FP16}$A$_{FP16}$ on A100", fontsize=30, y=1.05)

for ax, (title, providers, times_data) in zip(axes, scenarios):
    # 将provider转换为数字（假设是批大小等，可根据实际情况修改）
    
    # 找到cuBLAS的执行时间
    cublas_idx = method_labels.index("cuBLAS-W$_{FP16}$A$_{FP16}$")  # 假设方法名称是"cuBLAS"
    cublas_times = np.array(times_data[cublas_idx][1], dtype=float)
    max_speedup = 0
    # 对每种方法绘制归一化加速比
    for i, (method_label, method_times) in enumerate(times_data):
        method_times = np.array(method_times, dtype=float)
        x_vals = np.arange(len(providers))
        # 计算归一化加速比
        normalized_speedup = cublas_times / method_times
        max_val = np.max(normalized_speedup)
        max_speedup = max(max_speedup, max_val)
        ax.plot(x_vals, normalized_speedup, 
                marker=markers[i], 
                color=colers_sets[i], 
                label=method_label, 
                linestyle='-')
    ax.set_ylim(0, max_speedup * 1.2)
    ax.set_title(title, fontsize=24)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(providers, fontsize=16)
    ax.grid(False)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_xlabel("M", fontsize=18)

# 设置Y轴标签
axes[0].set_ylabel("Normalized Speedup vs \n cuBLAS-W$_{FP16}$A$_{FP16}$)", fontsize=20)

# 添加图例
handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, loc='upper center', ncol=num_methods, bbox_to_anchor=(0.5, 0.97), fontsize=17, frameon=False)

fig.tight_layout()

# 保存图像
plt.savefig("pdf/contiguous_batching_benchmark_a100.pdf", bbox_inches='tight')
plt.savefig("png/contiguous_batching_benchmark_a100.png", bbox_inches='tight', transparent=False, dpi=255)
import numpy as np
import matplotlib.pyplot as plt

data = [
    # batch=128
    {"batch": 128, "kv_ctx": 1024,  "torch": 0.40, "triton": 18.31, "tilelang": 107.7233317, "aiter": 148.242023},
    {"batch": 128, "kv_ctx": 2048,  "torch": 0.45, "triton": 23.96, "tilelang": 114.3736840, "aiter": 148.128263},
    {"batch": 128, "kv_ctx": 4096,  "torch": 0.52, "triton": 34.00, "tilelang": 125.1191051, "aiter": 153.470299},
    {"batch": 128, "kv_ctx": 8192,  "torch": 0.54, "triton": 43.81, "tilelang": 133.8617848, "aiter": 154.067245},
    {"batch": 128, "kv_ctx": 16384, "torch": 0.55, "triton": 51.53, "tilelang": 138.7824009, "aiter": 157.541018},
    # batch=64
    {"batch": 64,  "kv_ctx": 1024,  "torch": 0.39, "triton": 17.10, "tilelang": 112.3900851, "aiter": 92.877927},
    {"batch": 64,  "kv_ctx": 2048,  "torch": 0.43, "triton": 22.99, "tilelang": 113.1726112, "aiter": 117.593644},
    {"batch": 64,  "kv_ctx": 4096,  "torch": 0.51, "triton": 33.37, "tilelang": 119.4211435, "aiter": 120.832907},
    {"batch": 64,  "kv_ctx": 8192,  "torch": 0.54, "triton": 43.51, "tilelang": 129.4906100, "aiter": 136.262128},
    {"batch": 64,  "kv_ctx": 16384, "torch": 0.55, "triton": 51.03, "tilelang": 135.4093992, "aiter": 137.664726},
]

# Data
colers_sets = [
    # nilu
    # (20 / 255, 54 / 255, 95 / 255),
    # (118 / 255, 162 / 255, 185 / 255),
    # (191 / 255, 217 / 255, 229 / 255),
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
hatch_patterns = [
    "x",
    "\\",
    "*",
    "o",
    "O",
    ".",
    "-",
    "+"
]



# 新增绘图代码
def plot_comparison():
    plt.figure(figsize=(16, 6))
    
    # 准备数据
    batches = {64: [], 128: []}
    for d in data:
        batches[d["batch"]].append(d)
    
    # 创建子图
    for idx, (batch, group_data) in enumerate(batches.items(), 1):
        ax = plt.subplot(1, 2, idx)
        
        # 提取数据
        kv_ctx = [d["kv_ctx"] for d in group_data]
        frameworks = {
            "PyTorch": [d["torch"] for d in group_data],
            "Triton": [d["triton"] for d in group_data],
            "TileLang": [d["tilelang"] for d in group_data],
            "Aiter-ASM": [d["aiter"] for d in group_data]
        }
        
        bar_width = 0.22
        x = np.arange(len(kv_ctx))
        
        for i, (name, values) in enumerate(frameworks.items()):
            offset = bar_width * i
            bars = ax.bar(x + offset, values, bar_width, label=name, color=colers_sets[i], hatch=hatch_patterns[i])
            
            for bar in bars:
                height = bar.get_height()
                name = f'{height:.1f}'
                if len(name) > 3:
                    name = name[:3]
                ax.annotate(name,
                            xy=(bar.get_x() + bar.get_width() / 2 - 0.01, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=11, rotation=0, color="black")

        ax.set_title(f'Batch Size = {batch}', fontsize=20, pad=5)
        ax.set_xlabel('KV Context Length', fontsize=16)
        ax.set_ylabel('Throughput (TFLOPs)', fontsize=20)
        ax.set_xticks(x + bar_width + 0.05)
        ax.set_xticklabels([f"{ctx//1024}K" for ctx in kv_ctx], fontsize=14)
        max_height = max(max(values) for values in frameworks.values())
        ax.set_ylim(0, max_height * 1.1)
        
        # 移除网格线
        ax.grid(False)

    # 创建单个图例
    handles, labels = ax.get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='upper center', ncol=2, fontsize=16, bbox_to_anchor=(0.5, 0.92), fancybox=False, shadow=False, frameon=False,)
    # add a big title
    plt.suptitle('FlashMLA Performance on AMD MI300x', fontsize=24)

    plt.tight_layout(pad=3.0)
    plt.savefig("flashmla-amd.pdf", bbox_inches='tight', dpi=300)
    plt.savefig("flashmla-amd.png", bbox_inches='tight', dpi=300)
    plt.close()

plot_comparison()


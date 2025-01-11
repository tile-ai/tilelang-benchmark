import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

rows, cols = 8, 32
data = np.zeros((rows, cols), dtype=int)
for c in range(cols):
    val = c
    data[:, c] = val  # 每一行都相同数值

def int_to_hex_str(i):
    # return f"{i:02X}"
    return f"{i:02}"

# 1) 创建图形
plt.figure(figsize=(12, 3))
ax = plt.gca()

# 2) 手动绘制灰色网格
nrows, ncols = data.shape
for i in range(nrows):
    for j in range(ncols):
        # 设置灰度，按数值映射到一个灰度值（0到255）
        gray_val = 0.7
        rect = patches.Rectangle((j, i), 1, 1, linewidth=0.5, 
                                 edgecolor='black', facecolor=(gray_val, gray_val, gray_val))
        ax.add_patch(rect)

        # 添加十六进制文本
        val_hex = int_to_hex_str(data[i, j])
        ax.text(j + 0.5, i + 0.5, val_hex, ha='center', va='center', color='black', fontsize=14)

ax.set_xlim(0, ncols)
ax.set_ylim(0, nrows)
plt.xticks([])
plt.yticks([])

# 4) 保存图像
plt.tight_layout()
plt.savefig(
    "pdf/0.before_layout_inference.pdf",
    bbox_inches="tight",
)
plt.savefig(
    "png/0.before_layout_inference.png",
    bbox_inches="tight",
    transparent=False,
    dpi=255,
)
# save svg
import os
os.makedirs("svg", exist_ok=True)
plt.savefig("svg/0.before_layout_inference.svg", format="svg", bbox_inches="tight")

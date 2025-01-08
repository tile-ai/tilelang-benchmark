import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_swizzle_layout(row_idx, col_idx, row_size, dtype="float16"):
    from tilelang import tvm as tvm
    from tvm import DataType
    BANK_SIZE_BYTES = 128
    if isinstance(dtype, str):
        dtype = DataType(dtype)
    col_idx_outer, col_idx_inner = col_idx // (BANK_SIZE_BYTES // dtype.bits), col_idx % (
        BANK_SIZE_BYTES // dtype.bits)
    #  use transaction bits to support diverse dtype.
    #  for fp16, 64 elems * 16 bits = 1024 bits, 32 elems * 32 bits = 512 bits
    #  for int8, 128 elems * 8 bits = 1024 bits, 64 elems * 8 bits = 512 bits
    coalescent_bits = dtype.bits * row_size
    # permutation on 4 banks, each bank has 32 bits
    bank_elems = BANK_SIZE_BYTES // dtype.bits
    new_col_idx_outer = None

    if coalescent_bits % 1024 == 0:
        #   Use 8 * 8 permuted layout
        #   Every number below corresponds to 8 consecutive fp16 number in shared mem, i.e. one read
        #   Every row below corresponds to 32 banks
        #   0  1  2  3  4  5  6  7    ==>    0  1  2  3  4  5  6  7
        #   0  1  2  3  4  5  6  7    ==>    1  0  3  2  5  4  7  6
        #   0  1  2  3  4  5  6  7    ==>    2  3  0  1  6  7  4  5
        #   0  1  2  3  4  5  6  7    ==>    3  2  1  0  7  6  5  4
        #   0  1  2  3  4  5  6  7    ==>    4  5  6  7  0  1  2  3
        #   0  1  2  3  4  5  6  7    ==>    5  4  7  6  1  0  3  2
        #   0  1  2  3  4  5  6  7    ==>    6  7  4  5  2  3  0  1
        #   0  1  2  3  4  5  6  7    ==>    7  6  5  4  3  2  1  0
        row_idx_sub = row_idx % bank_elems
        new_col_idx_outer = col_idx_outer ^ row_idx_sub
    else:
        assert coalescent_bits % 512 == 0
        #  Use 8 * 4 permuted layout
        #  Every number below corresponds to 8 consecutive fp16 number in shared mem, i.e. one read
        #  Every row below corresponds to 16 banks
        #  0  1  2  3    ==>    0  1  2  3
        #  0  1  2  3    ==>    0  1  2  3
        #  0  1  2  3    ==>    1  0  3  2
        #  0  1  2  3    ==>    1  0  3  2
        #  0  1  2  3    ==>    2  3  0  1
        #  0  1  2  3    ==>    2  3  0  1
        #  0  1  2  3    ==>    3  2  1  0
        #  0  1  2  3    ==>    3  2  1  0
        #  View with 8 elements per row:
        #  0  1  2  3  4  0  1  2  3    ==>    0  1  2  3  0  1  2  3
        #  0  1  2  3  4  0  1  2  3    ==>    1  0  3  2  1  0  3  2
        #  0  1  2  3  4  0  1  2  3    ==>    2  3  0  1  2  3  0  1
        #  0  1  2  3  4  0  1  2  3    ==>    3  2  1  0  3  2  1  0
        row_idx_sub = row_idx % bank_elems
        #  Interleave elems per byte
        interleave_elems = 32 // dtype.bits
        new_col_idx_outer = col_idx_outer ^ (row_idx_sub // interleave_elems)

    return row_idx, new_col_idx_outer * bank_elems + col_idx_inner

# get colormap
cmap = plt.get_cmap('rainbow', 32)

threads = 32
# 根据采样生成颜色列表
raw_colors = [cmap(i) for i in range(threads)]
colors = raw_colors
elems_per_thread = 8

rows, cols = 8, 32
data = np.zeros((rows, cols), dtype=int)
for c in range(cols):
    val = c
    data[:, c] = val  # 每一行都相同数值
# 1) 生成 swizzled_data
swizzled_data = np.zeros_like(data)
nrows, ncols = data.shape
for i in range(nrows):
    for j in range(ncols):
        swizzled_i, swizzled_j = get_swizzle_layout(i, j, ncols)
        swizzled_data[swizzled_i, swizzled_j] = data[i, j]

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
        ri = (nrows - 1) - i
        swizzled_i, swizzled_j = get_swizzle_layout(ri, j, ncols)
        id = swizzled_i * ncols + swizzled_j
        thread_id = id // elems_per_thread
        color = colors[swizzled_j]
        rect = patches.Rectangle((j, i), 1, 1, linewidth=0.5, 
                                 edgecolor='black', facecolor=color)
        ax.add_patch(rect)

        # 添加十六进制文本
        val_hex = int_to_hex_str(swizzled_data[ri, j])

        ax.text(j + 0.5, i + 0.5, val_hex, ha='center', va='center', color='black', fontsize=14)

ax.set_xlim(0, ncols)
ax.set_ylim(0, nrows)
plt.xticks([])
plt.yticks([])

# 4) 保存图像
plt.tight_layout()
plt.savefig(
    "pdf/2.layout_swizzled.pdf",
    bbox_inches="tight",
)
plt.savefig(
    "png/2.layout_swizzled.png",
    bbox_inches="tight",
    transparent=False,
    dpi=255,
)

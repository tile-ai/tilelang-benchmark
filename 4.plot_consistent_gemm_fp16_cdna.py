import matplotlib.pyplot as plt
import numpy as np
colormap = plt.cm.summer# LinearSegmentedColormap
gemm_provider = ["M0","M1","M2","M3","M4","M5","M6","M7","M8","M9","M10","M11","M12", "M13", "M14", "M15"]

gemm_times_data = [
    ('cuBLAS', (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)),
    ('Triton-RTX4090', (
        1.057756311,
        1.122238439,
        1.068354726,
        1.02900137,
        1.098310271,
        1.059065265,
        1.076218166,
        1.00879544,
        1.112138798,
        1.132122181,
        1.103256832,
        0.912380094,
        1.140279153,
        1.131313728,
        0.995446274,
        0.817311392,
)),
    ('TileLang-RTX4090', (
        1.031007752,
        1.079002079,
        1.063856169,
        1.054937868,
        1.072,
        1.025613661,
        1.074842767,
        1.041515651,
        0.960144928,
        1.142857143,
        1.119362113,
        1.036712419,
        1.149350649,
        1.146527386,
        1.144559628,
        1.014893265,
    )),
    ('Triton-A100', (        
        1.119135782,
        1.047157362,
        1.104746704,
        0.884558298,
        1.276198292,
        1.095171952,
        1.005747139,
        0.930746534,
        1.030117904,
        1.037261526,
        0.973705439,
        0.955358957,
        1.083685518,
        0.891525419,
        0.889902312,
        0.749902698,)),
    ('TileLang-A100', (        
        1.005649718,
        0.953571429,
        1.178958785,
        0.941208198,
        1.134020619,
        1.013282732,
        0.992522847,
        0.957142857,
        0.923303835,
        0.982056256,
        0.974006532,
        0.917402997,
        0.980427046,
        0.957977034,
        0.993610224,
        0.963014241,)),
    ('Triton-H100', (
        1.016853933,
        1.017377567,
        0.868672259,
        0.91579743,
        1.071935157,
        0.900181019,
        0.939193042,
        0.904193808,
        1.046610169,
        0.894683027,
        0.932900199,
        0.914866443,
        0.944474687,
        0.909524112,
        0.911297682,
        0.908247048,
)),
    ('TileLang-H100', (
        1.128898129,
        1.090909091,
        0.890437361,
        1.10325988,
        1.243243243,
        1.046602526,
        0.949139111,
        1.147501393,
        1.148837209,
        1.07477353,
        1.089215728,
        1.075962405,
        1.060513447,
        1.021770164,
        0.98013427,
        0.982298625,
    )),
    ('rocBLAS', (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)),
    ('Triton-MI300X', (
        1.332430016,
        0.840621389,
        0.801632595,
        0.722033575,
        1.312018023,
        0.884042359,
        0.736635768,
        0.919255279,
        1.295925895,
        0.750511988,
        0.745007175,
        0.852064912,
        0.995495241,
        0.810292553,
        0.725370662,
        1.012706186,
    )),
    ('TileLang-MI300X', (
        1.776394598,
        1.159938508,
        1.000935229,
        1.042062307,
        1.522236301,
        1.212251082,
        0.894957466,
        1.213615281,
        1.820852809,
        1.035873826,
        0.862131213,
        1.064594578,
        1.301794791,
        1.01619447,
        0.831172698,
        1.266178748,
    )),
]

providers = gemm_provider
times_data = gemm_times_data[-3:]
num_ops = 8
providers = providers[:num_ops]
offset = 4
for i in range(len(times_data)):
    times_data[i] = (times_data[i][0], times_data[i][1][offset:num_ops + offset])
_1x_baseline = "rocBLAS"
_1x_baseline_times = dict(times_data)[_1x_baseline]

# 计算其他方法相对加速比
speed_up_data = []
for label, times in times_data:
    speed_up_data.append((label, times))

print(speed_up_data)
# Data
# colers_sets = [
#     # nilu
#     # (20 / 255, 54 / 255, 95 / 255),
#     # (118 / 255, 162 / 255, 185 / 255),
#     # (191 / 255, 217 / 255, 229 / 255),
#     # (214 / 255, 79 / 255, 56 / 255),
#     # (112 / 255, 89 / 255, 146 / 255),
#     # dori
#     (169 / 255, 115 / 255, 153 / 255),
#     (248 / 255, 242 / 255, 236 / 255),
#     (214 / 255, 130 / 255, 148 / 255),
#     (243 / 255, 191 / 255, 202 / 255),
#     # coller
#     (124 / 255, 134 / 255, 65 / 255),
#     (185 / 255, 198 / 255, 122 / 255),
#     (248 / 255, 231 / 255, 210 / 255),
#     (182 / 255, 110 / 255, 151 / 255),
# ]
# colers_sets = [
#     # nilu
#     # (20 / 255, 54 / 255, 95 / 255),
#     # (118 / 255, 162 / 255, 185 / 255),
#     (243 / 255, 191 / 255, 202 / 255),
#     (191 / 255, 217 / 255, 229 / 255),
#     # (214 / 255, 79 / 255, 56 / 255),
#     # (112 / 255, 89 / 255, 146 / 255),
#     # dori
#     (169 / 255, 115 / 255, 153 / 255),
#     #  (248 / 255, 242 / 255, 236 / 255),
#     (214 / 255, 130 / 255, 148 / 255),
#     # coller
#     # (124 / 255, 134 / 255, 65 / 255),
#     # (185 / 255, 198 / 255, 122 / 255),
#     (248 / 255, 231 / 255, 210 / 255),
#     (182 / 255, 110 / 255, 151 / 255),
# ]
colormap = plt.cm.rainbow

colers_sets = [colormap(i) for i in np.linspace(0, 1, len(speed_up_data))]

hatch_patterns = ["x", "\\", "*", "o", "O", ".", "-", "+"]

# Create an array for x-axis positions
x = np.arange(len(providers))

# Set the width of the bars
bar_width = 0.24

# Plotting
fig, ax = plt.subplots(figsize=(6, 3))
x = np.arange(len(providers))

# Draw cublas as a horizontal dashed line
ax.axhline(y=1, color="black", linestyle="dashed", label="rocBLAS")


# Draw a vertical dashed line to separate the two data parts
def get_inverse(a):
    inverse = [1 / i for i in a]
    return inverse

# Create bars using a loop
for i, (label, speedup) in enumerate(speed_up_data):
    color = colormap(i / len(speed_up_data))
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
ax.set_ylim(0, max_speedup * 1.4)
# 调整图例位置和大小
legend_fontsize = 14

handles, labels = ax.get_legend_handles_labels()

# 将图例放置在图表中间
ax.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.04),
    ncol=2,
    fontsize=legend_fontsize,
    frameon=False,
)
# X-axis and labels
ax.set_xlabel("Shapes from LLM", fontsize=18)
ax.set_ylabel("Speedup vs rocBLAS", fontsize=20)
ax.set_xticks(x + len(speed_up_data) * bar_width / len(times_data))
ax.set_xticklabels(providers)
ax.grid(axis="y", linestyle="--", linewidth=0.5)

# disable grid
ax.grid(False)


# add a title
plt.title("MI300X", fontsize=18)

# Save the plot to a file
plt.savefig("pdf/op_benchmark_consistent_gemm_fp16_amd.pdf", bbox_inches='tight')
plt.savefig("png/op_benchmark_consistent_gemm_fp16_amd.png", bbox_inches='tight', transparent=False, dpi=255)

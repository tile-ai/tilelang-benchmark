from pygments.lexers import PythonLexer
from pygments.token import Keyword, Name
from pygments import highlight
from pygments.formatters import HtmlFormatter, SvgFormatter
from pygments.style import Style
from pygments.token import Token
from pygments.styles.default import DefaultStyle


class CustomPythonLexer(PythonLexer):
    # TILELANG_LANGUAGE_KEYWORDS = {"T"}
    EXTRA_KEYWORDS = {
        "T",
        "fill",
        "jit",
        "Buffer",
        "Pipelined",
        "Parallel",
        "alloc_shared",
        "alloc_fragment",
        "clear",
        "copy",
        "gemm",
        "ceildiv",
        "Kernel",
        "annotate_layout",
        "use_swizzle",
        "float32",
    }
    VARIABLES = {
        "A",
        "B",
        "C",
        "A_shared",
        "B_shared",
        "C_local",
    }
    HIGHLIGHT_KEYWORDS = {
        "your_customized_layout",
    }

    def get_tokens_unprocessed(self, text):
        for index, token, value in super().get_tokens_unprocessed(text):
            if value in self.EXTRA_KEYWORDS:
                yield index, Token.Keyword.TileLangSyntax, value  # 自定义 Token
            elif value in self.VARIABLES:
                yield index, Token.Name.Variable, value
            elif value in self.HIGHLIGHT_KEYWORDS:
                yield index, Token.Keyword.Highlight, value
            else:
                yield index, token, value


class CustomStyle(DefaultStyle):  # 继承 DefaultStyle 或其他主题样式
    # 在原有 styles 的基础上扩展
    styles = dict(DefaultStyle.styles)  # 保留 DefaultStyle 的所有样式

    # 添加自定义样式
    styles[Token.Keyword.TileLangSyntax] = "bold #9467bd"
    styles[Token.Keyword] = "bold #FF0000"  # 红色粗体
    styles[Token.Operator.Word] = "bold #1f77b4"
    styles[Token.Name.Function] = "bold #0000FF"
    # styles[Token.Name.Class] = "bold #9467bd"
    styles[Token.Name.Namespace] = "bold #9467bd"
    styles[Token.Name.Variable] = "bold #d62728"
    styles[Token.Literal.Number] = "bold #1f77b4"
    styles[Token.Keyword.Highlight] = "bold #FF0000"  # 橙色粗体


# 测试代码
code = r"""
@tilelang.jit
def flash_attention(
    Q: T.Buffer,
    K: T.Buffer,
    V: T.Buffer,
    Output: T.Buffer,
):
    with T.Kernel(
            T.ceildiv(seq_len, block_M), batch, heads, threads=threads) as (bx, by, bz):
        Q_shared = T.alloc_shared(Q_shared_shape, dtypeQKV)
        Q_local = T.alloc_fragment(Q_local_shape, dtypeQKV)
        K_shared = T.alloc_shared(K_shared_shape, dtypeQKV)
        V_shared = T.alloc_shared(V_shared_shape, dtypeQKV)
        score_QK = T.alloc_fragment((block_M, block_N), dtypeAccu)
        score_QK_sum = T.alloc_fragment((block_M), dtypeAccu)
        score_QK_qkvtype = T.alloc_fragment((block_M, block_N), dtypeQKV)
        score_scale = T.alloc_fragment((block_M), dtypeAccu)
        local_rowmax = T.alloc_fragment((block_M), dtypeAccu)
        prev_rowmax = T.alloc_fragment((block_M), dtypeAccu)
        global_l = T.alloc_fragment((block_M), dtypeAccu)
        block_output = T.alloc_fragment((block_M, dim), dtypeOut)

        T.copy(Q[by, bx * block_M:(bx + 1) * block_M, bz, :], Q_shared)
        T.copy(Q_shared, Q_local)
        for i, j in T.Parallel(block_M, dim):
            Q_local[i, j] *= scale
        T.fill(block_output, 0)
        T.fill(global_l, 0)
        T.fill(local_rowmax, -T.infinity(dtypeAccu))

        for k in T.Pipelined(
                T.ceildiv((bx + 1) *
                            block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N),
                num_stages=num_stages):
            T.copy(K[by, k * block_N:(k + 1) * block_N, bz, :], K_shared)
            T.copy(V[by, k * block_N:(k + 1) * block_N, bz, :], V_shared)
            if is_causal:
                for i, j in T.Parallel(block_M, block_N):
                    score_QK[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0,
                                                    -T.infinity(dtypeAccu))
            else:
                T.fill(score_QK, 0)
            T.gemm(
                Q_local,
                K_shared,
                score_QK,
                policy=T.GemmWarpPolicy.FullRow,
            )
            T.copy(local_rowmax, prev_rowmax)
            T.reduce_max(score_QK, local_rowmax, dim=1, clear=False)
            for i, j in T.Parallel(block_M, block_N):
                score_QK[i, j] = T.exp2(score_QK[i, j] - local_rowmax[i])
            for i in T.Parallel(block_M):
                score_scale[i] = T.exp2(prev_rowmax[i] - local_rowmax[i])
            for i, j in T.Parallel(block_M, dim):
                block_output[i, j] *= score_scale[i]
            T.copy(score_QK, score_QK_qkvtype)
            T.gemm(
                score_QK_qkvtype,
                V_shared,
                block_output,
                policy=T.GemmWarpPolicy.FullRow,
            )
            T.reduce_sum(score_QK, score_QK_sum, dim=1)
            for i in T.Parallel(block_M):
                global_l[i] = global_l[i] * score_scale[i] + score_QK_sum[i]
        for i, j in T.Parallel(block_M, dim):
            block_output[i, j] /= global_l[i]
        T.copy(block_output, Output[by, bx * block_M:(bx + 1) * block_M, bz, :])
"""

formatter = SvgFormatter(
    style=CustomStyle,
    full=True,
    lineanchors="line",
    linenos=True,
    # linenostyle=(
    #     "font-size:12px; font-family:monospace; fill:#ffffff; "
    #     "background-color:#343a40; padding:5px 10px; border-radius:4px; "
    #     "text-align:center; margin-right:5px;"
    # ),
)


svg_code = highlight(code, CustomPythonLexer(), formatter)

import os

file_path = os.path.abspath(__file__)

svg_path = file_path.replace(".py", ".svg")
with open(svg_path, "w", encoding="utf-8") as f:
    f.write(svg_code)

print(f"SVG file saved as {svg_path}")

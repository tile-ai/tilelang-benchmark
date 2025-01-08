from pygments.lexers import PythonLexer
from pygments import highlight
from pygments.formatters import SvgFormatter
from pygments.token import Token
from pygments.styles.default import DefaultStyle

class CustomPythonLexer(PythonLexer):

    EXTRA_KEYWORDS = {"T", "Parallel", "get_thread_env", "vectorized", "Pipelined", "alloc_shared", "alloc_fragment", "clear", "copy", "gemm", "ceildiv", "Kernel"}
    VARIABLES = {
        "A",
        "B",
        "C",
        "A_shared",
        "B_shared",
        "C_local",
    }
    NEXT_LINE_ARROWS = {
        "→",
    }

    def get_tokens_unprocessed(self, text):
        for index, token, value in super().get_tokens_unprocessed(text):
            if value in self.EXTRA_KEYWORDS:
                yield index, Token.Keyword.TileLangSyntax, value  # 自定义 Token
            elif value in self.VARIABLES:
                yield index, Token.Name.Variable, value
            elif value in self.NEXT_LINE_ARROWS:
                yield index, Token.Keyword.NextLineArrow, value
            else:
                yield index, token, value


class CustomStyle(DefaultStyle):  # 继承 DefaultStyle 或其他主题样式
    # 在原有 styles 的基础上扩展
    styles = dict(DefaultStyle.styles)  # 保留 DefaultStyle 的所有样式

    # 添加自定义样式
    styles[Token.Keyword.TileLangSyntax] = "bold #9467bd"
    styles[Token.Keyword.NextLineArrow] = "bold #808080" # grey color for next line symbol "bold #808080"
    styles[Token.Keyword] = "bold #FF0000"  # 红色粗体
    styles[Token.Operator.Word] = "bold #1f77b4"
    styles[Token.Name.Function] = "bold #0000FF"
    # styles[Token.Name.Class] = "bold #9467bd"
    styles[Token.Name.Namespace] = "bold #9467bd"
    styles[Token.Name.Variable] = "bold #d62728"
    styles[Token.Literal.Number] = "bold #1f77b4"
    styles[Token.Operator] = "#1f77b4"


code = """
tid = T.get_thread_env(32, "threadIdx.x")
for v in T.vectorized(8):
    A_shared[tid // 4, (((tid % 4 * 8 + v % 
    → 8) // 8) ^ ((tid // 4) % 8 // 2)) * 8 +
    → (tid % 4 * 8 + v % 8) % 8] = A[tid // 4,
    → tid % 4 * 8 + v % 8]
"""

formatter = SvgFormatter(
    style=CustomStyle,
    full=True,
    lineanchors="line",
    # linenos=True,
)


svg_code = highlight(code, CustomPythonLexer(), formatter)
svg_code = svg_code.replace(
    "<style>",
    "<style>\ntext { white-space: pre-wrap; max-width: 80ch; overflow-wrap: break-word; }"
)

import os
file_path = os.path.abspath(__file__)

svg_path = file_path.replace(".py", ".svg")
with open(svg_path, "w", encoding="utf-8") as f:
    f.write(svg_code)

print(f"SVG file saved as {svg_path}")

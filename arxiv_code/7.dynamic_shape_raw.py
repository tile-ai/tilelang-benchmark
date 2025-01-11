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
        "Pipelined",
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
code = """
for i in T.unroll(2):
    for vec in range(8):
        A_shared[i * 1024 + (v >> 2) * 32 + (((((v & 31) >> 4) + ((v & 3) >> 1)) & 1) * 16) + ((((v & 15) >> 3) + (v & 1)) & 1) * 8 + vec] = T.if_then_else(
            by * 64 + i * 32 + (v >> 2) < m and (v & 3) * 8 < k,
            A_1[(v & 3) * 8 + (by * 64 + i * 32 + (v >> 2)) * k + vec],
            T.float16(0)
        )
"""

formatter = SvgFormatter(
    style=CustomStyle,
    full=True,
    lineanchors="line",
    # linenos=True,
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

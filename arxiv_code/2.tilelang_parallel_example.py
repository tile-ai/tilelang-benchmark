from pygments.lexers import PythonLexer
from pygments import highlight
from pygments.formatters import SvgFormatter
from pygments.token import Token
from pygments.styles.default import DefaultStyle

class CustomPythonLexer(PythonLexer):

    EXTRA_KEYWORDS = {"T", "Parallel", "Pipelined", "alloc_shared", "alloc_fragment", "clear", "copy", "gemm", "ceildiv", "Kernel"}
    VARIABLES = {
        "A",
        "B",
        "C",
        "A_shared",
        "B_shared",
        "C_local",
    }

    def get_tokens_unprocessed(self, text):
        for index, token, value in super().get_tokens_unprocessed(text):
            if value in self.EXTRA_KEYWORDS:
                yield index, Token.Keyword.TileLangSyntax, value  # 自定义 Token
            elif value in self.VARIABLES:
                yield index, Token.Name.Variable, value
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


code = """
for i, k in T.Parallel(8, 32):
    A_shared[i, k] = A[by * BM + i, ko * BK + k]
"""

formatter = SvgFormatter(
    style=CustomStyle,
    full=True,
    lineanchors="line",
    # linenos=True,
)


svg_code = highlight(code, CustomPythonLexer(), formatter)

import os
file_path = os.path.abspath(__file__)

svg_path = file_path.replace(".py", ".svg")
with open(svg_path, "w", encoding="utf-8") as f:
    f.write(svg_code)

print(f"SVG file saved as {svg_path}")

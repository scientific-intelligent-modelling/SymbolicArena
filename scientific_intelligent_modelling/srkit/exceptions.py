"""srkit 运行期异常类型。"""


class NoValidOutputError(RuntimeError):
    """算法正常结束但没有产生可评估符号表达式。"""


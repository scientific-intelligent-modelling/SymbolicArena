"""科学智能建模工具箱(SIM)"""

__version__ = "1.0.0"

# 在__init__.py中修改工具导入
class LazyToolLoader:
    """延迟加载工具，直到实际使用时才初始化"""
    
    def __init__(self, tool_name, cuda_version=None):
        self.tool_name = tool_name
        self.cuda_version = cuda_version
        self._tool = None
    
    def __getattr__(self, name):
        if self._tool is None:
            from .enhanced_subprocess import ToolProxy
            self._tool = ToolProxy(self.tool_name)
        return getattr(self._tool, name)

# 使用延迟加载
sklearn_tool = LazyToolLoader('sklearn_tool')
torch_tool = LazyToolLoader('torch_1_8_tool')

# 导出工具
__all__ = ['sklearn_tool', 'torch_tool']

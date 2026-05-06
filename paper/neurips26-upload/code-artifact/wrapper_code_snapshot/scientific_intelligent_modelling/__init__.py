"""Scientific Intelligent Modelling toolkit(SIM)"""

__version__ = "1.0.0"

# note__init__.pynote
class LazyToolLoader:
    """note note"""
    
    def __init__(self, tool_name, cuda_version=None):
        self.tool_name = tool_name
        self.cuda_version = cuda_version
        self._tool = None
    
    def __getattr__(self, name):
        if self._tool is None:
            from .enhanced_subprocess import ToolProxy
            self._tool = ToolProxy(self.tool_name)
        return getattr(self._tool, name)

# note
sklearn_tool = LazyToolLoader('sklearn_tool')
torch_tool = LazyToolLoader('torch_1_8_tool')

# note
__all__ = ['sklearn_tool', 'torch_tool']

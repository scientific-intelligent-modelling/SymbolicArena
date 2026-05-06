"""配置管理器，负责加载和访问配置"""

import os
import json
from pathlib import Path

class ConfigManager:
    """管理工具箱的配置"""
    
    def __init__(self, config_dir=None):
        if config_dir is None:
            # 如果未指定配置目录，使用包内的config目录
            self.config_dir = Path(__file__).parent.parent / "config"
        else:
            self.config_dir = Path(config_dir)
        
        self.configs = {}
        self._load_configs()
    
    def _load_configs(self):
        """加载所有配置文件"""
        for config_file in self.config_dir.glob("*.json"):
            config_name = config_file.stem
            with open(config_file, 'r') as f:
                self.configs[config_name] = json.load(f)
    
    def get_config(self, config_name):
        """获取指定配置"""
        return self.configs.get(config_name, {})

    def get_env_name_by_tool(self, tool_name):
        """获取工具对应的conda环境名称"""
        toolbox_config = self.get_config("toolbox_config")
        tool_mapping = toolbox_config.get("tool_mapping", {})
        
        if tool_name in tool_mapping:
            return tool_mapping[tool_name].get("env")
        return None
    
    def get_env_config(self, env_name):
        """获取指定conda环境的配置"""
        envs_config = self.get_config("envs_config")
        env_list = envs_config.get("env_list", {})
        
        if env_name in env_list:
            return env_list.get(env_name)
        return None

# 创建全局配置管理器实例
config_manager = ConfigManager()

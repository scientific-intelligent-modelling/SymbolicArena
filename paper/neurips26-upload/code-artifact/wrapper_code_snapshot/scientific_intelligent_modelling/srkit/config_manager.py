"""note note"""

import os
import json
from pathlib import Path

class ConfigManager:
    """notetoolboxnote"""
    
    def __init__(self, config_dir=None):
        if config_dir is None:
            # note noteconfignote
            self.config_dir = Path(__file__).parent.parent / "config"
        else:
            self.config_dir = Path(config_dir)
        
        self.configs = {}
        self._load_configs()
    
    def _load_configs(self):
        """note"""
        for config_file in self.config_dir.glob("*.json"):
            config_name = config_file.stem
            with open(config_file, 'r') as f:
                self.configs[config_name] = json.load(f)
    
    def get_config(self, config_name):
        """note"""
        return self.configs.get(config_name, {})

    def get_env_name_by_tool(self, tool_name):
        """notecondaenvironment note"""
        toolbox_config = self.get_config("toolbox_config")
        tool_mapping = toolbox_config.get("tool_mapping", {})
        
        if tool_name in tool_mapping:
            return tool_mapping[tool_name].get("env")
        return None
    
    def get_env_config(self, env_name):
        """notecondaenvironment note"""
        envs_config = self.get_config("envs_config")
        env_list = envs_config.get("env_list", {})
        
        if env_name in env_list:
            return env_list.get(env_name)
        return None

# note
config_manager = ConfigManager()

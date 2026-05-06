"""Conda环境管理器，负责创建和删除环境"""

import subprocess
import os
import sys
import json
import logging
import shutil
from pathlib import Path
from .config_manager import config_manager


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("conda_env_manager")

class EnvManager:
    """管理Conda环境的创建和删除"""
    
    def __init__(self):
        self.config_manager = config_manager
        self.env_config = self.config_manager.get_config("envs_config")
        self.conda_base_path = self._get_conda_base_path()

    def _get_conda_base_path(self):
        """获取conda安装路径"""
        try:
            # 使用conda info命令获取conda的安装信息
            returncode, stdout, stderr = self.run_command(
                ["conda", "info", "--json"], show_output=False
            )
            import json
            conda_info = json.loads(stdout)
            # ANSI 转义序列
            GREEN = "\033[92m"  # 绿色
            RESET = "\033[0m"  # 重置颜色
            print(f"{GREEN}conda_prefix: {conda_info['conda_prefix']}{RESET}")
            return conda_info['conda_prefix']
        except Exception as e:
            # 在某些受限环境（如系统权限限制导致conda插件不可运行）下，
            # 使用已知路径兜底，避免工具链完全不可用
            fallback = self._get_conda_base_path_fallback()
            if fallback:
                print(f"\033[93m警告: conda info获取失败，使用兜底路径 {fallback} 继续。原因: {e}\033[0m")
                return fallback
            logger.error(f"获取conda路径失败: {e}")
            return None

    def _get_conda_base_path_fallback(self):
        """在conda info不可用时的兼容路径探测"""
        # 1) 常见环境变量
        env_candidates = [
            os.environ.get("CONDA_PREFIX"),
            os.environ.get("CONDA_ROOT"),
            os.environ.get("CONDA_HOME"),
            os.environ.get("MAMBA_ROOT_PREFIX"),
        ]
        for candidate in env_candidates:
            if candidate:
                base = Path(candidate).expanduser().resolve()
                if base.joinpath("bin", "conda").exists():
                    return str(base)
                if base.parent.name == "envs":
                    return str(base.parent.parent)

        # 2) 从conda可执行文件反推
        conda_exe = shutil.which("conda")
        if conda_exe:
            conda_exe = Path(conda_exe).resolve()
            base = conda_exe.parent.parent
            if base.joinpath("bin", "activate").exists() or base.joinpath("bin", "conda").exists():
                return str(base)
        return None
        
    def check_conda_installed(self):
        """检查是否安装Conda并可在PATH中使用"""
        try:
            returncode, stdout, stderr = self.run_command(["conda", "--version"])
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            print("错误: Conda未安装或不在PATH中。")
            return False
    
    def list_environments(self):
        """打印可用环境列表"""
        env_list = self.env_config.get("env_list", {})
        if not env_list:
            print("配置中未找到环境。")
            return
        
        print("\n**可用环境:**")
        for i, (env_name, env_details) in enumerate(env_list.items(), 1):
            python_version = env_details.get("python_version", "未指定")
            comments = env_details.get("comments", "")
            
            print(f"{i}. {env_name} (Python {python_version})")
            if comments:
                print(f"   备注: {comments}")
            
            # # 打印包信息
            # packages = env_details.get("packages", [])
            # if packages:
            #     print(f"   包: {', '.join(packages)}")
            
            # # 打印渠道信息
            # channels = env_details.get("channels", [])
            # if channels:
            #     print(f"   渠道: {', '.join(channels)}")
                
            # 打印安装后命令
            # post_commands = env_details.get("post_install_commands", [])
            # if post_commands:
            #     print(f"   安装后命令:")
            #     for cmd in post_commands:
            #         print(f"     - {cmd}")
            
            print()  # 空行提高可读性
    
    def get_env_path(self, env_name):
        """获取Conda环境的路径"""
        try:
            returncode, stdout, stderr = self.run_command(
                ["conda", "info", "--envs", "--json"], show_output=False
            )
            env_data = json.loads(stdout)
            envs = env_data.get("envs", [])
            # ANSI 转义序列
            GREEN = "\033[92m"  # 绿色
            RESET = "\033[0m"  # 重置颜色
            print(f"{GREEN}envs: {envs}{RESET}")
            # 寻找指定名称的环境
            for env_path in envs:
                if os.path.basename(env_path) == env_name:
                    return env_path
                
            return None
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"获取环境路径时出错: {e}")
            return None
    def get_env_python(self, conda_env_name):
        """获取指定环境的Python可执行文件路径"""
        if not self.conda_base_path:
            return None
        
        if os.name == 'nt':  # Windows
            python_path = os.path.join(self.conda_base_path, "envs", conda_env_name, "python.exe")
        else:  # Linux/MacOS
            python_path = os.path.join(self.conda_base_path, "envs", conda_env_name, "bin", "python")
        
        if os.path.exists(python_path):
            return python_path
        return None

    def get_post_commands_marker_path(self, env_name):
        """获取记录已执行后处理命令的标记文件路径"""
        env_path = self.get_env_path(env_name)
        if not env_path:
            return None
        return os.path.join(env_path, ".post_commands_executed")
    
    def record_post_command_execution(self, env_name, command):
        """记录后处理命令已成功执行"""
        marker_path = self.get_post_commands_marker_path(env_name)
        if not marker_path:
            return False
            
        executed_commands = set()
        
        # 如果标记文件存在，读取已执行的命令
        if os.path.exists(marker_path):
            with open(marker_path, 'r') as f:
                executed_commands = set(line.strip() for line in f)
        
        # 添加新执行的命令
        executed_commands.add(command)
        
        # 写入更新后的命令列表
        with open(marker_path, 'w') as f:
            for cmd in executed_commands:
                f.write(f"{cmd}\n")
        
        return True
    
    def get_executed_post_commands(self, env_name):
        """获取已执行的后处理命令列表"""
        marker_path = self.get_post_commands_marker_path(env_name)
        if not marker_path or not os.path.exists(marker_path):
            return set()
            
        with open(marker_path, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    
    def check_environment(self, env_name):
        """
        检查指定的环境是否存在并满足条件。
        
        返回:
            - (True, None): 环境存在且满足所有条件
            - (False, "reason"): 环境不存在或不满足条件，reason 是原因说明
        """
        # 获取环境配置
        env_config = self.config_manager.get_env_config(env_name)
        if not env_config:
            return False, f"配置中未找到环境 '{env_name}'"
        
        # 检查环境是否存在
        existing_envs = self.get_existing_environments()
        if env_name not in existing_envs:
            return False, f"环境 '{env_name}' 不存在"
        
        # 检查环境中是否安装了所需的conda包
        conda_packages = env_config.get("conda_packages", [])
        if conda_packages:
            try:
                returncode, stdout, stderr = self.run_command(
                    ["conda", "list", "--name", env_name]
                )
                
                installed_packages = []
                for line in stdout.splitlines():
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            pkg_name = parts[0]
                            installed_packages.append(pkg_name)
                
                # 检查所需的conda包是否都已安装
                missing_packages = []
                for package in conda_packages:
                    # 处理包名可能包含版本号的情况
                    package_name = package.split('=')[0] if '=' in package else package
                    if package_name not in installed_packages:
                        missing_packages.append(package_name)
                
                if missing_packages:
                    return False, f"环境 '{env_name}' 缺少必需的conda包: {', '.join(missing_packages)}"
                        
            except subprocess.CalledProcessError as e:
                return False, f"检查环境 '{env_name}' 的conda包时出错: {e}"
        
        # 检查环境中是否安装了所需的pip包
        pip_packages = env_config.get("pip_packages", [])
        if pip_packages:
            try:
                # 使用 run_in_conda_env 替代 conda run
                # 使用 freeze 格式简化输出，pip freeze 不支持 --no-cache
                returncode, stdout, stderr = self.run_in_conda_env(
                    env_name=env_name,
                    command=["pip", "freeze"],
                    show_output=False
                )

                # 合并 stdout 和 stderr 以处理 pip 可能将警告等信息发送到 stderr 的情况
                full_output = stdout + stderr
                
                installed_pip_packages = []
                # 解析 'package==version' 格式的输出
                for line in full_output.splitlines():
                    if '==' in line:
                        pkg_name = line.split('==')[0].lower()
                        installed_pip_packages.append(pkg_name)

                # 检查所需的pip包是否都已安装
                missing_pip_packages = []
                for package in pip_packages:
                    # 处理包名可能包含版本号的情况
                    package_name = package.split('=')[0] if '=' in package else package
                    package_name = package_name.lower()  # 转换为小写以进行比较
                    if package_name not in installed_pip_packages:
                        missing_pip_packages.append(package_name)
                
                if missing_pip_packages:
                    return False, f"环境 '{env_name}' 缺少必需的pip包: {', '.join(missing_pip_packages)}"
                        
            except subprocess.CalledProcessError as e:
                return False, f"检查环境 '{env_name}' 的pip包时出错: {e}"
        
        # 检查Python版本是否匹配
        required_python = env_config.get("python_version")
        if required_python:
            try:
                # 使用 run_in_conda_env 替代 conda run
                returncode, stdout, stderr = self.run_in_conda_env(
                    env_name=env_name,
                    command=["python", "--version"],
                    show_output=False
                )
                python_version = stdout.strip()
                # Python版本输出通常是"Python X.Y.Z"格式
                if required_python not in python_version:
                    return False, f"环境 '{env_name}' Python版本不匹配，需要 {required_python}，实际是 {python_version}"
            except subprocess.CalledProcessError as e:
                return False, f"检查环境 '{env_name}' 的Python版本时出错: {e}"
        
        # 检查后处理命令是否已执行
        post_commands = env_config.get("post_install_commands", [])
        if post_commands:
            executed_commands = self.get_executed_post_commands(env_name)
            missing_commands = [cmd for cmd in post_commands if cmd not in executed_commands]
            
            if missing_commands:
                return False, f"环境 '{env_name}' 缺少必需的后处理命令: {', '.join(missing_commands)}"
        
        return True, None

    def check_all_environments(self):
        """
        检查所有配置的环境的状态。
        每检查一个环境就输出一个，并且在全部检查完后进行汇总。
        
        返回:
            - configured_envs: 已经配置好的环境列表
            - unconfigured_envs: 未配置好的环境列表，每个元素是 (env_name, reason) 的元组
        """
        env_list = self.env_config.get("env_list", {})
        configured_envs = []
        unconfigured_envs = []
        
        print("开始检查所有环境...")
        for env_name in env_list:
            exists, reason = self.check_environment(env_name)
            if exists:
                configured_envs.append(env_name)
                print(f"环境 '{env_name}' 配置正确。")
            else:
                unconfigured_envs.append((env_name, reason))
                print(f"环境 '{env_name}' 配置失败: {reason}")
        
        # 汇总结果
        print("\n检查完成，汇总结果:")
        print(f"配置正确的环境: {len(configured_envs)} 个")
        if configured_envs:
            print("  - " + "\n  - ".join(configured_envs))
        
        print(f"配置失败的环境: {len(unconfigured_envs)} 个")
        if unconfigured_envs:
            for env_name, reason in unconfigured_envs:
                print(f"  - {env_name}: {reason}")
        
        return configured_envs, unconfigured_envs


    def create_environment(self, env_name):
        """根据配置创建Conda环境，如果环境已存在且满足条件则跳过创建"""
        # 先检查环境是否已存在且满足条件
        exists, reason = self.check_environment(env_name)
        if exists:
            print(f"环境 '{env_name}' 已存在且满足所有条件，跳过创建。")
            return True
        
        # 如果环境存在但不满足条件，可以选择先删除再创建，或者提醒用户
        if reason and "不存在" not in reason:
            print(f"警告: {reason}")
            choice = input(f"环境 '{env_name}' 存在但不满足要求，是否重新创建? (y/n): ").strip().lower()
            if choice != 'y':
                print(f"跳过创建环境 '{env_name}'")
                return False
            
            # 先删除现有环境
            self.delete_environment(env_name)
        
        # 以下是创建环境的逻辑
        env_details = self.config_manager.get_env_config(env_name)
        if not env_details:
            print(f"错误: 配置中未找到环境'{env_name}'。")
            return False

        python_version = env_details.get("python_version", "3.10")
        conda_packages = env_details.get("conda_packages", [])
        pip_packages = env_details.get("pip_packages", [])
        channels = env_details.get("channels", [])
        post_commands = env_details.get("post_install_commands", [])

        # 构建conda create命令
        cmd = ["conda", "create", "-y", "-n", env_name, f"python={python_version}"]

        # 添加指定的conda包
        cmd.extend(conda_packages)

        # 添加指定的渠道
        for channel in channels:
            cmd.extend(["-c", channel])
        
        print(f"创建环境'{env_name}'...")
        try:
            # 执行conda create命令
            returncode, stdout, stderr = self.run_command(cmd)
            
            # 安装pip包
            if pip_packages:
                print(f"为'{env_name}'安装pip包...")
                all_pip_packages_installed = True
                
                for package in pip_packages:
                    print(f"正在安装pip包: {package}...")

                    cmd = ["pip", "install", package]
                    # pip_cmd = ["conda", "run", "-n", env_name, 
                    
                    try:
                        # 使用直接输出到终端的方式运行命令
                        returncode, stdout, stderr = self.run_in_conda_env(env_name=env_name, command=cmd,show_output=True)
                        print(f"pip包 {package} 安装完成。")
                    except subprocess.CalledProcessError as e:
                        print(f"安装pip包 {package} 时出错: {e}")
                        all_pip_packages_installed = False
                        
                        # 可以选择是否继续安装其他包
                        if input(f"安装 {package} 失败。是否继续安装其他包? (y/n): ").lower() != 'y':
                            return False
                
                if not all_pip_packages_installed:
                    print("警告: 某些pip包安装失败")
                    if input("是否继续环境创建过程? (y/n): ").lower() != 'y':
                        return False
                else:
                    print("所有pip包安装完成。")

            
            # 执行安装后命令
            if post_commands:
                print(f"为'{env_name}'执行安装后命令...")
                all_commands_succeeded = True
                
                for command in post_commands:
                    print(f"\n==== 开始执行后处理命令: {command} ====")
                    try:
                        # 将命令按照引号和空格正确拆分
                        import shlex
                        command_parts = shlex.split(command)
                        
                        # 使用 run_in_conda_env 替代 conda run
                        print(f"在环境 '{env_name}' 中执行命令: {' '.join(command_parts)}")
                        
                        # 执行命令，直接将输出显示到终端
                        returncode, stdout, stderr = self.run_in_conda_env(
                            env_name=env_name,
                            command=command_parts,
                            show_output=True
                        )
                        
                        print(f"==== 后处理命令执行成功 ====")
                        
                        # 记录成功执行的命令
                        self.record_post_command_execution(env_name, command)
                        
                    except subprocess.CalledProcessError as e:
                        all_commands_succeeded = False
                        print(f"==== 后处理命令执行失败 ====")
                        print(f"错误信息: {e}")
                        
                        # 可以选择是否继续执行剩余命令
                        if input("命令执行失败。是否继续执行下一个命令? (y/n): ").lower() != 'y':
                            break
                
                if all_commands_succeeded:
                    print(f"\n所有后处理命令执行成功！")
                else:
                    print(f"\n警告: 某些后处理命令执行失败")
            return True
        except subprocess.CalledProcessError as e:
            print(f"创建环境'{env_name}'时出错: {e}")
            return False

    
    def delete_environment(self, env_name):
        """删除Conda环境"""
        # 删除环境前先记录标记文件路径，以便我们可以在删除环境后删除该文件
        marker_path = self.get_post_commands_marker_path(env_name)
        
        print(f"删除环境'{env_name}'...")
        try:
            returncode, stdout, stderr = self.run_command(["conda", "env", "remove", "-y", "-n", env_name])
            print(f"环境'{env_name}'删除成功!")
            
            # 尝试删除标记文件（如果环境已被删除，此操作可能会失败，但这没关系）
            if marker_path and os.path.exists(marker_path):
                try:
                    os.remove(marker_path)
                except (OSError, IOError):
                    pass
                    
            return True
        except subprocess.CalledProcessError as e:
            print(f"删除环境'{env_name}'时出错: {e}")
            return False
    
    def get_existing_environments(self):
        """获取现有Conda环境列表"""
        returncode, stdout, stderr = self.run_command(["conda", "env", "list"])
        existing_envs = []
        for line in stdout.splitlines():
            if line and not line.startswith('#'):
                env_name = line.split()[0]
                if env_name != "base":  # 排除base环境
                    existing_envs.append(env_name)
        return existing_envs
    
    def run_in_conda_env(self, env_name, command, show_output=True):
        """
        在指定的conda环境中执行命令，实时捕获并显示输出
        
        参数:
            env_name: conda环境名称
            command: 要执行的命令（列表形式或字符串）
            show_output: 是否实时显示输出
            
        返回:
            (returncode, stdout, stderr) 元组
        """
        # if command[0] == "pip": 
        #     command.insert(2, "--progress-bar=on")
        # 确保command是列表形式
        if isinstance(command, str):
            import shlex
            command = shlex.split(command)
        
        # 获取conda的路径
        conda_path = self.conda_base_path
        if not conda_path:
            print("错误: 无法获取conda安装路径")
            return 1, "", "无法获取conda安装路径"
            
        # 构建激活环境的命令
        if os.name == 'nt':  # Windows
            # Windows下使用不同的激活方式
            activate_cmd = f"conda activate {env_name} && "
            shell = True
            full_command = activate_cmd + " ".join(command)
        else:  # Linux/MacOS
            # 使用source激活环境
            activate_path = os.path.join(conda_path, "bin", "activate")
            activate_cmd = f"source {activate_path} {env_name} && "
            shell = True
            full_command = activate_cmd + " ".join(command)
            
        # ANSI 转义序列
        CYAN = "\033[96m"  # 青色
        RESET = "\033[0m"  # 重置颜色
        
        print(f"{CYAN}执行命令: {full_command}{RESET}")

        # 启动子进程，使用shell模式执行组合命令
        process = subprocess.Popen(
            full_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
            # bufsize=0,  # 移除：让 Popen 在 text=True 时处理缓冲
            executable='/bin/bash' if os.name != 'nt' else None  # 在Linux/Mac上指定bash
        )

        # 如果不需要实时显示，使用 communicate() 来可靠地获取所有输出
        if not show_output:
            stdout, stderr = process.communicate()
            return process.returncode, stdout, stderr

        # 以下是需要实时显示输出的情况
        # 存储所有输出
        stdout_all = []
        stderr_all = []

        # 使用 select 读取输出
        import select
        import sys # 用于打印错误

        # 实时读取和处理输出
        GREEN = "\033[92m"  # 绿色
        RED = "\033[91m"    # 红色
        RESET = "\033[0m"  # 重置颜色
        CYAN = "\033[96m"  # 青色

        error_keywords = ["error", "failed", "not found", "exception", "fatal"]

        while process.poll() is None:  # 进程仍在运行
            # 确保管道存在
            read_pipes = []
            if process.stdout:
                read_pipes.append(process.stdout)
            if process.stderr:
                read_pipes.append(process.stderr)

            if not read_pipes: # 如果 Popen 成功，这不应该发生
                break

            # 等待管道可读
            ready_pipes, _, _ = select.select(read_pipes, [], [], 0.1)

            for pipe in ready_pipes:
                try:
                    line = pipe.readline()
                    # readline() 在 EOF 时应返回空字符串 ''
                    # 如果返回 None，则显式处理（尽管不常见）
                    if line is None:
                        continue
                    if not line: # 处理 EOF (空字符串)
                        continue

                    if pipe == process.stdout:
                        stdout_all.append(line)
                        if show_output:
                            print(f"{GREEN}{line}{RESET}", end='')
                    else:  # pipe == process.stderr
                        stderr_all.append(line)
                        if show_output:
                            is_error = any(keyword in line.lower() for keyword in error_keywords)
                            color = RED if is_error else CYAN
                            print(f"{color}{line}{RESET}", end='')
                except (IOError, OSError) as e:
                    # 记录或打印错误以进行调试
                    print(f"读取管道时出错: {e}", file=sys.stderr)
                    # 可以选择忽略、中断或以其他方式处理
                    pass
                except Exception as e: # 捕获其他潜在异常
                    print(f"处理管道输出时发生意外错误: {e}", file=sys.stderr)
                    pass


        # 捕获剩余的输出
        # 注意：在上面的循环之后，communicate() 可能会阻塞或返回空，
        # 因为大部分或所有输出应该已经被 readline 读取了。
        # 但为了确保捕获所有内容，仍然调用它。
        stdout_rem, stderr_rem = process.communicate()
        if stdout_rem:
            stdout_all.append(stdout_rem)
            if show_output:
                print(f"{GREEN}{stdout_rem}{RESET}", end='')
        if stderr_rem:
            stderr_all.append(stderr_rem)
            if show_output:
                # 在这里也应用颜色逻辑
                for line in stderr_rem.splitlines():
                    is_error = any(keyword in line.lower() for keyword in error_keywords)
                    color = RED if is_error else CYAN
                    print(f"{color}{line}{RESET}")


        return process.returncode, ''.join(stdout_all), ''.join(stderr_all)
    
    def run_command(self, command, show_output=True):
        """
        运行shell命令并实时捕获输出
        
        参数:
            command: 要执行的命令（列表形式）
            show_output: 是否实时显示输出
            
        返回:
            (returncode, stdout, stderr) 元组
        """
        # ANSI 转义序列
        GRAY = "\033[90m"   # 亮灰色
        RESET = "\033[0m"   # 重置颜色
        RED = "\033[91m"    # 红色
        LIGHT_GREEN = "\033[92m"  # 亮绿色
        GREEN = "\033[32m"  # 绿色
        YELLOW = "\033[93m" # 黄色
        BLUE = "\033[94m"   # 蓝色
        MAGENTA = "\033[95m" # 洋红色
        CYAN = "\033[96m"   # 青色
        WHITE = "\033[97m"  # 白色
        
        # 启动子进程
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,      # 将输出作为文本处理
            bufsize=0       # 关键参数：设置为0禁用缓冲
        )  

        # 存储所有输出
        stdout_all = []
        stderr_all = []

        if show_output:
            # 实时读取输出
            for line in process.stdout:
                stdout_all.append(line)  # 收集标准输出
                print(f"{CYAN}{line}{RESET}", end='')  # 打印标准输出，颜色为青色

        # 等待进程结束
        stdout, stderr = process.communicate()  # 捕获剩余输出
        if stdout:
            stdout_all.append(stdout)  # 收集最终的标准输出
            if show_output:
                print(f"{CYAN}{stdout}{RESET}", end='')

        # 定义错误关键词
        error_keywords = ["error", "failed", "not found", "exception", "fatal"]  # 可自定义此列表

        if stderr:
            stderr_all.append(stderr)  # 收集标准错误
            if show_output:
                for line in stderr.splitlines():  # 将stderr拆分为行
                    is_error = any(keyword in line.lower() for keyword in error_keywords)  # 检查错误关键词
                    if is_error:
                        print(f"{RED}{line}{RESET}")  # 用红色打印错误
                    else:
                        print(f"{BLUE}{line}{RESET}")  # 用蓝色打印信息性消息

        return process.returncode, ''.join(stdout_all), ''.join(stderr_all)  # 返回返回码及所有输出
    
    def run_cli(self):
        """运行命令行界面"""
        if not self.check_conda_installed():
            print("请安装Conda或确保它已正确配置在您的PATH中。")
            return
        
        env_list = self.env_config.get("env_list", {})
        
        if not env_list:
            print("配置中未找到环境。退出。")
            return
        
        while True:
            print("\n" + "="*50)
            print("**Conda环境管理器**")
            print("="*50)
            print("1. 创建环境")
            print("2. 删除环境")
            print("3. 检查环境状态")
            print("4. 退出")
            
            choice = input("\n请输入您的选择 (1-4): ").strip()
            
            if choice == "1":
                self.list_environments()
                
                env_names = list(env_list.keys())
                env_input = input("\n请输入要创建的环境的编号或名称 (输入'all'创建所有环境): ").strip()
                
                if env_input.lower() == 'all':
                    print("\n创建所有环境...")
                    for env_name in env_names:
                        self.create_environment(env_name)
                else:
                    try:
                        # 检查输入是否为数字
                        env_idx = int(env_input) - 1
                        if 0 <= env_idx < len(env_names):
                            env_name = env_names[env_idx]
                            self.create_environment(env_name)
                        else:
                            print("无效的环境编号。")
                    except ValueError:
                        # 输入不是数字，尝试作为环境名称
                        if env_input in env_names:
                            self.create_environment(env_input)
                        else:
                            print(f"配置中未找到环境'{env_input}'。")
            
            elif choice == "2":
                existing_envs = self.get_existing_environments()
                
                if not existing_envs:
                    print("没有可删除的conda环境 (不包括base)。")
                    continue
                
                print("\n**现有conda环境:**")
                for i, env_name in enumerate(existing_envs, 1):
                    print(f"{i}. {env_name}")
                
                env_input = input("\n请输入要删除的环境的编号或名称 (输入'all'删除所有环境): ").strip()
                
                if env_input.lower() == 'all':
                    confirm = input("您确定要删除所有环境吗？此操作无法撤销! (y/n): ").strip().lower()
                    if confirm == 'y':
                        print("\n删除所有环境...")
                        for env_name in existing_envs:
                            self.delete_environment(env_name)
                    else:
                        print("操作已取消。")
                else:
                    try:
                        # 检查输入是否为数字
                        env_idx = int(env_input) - 1
                        if 0 <= env_idx < len(existing_envs):
                            env_name = existing_envs[env_idx]
                            self.delete_environment(env_name)
                        else:
                            print("无效的环境编号。")
                    except ValueError:
                        # 输入不是数字，尝试作为环境名称
                        if env_input in existing_envs:
                            self.delete_environment(env_input)
                        else:
                            print(f"未找到环境'{env_input}'。")
            
            elif choice == "3":
                configured_envs, unconfigured_envs = self.check_all_environments()
            
            elif choice == "4":
                print("退出Conda环境管理器。再见!")
                break
            
            else:
                print("无效选择。请输入1到4之间的数字。")


class _LazyEnvManager:
    """惰性创建的 EnvManager，避免导入时做昂贵的 Conda 探测"""
    def __init__(self):
        self._instance = None

    def _get(self):
        if self._instance is None:
            self._instance = EnvManager()
        return self._instance

    def __getattr__(self, name):
        return getattr(self._get(), name)

# 导出一个与原名兼容的惰性代理，保持外部引用不变
env_manager = _LazyEnvManager()

def main():
    """主函数"""
    # 导入ConfigManager
    env_manager.run_cli()

if __name__ == "__main__":
    main()

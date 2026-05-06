"""Condaenvironment note noteenvironment"""

import subprocess
import os
import sys
import json
import logging
import shutil
from pathlib import Path
from .config_manager import config_manager


# note
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("conda_env_manager")

class EnvManager:
    """noteCondaenvironment note"""
    
    def __init__(self):
        self.config_manager = config_manager
        self.env_config = self.config_manager.get_config("envs_config")
        self.conda_base_path = self._get_conda_base_path()

    def _get_conda_base_path(self):
        """notecondanote"""
        try:
            # noteconda infonotecondanote
            returncode, stdout, stderr = self.run_command(
                ["conda", "info", "--json"], show_output=False
            )
            import json
            conda_info = json.loads(stdout)
            # ANSI note
            GREEN = "\033[92m"  # note
            RESET = "\033[0m"  # note
            print(f"{GREEN}conda_prefix: {conda_info['conda_prefix']}{RESET}")
            return conda_info['conda_prefix']
        except Exception as e:
            # noteenvironment notecondanote note 
            # note note
            fallback = self._get_conda_base_path_fallback()
            if fallback:
                print(f"\033[93mnote: conda infonote note {fallback} note note: {e}\033[0m")
                return fallback
            logger.error(f"notecondanote: {e}")
            return None

    def _get_conda_base_path_fallback(self):
        """noteconda infonote"""
        # 1) noteenvironment note
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

        # 2) notecondanote
        conda_exe = shutil.which("conda")
        if conda_exe:
            conda_exe = Path(conda_exe).resolve()
            base = conda_exe.parent.parent
            if base.joinpath("bin", "activate").exists() or base.joinpath("bin", "conda").exists():
                return str(base)
        return None
        
    def check_conda_installed(self):
        """noteCondanotePATHnote"""
        try:
            returncode, stdout, stderr = self.run_command(["conda", "--version"])
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            print("note: Condanot installednotePATHnote ")
            return False
    
    def list_environments(self):
        """noteenvironment note"""
        env_list = self.env_config.get("env_list", {})
        if not env_list:
            print("noteenvironment ")
            return
        
        print("\n**noteenvironment:**")
        for i, (env_name, env_details) in enumerate(env_list.items(), 1):
            python_version = env_details.get("python_version", "note")
            comments = env_details.get("comments", "")
            
            print(f"{i}. {env_name} (Python {python_version})")
            if comments:
                print(f"   note: {comments}")
            
            # # note
            # packages = env_details.get("packages", [])
            # if packages:
            #     print(f"   note: {', '.join(packages)}")
            
            # # note
            # channels = env_details.get("channels", [])
            # if channels:
            #     print(f"   note: {', '.join(channels)}")
                
            # note
            # post_commands = env_details.get("post_install_commands", [])
            # if post_commands:
            #     print(f"   note:")
            #     for cmd in post_commands:
            #         print(f"     - {cmd}")
            
            print()  # note
    
    def get_env_path(self, env_name):
        """noteCondaenvironment note"""
        try:
            returncode, stdout, stderr = self.run_command(
                ["conda", "info", "--envs", "--json"], show_output=False
            )
            env_data = json.loads(stdout)
            envs = env_data.get("envs", [])
            # ANSI note
            GREEN = "\033[92m"  # note
            RESET = "\033[0m"  # note
            print(f"{GREEN}envs: {envs}{RESET}")
            # noteenvironment
            for env_path in envs:
                if os.path.basename(env_path) == env_name:
                    return env_path
                
            return None
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"noteenvironment note: {e}")
            return None
    def get_env_python(self, conda_env_name):
        """noteenvironment notePythonnote"""
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
        """note"""
        env_path = self.get_env_path(env_name)
        if not env_path:
            return None
        return os.path.join(env_path, ".post_commands_executed")
    
    def record_post_command_execution(self, env_name, command):
        """note"""
        marker_path = self.get_post_commands_marker_path(env_name)
        if not marker_path:
            return False
            
        executed_commands = set()
        
        # note note
        if os.path.exists(marker_path):
            with open(marker_path, 'r') as f:
                executed_commands = set(line.strip() for line in f)
        
        # note
        executed_commands.add(command)
        
        # note
        with open(marker_path, 'w') as f:
            for cmd in executed_commands:
                f.write(f"{cmd}\n")
        
        return True
    
    def get_executed_post_commands(self, env_name):
        """note"""
        marker_path = self.get_post_commands_marker_path(env_name)
        if not marker_path or not os.path.exists(marker_path):
            return set()
            
        with open(marker_path, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    
    def check_environment(self, env_name):
        """
        noteenvironment note 
        
        note:
            - (True, None): environment note
            - (False, "reason"): environment note reason note
        """
        # noteenvironment note
        env_config = self.config_manager.get_env_config(env_name)
        if not env_config:
            return False, f"noteenvironment '{env_name}'"
        
        # noteenvironment note
        existing_envs = self.get_existing_environments()
        if env_name not in existing_envs:
            return False, f"environment '{env_name}' note"
        
        # noteenvironment notecondanote
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
                
                # notecondanote
                missing_packages = []
                for package in conda_packages:
                    # note
                    package_name = package.split('=')[0] if '=' in package else package
                    if package_name not in installed_packages:
                        missing_packages.append(package_name)
                
                if missing_packages:
                    return False, f"environment '{env_name}' notecondanote: {', '.join(missing_packages)}"
                        
            except subprocess.CalledProcessError as e:
                return False, f"noteenvironment '{env_name}' notecondanote: {e}"
        
        # noteenvironment notepipnote
        pip_packages = env_config.get("pip_packages", [])
        if pip_packages:
            try:
                # note run_in_conda_env note conda run
                # note freeze note pip freeze unsupported --no-cache
                returncode, stdout, stderr = self.run_in_conda_env(
                    env_name=env_name,
                    command=["pip", "freeze"],
                    show_output=False
                )

                # note stdout note stderr note pip note stderr note
                full_output = stdout + stderr
                
                installed_pip_packages = []
                # note 'package==version' note
                for line in full_output.splitlines():
                    if '==' in line:
                        pkg_name = line.split('==')[0].lower()
                        installed_pip_packages.append(pkg_name)

                # notepipnote
                missing_pip_packages = []
                for package in pip_packages:
                    # note
                    package_name = package.split('=')[0] if '=' in package else package
                    package_name = package_name.lower()  # note
                    if package_name not in installed_pip_packages:
                        missing_pip_packages.append(package_name)
                
                if missing_pip_packages:
                    return False, f"environment '{env_name}' notepipnote: {', '.join(missing_pip_packages)}"
                        
            except subprocess.CalledProcessError as e:
                return False, f"noteenvironment '{env_name}' notepipnote: {e}"
        
        # notePythonnote
        required_python = env_config.get("python_version")
        if required_python:
            try:
                # note run_in_conda_env note conda run
                returncode, stdout, stderr = self.run_in_conda_env(
                    env_name=env_name,
                    command=["python", "--version"],
                    show_output=False
                )
                python_version = stdout.strip()
                # Pythonnote"Python X.Y.Z"note
                if required_python not in python_version:
                    return False, f"environment '{env_name}' Pythonnote note {required_python} note {python_version}"
            except subprocess.CalledProcessError as e:
                return False, f"noteenvironment '{env_name}' notePythonnote: {e}"
        
        # note
        post_commands = env_config.get("post_install_commands", [])
        if post_commands:
            executed_commands = self.get_executed_post_commands(env_name)
            missing_commands = [cmd for cmd in post_commands if cmd not in executed_commands]
            
            if missing_commands:
                return False, f"environment '{env_name}' note: {', '.join(missing_commands)}"
        
        return True, None

    def check_all_environments(self):
        """
        noteenvironment note 
        noteenvironment note note 
        
        note:
            - configured_envs: noteenvironment note
            - unconfigured_envs: noteenvironment note note (env_name, reason) note
        """
        env_list = self.env_config.get("env_list", {})
        configured_envs = []
        unconfigured_envs = []
        
        print("noteenvironment...")
        for env_name in env_list:
            exists, reason = self.check_environment(env_name)
            if exists:
                configured_envs.append(env_name)
                print(f"environment '{env_name}' note ")
            else:
                unconfigured_envs.append((env_name, reason))
                print(f"environment '{env_name}' note: {reason}")
        
        # note
        print("\nnote note:")
        print(f"noteenvironment: {len(configured_envs)} note")
        if configured_envs:
            print("  - " + "\n  - ".join(configured_envs))
        
        print(f"noteenvironment: {len(unconfigured_envs)} note")
        if unconfigured_envs:
            for env_name, reason in unconfigured_envs:
                print(f"  - {env_name}: {reason}")
        
        return configured_envs, unconfigured_envs


    def create_environment(self, env_name):
        """noteCondaenvironment noteenvironment note"""
        # noteenvironment note
        exists, reason = self.check_environment(env_name)
        if exists:
            print(f"environment '{env_name}' note note ")
            return True
        
        # noteenvironment note note note
        if reason and "note" not in reason:
            print(f"note: {reason}")
            choice = input(f"environment '{env_name}' note note? (y/n): ").strip().lower()
            if choice != 'y':
                print(f"noteenvironment '{env_name}'")
                return False
            
            # noteenvironment
            self.delete_environment(env_name)
        
        # noteenvironment note
        env_details = self.config_manager.get_env_config(env_name)
        if not env_details:
            print(f"note: noteenvironment'{env_name}' ")
            return False

        python_version = env_details.get("python_version", "3.10")
        conda_packages = env_details.get("conda_packages", [])
        pip_packages = env_details.get("pip_packages", [])
        channels = env_details.get("channels", [])
        post_commands = env_details.get("post_install_commands", [])

        # noteconda createnote
        cmd = ["conda", "create", "-y", "-n", env_name, f"python={python_version}"]

        # notecondanote
        cmd.extend(conda_packages)

        # note
        for channel in channels:
            cmd.extend(["-c", channel])
        
        print(f"noteenvironment'{env_name}'...")
        try:
            # noteconda createnote
            returncode, stdout, stderr = self.run_command(cmd)
            
            # notepipnote
            if pip_packages:
                print(f"note'{env_name}'notepipnote...")
                all_pip_packages_installed = True
                
                for package in pip_packages:
                    print(f"notepipnote: {package}...")

                    cmd = ["pip", "install", package]
                    # pip_cmd = ["conda", "run", "-n", env_name, 
                    
                    try:
                        # note
                        returncode, stdout, stderr = self.run_in_conda_env(env_name=env_name, command=cmd,show_output=True)
                        print(f"pipnote {package} note ")
                    except subprocess.CalledProcessError as e:
                        print(f"notepipnote {package} note: {e}")
                        all_pip_packages_installed = False
                        
                        # note
                        if input(f"note {package} note note? (y/n): ").lower() != 'y':
                            return False
                
                if not all_pip_packages_installed:
                    print("note: notepipnote")
                    if input("noteenvironment note? (y/n): ").lower() != 'y':
                        return False
                else:
                    print("notepipnote ")

            
            # note
            if post_commands:
                print(f"note'{env_name}'note...")
                all_commands_succeeded = True
                
                for command in post_commands:
                    print(f"\n==== note: {command} ====")
                    try:
                        # note
                        import shlex
                        command_parts = shlex.split(command)
                        
                        # note run_in_conda_env note conda run
                        print(f"noteenvironment '{env_name}' note: {' '.join(command_parts)}")
                        
                        # note note
                        returncode, stdout, stderr = self.run_in_conda_env(
                            env_name=env_name,
                            command=command_parts,
                            show_output=True
                        )
                        
                        print(f"==== note ====")
                        
                        # note
                        self.record_post_command_execution(env_name, command)
                        
                    except subprocess.CalledProcessError as e:
                        all_commands_succeeded = False
                        print(f"==== note ====")
                        print(f"note: {e}")
                        
                        # note
                        if input("note note? (y/n): ").lower() != 'y':
                            break
                
                if all_commands_succeeded:
                    print(f"\nnote!")
                else:
                    print(f"\nnote: note")
            return True
        except subprocess.CalledProcessError as e:
            print(f"noteenvironment'{env_name}'note: {e}")
            return False

    
    def delete_environment(self, env_name):
        """noteCondaenvironment"""
        # noteenvironment note noteenvironment note
        marker_path = self.get_post_commands_marker_path(env_name)
        
        print(f"noteenvironment'{env_name}'...")
        try:
            returncode, stdout, stderr = self.run_command(["conda", "env", "remove", "-y", "-n", env_name])
            print(f"environment'{env_name}'note!")
            
            # note noteenvironment note note note 
            if marker_path and os.path.exists(marker_path):
                try:
                    os.remove(marker_path)
                except (OSError, IOError):
                    pass
                    
            return True
        except subprocess.CalledProcessError as e:
            print(f"noteenvironment'{env_name}'note: {e}")
            return False
    
    def get_existing_environments(self):
        """noteCondaenvironment note"""
        returncode, stdout, stderr = self.run_command(["conda", "env", "list"])
        existing_envs = []
        for line in stdout.splitlines():
            if line and not line.startswith('#'):
                env_name = line.split()[0]
                if env_name != "base":  # notebaseenvironment
                    existing_envs.append(env_name)
        return existing_envs
    
    def run_in_conda_env(self, env_name, command, show_output=True):
        """
        notecondaenvironment note note
        
        note:
            env_name: condaenvironment note
            command: note note 
            show_output: note
            
        note:
            (returncode, stdout, stderr) note
        """
        # if command[0] == "pip": 
        #     command.insert(2, "--progress-bar=on")
        # notecommandnote
        if isinstance(command, str):
            import shlex
            command = shlex.split(command)
        
        # notecondanote
        conda_path = self.conda_base_path
        if not conda_path:
            print("note: notecondanote")
            return 1, "", "notecondanote"
            
        # noteenvironment note
        if os.name == 'nt':  # Windows
            # Windowsnote
            activate_cmd = f"conda activate {env_name} && "
            shell = True
            full_command = activate_cmd + " ".join(command)
        else:  # Linux/MacOS
            # notesourcenoteenvironment
            activate_path = os.path.join(conda_path, "bin", "activate")
            activate_cmd = f"source {activate_path} {env_name} && "
            shell = True
            full_command = activate_cmd + " ".join(command)
            
        # ANSI note
        CYAN = "\033[96m"  # note
        RESET = "\033[0m"  # note
        
        print(f"{CYAN}note: {full_command}{RESET}")

        # note noteshellnote
        process = subprocess.Popen(
            full_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
            # bufsize=0,  # note note Popen note text=True note
            executable='/bin/bash' if os.name != 'nt' else None  # noteLinux/Macnotebash
        )

        # note note communicate() note
        if not show_output:
            stdout, stderr = process.communicate()
            return process.returncode, stdout, stderr

        # note
        # note
        stdout_all = []
        stderr_all = []

        # note select note
        import select
        import sys # note

        # note
        GREEN = "\033[92m"  # note
        RED = "\033[91m"    # note
        RESET = "\033[0m"  # note
        CYAN = "\033[96m"  # note

        error_keywords = ["error", "failed", "not found", "exception", "fatal"]

        while process.poll() is None:  # note
            # note
            read_pipes = []
            if process.stdout:
                read_pipes.append(process.stdout)
            if process.stderr:
                read_pipes.append(process.stderr)

            if not read_pipes: # note Popen note note
                break

            # note
            ready_pipes, _, _ = select.select(read_pipes, [], [], 0.1)

            for pipe in ready_pipes:
                try:
                    line = pipe.readline()
                    # readline() note EOF note ''
                    # note None note note 
                    if line is None:
                        continue
                    if not line: # note EOF (note)
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
                    # note
                    print(f"note: {e}", file=sys.stderr)
                    # note note
                    pass
                except Exception as e: # note
                    print(f"note: {e}", file=sys.stderr)
                    pass


        # note
        # note note communicate() note 
        # note readline note 
        # note note 
        stdout_rem, stderr_rem = process.communicate()
        if stdout_rem:
            stdout_all.append(stdout_rem)
            if show_output:
                print(f"{GREEN}{stdout_rem}{RESET}", end='')
        if stderr_rem:
            stderr_all.append(stderr_rem)
            if show_output:
                # note
                for line in stderr_rem.splitlines():
                    is_error = any(keyword in line.lower() for keyword in error_keywords)
                    color = RED if is_error else CYAN
                    print(f"{color}{line}{RESET}")


        return process.returncode, ''.join(stdout_all), ''.join(stderr_all)
    
    def run_command(self, command, show_output=True):
        """
        noteshellnote
        
        note:
            command: note note 
            show_output: note
            
        note:
            (returncode, stdout, stderr) note
        """
        # ANSI note
        GRAY = "\033[90m"   # note
        RESET = "\033[0m"   # note
        RED = "\033[91m"    # note
        LIGHT_GREEN = "\033[92m"  # note
        GREEN = "\033[32m"  # note
        YELLOW = "\033[93m" # note
        BLUE = "\033[94m"   # note
        MAGENTA = "\033[95m" # note
        CYAN = "\033[96m"   # note
        WHITE = "\033[97m"  # note
        
        # note
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,      # note
            bufsize=0       # note note0note
        )  

        # note
        stdout_all = []
        stderr_all = []

        if show_output:
            # note
            for line in process.stdout:
                stdout_all.append(line)  # note
                print(f"{CYAN}{line}{RESET}", end='')  # note note

        # note
        stdout, stderr = process.communicate()  # note
        if stdout:
            stdout_all.append(stdout)  # note
            if show_output:
                print(f"{CYAN}{stdout}{RESET}", end='')

        # note
        error_keywords = ["error", "failed", "not found", "exception", "fatal"]  # note

        if stderr:
            stderr_all.append(stderr)  # note
            if show_output:
                for line in stderr.splitlines():  # notestderrnote
                    is_error = any(keyword in line.lower() for keyword in error_keywords)  # note
                    if is_error:
                        print(f"{RED}{line}{RESET}")  # note
                    else:
                        print(f"{BLUE}{line}{RESET}")  # note

        return process.returncode, ''.join(stdout_all), ''.join(stderr_all)  # note
    
    def run_cli(self):
        """note"""
        if not self.check_conda_installed():
            print("noteCondanotePATHnote ")
            return
        
        env_list = self.env_config.get("env_list", {})
        
        if not env_list:
            print("noteenvironment note ")
            return
        
        while True:
            print("\n" + "="*50)
            print("**Condaenvironment note**")
            print("="*50)
            print("1. noteenvironment")
            print("2. noteenvironment")
            print("3. noteenvironment note")
            print("4. note")
            
            choice = input("\nnote (1-4): ").strip()
            
            if choice == "1":
                self.list_environments()
                
                env_names = list(env_list.keys())
                env_input = input("\nnoteenvironment note (note'all'noteenvironment): ").strip()
                
                if env_input.lower() == 'all':
                    print("\nnoteenvironment...")
                    for env_name in env_names:
                        self.create_environment(env_name)
                else:
                    try:
                        # note
                        env_idx = int(env_input) - 1
                        if 0 <= env_idx < len(env_names):
                            env_name = env_names[env_idx]
                            self.create_environment(env_name)
                        else:
                            print("noteenvironment note ")
                    except ValueError:
                        # note noteenvironment note
                        if env_input in env_names:
                            self.create_environment(env_input)
                        else:
                            print(f"noteenvironment'{env_input}' ")
            
            elif choice == "2":
                existing_envs = self.get_existing_environments()
                
                if not existing_envs:
                    print("notecondaenvironment (notebase) ")
                    continue
                
                print("\n**notecondaenvironment:**")
                for i, env_name in enumerate(existing_envs, 1):
                    print(f"{i}. {env_name}")
                
                env_input = input("\nnoteenvironment note (note'all'noteenvironment): ").strip()
                
                if env_input.lower() == 'all':
                    confirm = input("noteenvironment note?note! (y/n): ").strip().lower()
                    if confirm == 'y':
                        print("\nnoteenvironment...")
                        for env_name in existing_envs:
                            self.delete_environment(env_name)
                    else:
                        print("note ")
                else:
                    try:
                        # note
                        env_idx = int(env_input) - 1
                        if 0 <= env_idx < len(existing_envs):
                            env_name = existing_envs[env_idx]
                            self.delete_environment(env_name)
                        else:
                            print("noteenvironment note ")
                    except ValueError:
                        # note noteenvironment note
                        if env_input in existing_envs:
                            self.delete_environment(env_input)
                        else:
                            print(f"noteenvironment'{env_input}' ")
            
            elif choice == "3":
                configured_envs, unconfigured_envs = self.check_all_environments()
            
            elif choice == "4":
                print("noteCondaenvironment note note!")
                break
            
            else:
                print("note note1note4note ")


class _LazyEnvManager:
    """note EnvManager note Conda note"""
    def __init__(self):
        self._instance = None

    def _get(self):
        if self._instance is None:
            self._instance = EnvManager()
        return self._instance

    def __getattr__(self, name):
        return getattr(self._get(), name)

# note note
env_manager = _LazyEnvManager()

def main():
    """note"""
    # noteConfigManager
    env_manager.run_cli()

if __name__ == "__main__":
    main()

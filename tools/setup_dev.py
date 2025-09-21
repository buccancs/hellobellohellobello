#!/usr/bin/env python3
"""
Developer Environment Setup Script
Multi-Modal Physiological Sensing Platform

This script sets up the complete development environment including:
- Pre-commit hooks installation
- Linting tools configuration
- Testing environment validation
- IDE configuration helpers

Usage:
    python scripts/setup_dev.py [options]

Options:
    --hooks-only    Only install pre-commit hooks
    --validate      Validate existing setup
    --reinstall     Force reinstall of all components
    --ide           Setup IDE configuration files
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


class DevSetup:
    """Developer environment setup manager."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.project_root = Path.cwd()

    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message with appropriate prefix."""
        prefix = (
            "🔧"
            if level == "INFO"
            else "✅" if level == "SUCCESS" else "❌" if level == "ERROR" else "⚠️"
        )
        print(f"{prefix} {message}")

    def check_command(self, cmd: str) -> bool:
        """Check if a command is available."""
        return shutil.which(cmd) is not None

    def run_command(self, cmd: list[str], name: str, cwd: Path | None = None) -> bool:
        """Run a command and return success status."""
        try:
            self.log(f"Running {name}...")
            result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
            if result.returncode == 0:
                self.log(f"{name} completed successfully", "SUCCESS")
                return True
            else:
                self.log(f"{name} failed: {result.stderr}", "ERROR")
                return False
        except Exception as e:
            self.log(f"{name} failed with exception: {e}", "ERROR")
            return False

    def install_python_dependencies(self) -> bool:
        """Install required Python dependencies."""
        self.log("Installing Python development dependencies...")

        requirements_file = self.project_root / "pc_controller" / "requirements.txt"
        if requirements_file.exists():
            if not self.run_command(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                "Core Requirements",
            ):
                return False

        dev_tools = [
            "pre-commit>=3.0.0",
            "ruff>=0.12.0",
            "mypy>=1.17.0",
            "black>=25.0.0",
            "isort>=6.0.0",
            "bandit>=1.8.0",
            "pytest>=8.0.0",
            "pytest-cov>=6.0.0",
            "pytest-timeout>=2.4.0",
        ]

        for tool in dev_tools:
            if not self.run_command(
                [sys.executable, "-m", "pip", "install", tool], f"Installing {tool}"
            ):
                self.log(f"Failed to install {tool}, but continuing...", "ERROR")

        return True

    def install_pre_commit_hooks(self) -> bool:
        """Install and configure pre-commit hooks."""
        self.log("Setting up pre-commit hooks...")

        if not self.check_command("pre-commit"):
            self.log("pre-commit not found in PATH, trying to install...", "ERROR")
            if not self.run_command(
                [sys.executable, "-m", "pip", "install", "pre-commit"], "Installing pre-commit"
            ):
                return False

        if not self.run_command(["pre-commit", "install"], "Installing pre-commit hooks"):
            return False

        if not self.run_command(
            ["pre-commit", "install", "--hook-type", "commit-msg"], "Installing commit-msg hook"
        ):
            self.log(
                "Failed to install commit-msg hook, but pre-commit hooks are installed", "ERROR"
            )

        if not self.run_command(["pre-commit", "autoupdate"], "Updating pre-commit hooks"):
            self.log("Failed to update hooks, but they are installed", "ERROR")

        return True

    def validate_setup(self) -> bool:
        """Validate the development environment setup."""
        self.log("Validating development environment...")

        success = True

        python_version = sys.version_info
        if python_version < (3, 11):
            self.log(
                f"Python {python_version.major}.{python_version.minor} is too old, need 3.11+",
                "ERROR",
            )
            success = False
        else:
            self.log(
                f"Python {python_version.major}.{python_version.minor} is compatible", "SUCCESS"
            )

        required_tools = ["pre-commit", "git"]
        for tool in required_tools:
            if self.check_command(tool):
                self.log(f"{tool} is available", "SUCCESS")
            else:
                self.log(f"{tool} is not available", "ERROR")
                success = False

        python_packages = ["ruff", "mypy", "black", "isort", "pytest", "bandit"]
        for package in python_packages:
            try:
                __import__(package)
                self.log(f"Python package {package} is available", "SUCCESS")
            except ImportError:
                self.log(f"Python package {package} is not available", "ERROR")
                success = False

        # Check configuration files
        config_files = [
            ".pre-commit-config.yaml",
            "pyproject.toml",
            "pytest.ini",
            ".markdownlint.yaml",
        ]

        for config_file in config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                self.log(f"Configuration file {config_file} exists", "SUCCESS")
            else:
                self.log(f"Configuration file {config_file} is missing", "ERROR")
                success = False

        required_dirs = [
            "pc_controller/src",
            "pc_controller/tests",
            "android_sensor_node/app",
            "scripts",
        ]

        for directory in required_dirs:
            dir_path = self.project_root / directory
            if dir_path.exists():
                self.log(f"Directory {directory} exists", "SUCCESS")
            else:
                self.log(f"Directory {directory} is missing", "ERROR")
                success = False

        return success

    def test_pre_commit_hooks(self) -> bool:
        """Test pre-commit hooks on a sample file."""
        self.log("Testing pre-commit hooks...")

        if not self.run_command(["pre-commit", "run", "--all-files"], "Testing pre-commit hooks"):
            self.log("Pre-commit hooks found issues, but this is normal for first run", "ERROR")

        return True

    def setup_ide_config(self) -> None:
        """Setup IDE configuration files."""
        if not self.args.ide:
            return

        self.log("Setting up IDE configuration...")

        # VSCode settings
        vscode_dir = self.project_root / ".vscode"
        vscode_dir.mkdir(exist_ok=True)

        vscode_settings = {
            "python.defaultInterpreterPath": "./pc_controller/.venv/bin/python",
            "python.linting.enabled": True,
            "python.linting.ruffEnabled": True,
            "python.formatting.provider": "black",
            "python.formatting.blackArgs": ["--line-length=100"],
            "python.sortImports.args": ["--profile=black"],
            "python.testing.pytestEnabled": True,
            "python.testing.pytestArgs": ["pc_controller/tests"],
            "files.exclude": {
                "**/__pycache__": True,
                "**/*.pyc": True,
                "**/node_modules": True,
                "**/.pytest_cache": True,
                "**/.mypy_cache": True,
                "**/coverage.xml": True,
                "htmlcov": True,
            },
            "editor.formatOnSave": True,
            "editor.codeActionsOnSave": {"source.organizeImports": "explicit"},
        }

        import json

        with open(vscode_dir / "settings.json", "w") as f:
            json.dump(vscode_settings, f, indent=2)

        self.log("VSCode settings created", "SUCCESS")

        vscode_extensions = {
            "recommendations": [
                "ms-python.python",
                "ms-python.black-formatter",
                "ms-python.isort",
                "charliermarsh.ruff",
                "ms-python.mypy-type-checker",
                "ms-vscode.test-adapter-converter",
                "davidanson.vscode-markdownlint",
                "redhat.vscode-yaml",
                "streetsidesoftware.code-spell-checker",
            ]
        }

        with open(vscode_dir / "extensions.json", "w") as f:
            json.dump(vscode_extensions, f, indent=2)

        self.log("VSCode extensions recommendations created", "SUCCESS")

    def run_setup(self) -> int:
        """Run the complete development environment setup."""
        self.log("🚀 Setting up development environment for Multi-Modal Sensor Platform")

        success = True

        try:
            if not self.args.hooks_only and not self.install_python_dependencies():
                success = False

            if not self.install_pre_commit_hooks():
                success = False

            # Setup IDE configuration
            self.setup_ide_config()

            # Validate setup (unless reinstall mode)
            if not self.args.reinstall and not self.validate_setup():
                success = False

            if not self.test_pre_commit_hooks():
                success = False

        except KeyboardInterrupt:
            self.log("Setup interrupted by user", "ERROR")
            return 130
        except Exception as e:
            self.log(f"Setup failed with exception: {e}", "ERROR")
            return 1

        if success:
            self.log("🎉 Development environment setup completed successfully!", "SUCCESS")
            print("\n📋 Next steps:")
            print("1. Run 'python scripts/test_pipeline.py --all' to test everything")
            print("2. Make some changes and commit to test pre-commit hooks")
            print("3. Check out the documentation in docs/ for more details")
            return 0
        else:
            self.log("❌ Setup completed with errors", "ERROR")
            return 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Modal Sensor Platform Development Environment Setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--hooks-only", action="store_true", help="Only install pre-commit hooks")
    parser.add_argument("--validate", action="store_true", help="Only validate existing setup")
    parser.add_argument("--reinstall", action="store_true", help="Force reinstall all components")
    parser.add_argument("--ide", action="store_true", help="Setup IDE configuration files")

    args = parser.parse_args()

    setup = DevSetup(args)

    if args.validate:
        success = setup.validate_setup()
        return 0 if success else 1
    else:
        return setup.run_setup()


if __name__ == "__main__":
    sys.exit(main())

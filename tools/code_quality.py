#!/usr/bin/env python3
"""
Code quality analyzer and fixer for the Multi-Modal Sensor Platform.
This script sets up linting tools and fixes common code quality issues.
"""

import os
import subprocess
import sys
from pathlib import Path


def setup_linting_tools():
    """Install and configure linting tools"""
    print("Setting up code quality tools...")

    tools = ["ruff>=0.1.0", "mypy>=1.7.0", "black>=23.0.0", "isort>=5.12.0"]

    for tool in tools:
        try:
            print(f"Installing {tool}...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", tool], check=True, capture_output=True
            )
            print(f"✅ {tool} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {tool}: {e}")
            return False

    return True


def run_linting():
    """Run comprehensive linting on the codebase"""
    print("\n" + "=" * 50)
    print("Running code quality analysis...")
    print("=" * 50)

    pc_controller_src = Path("pc_controller/src")
    if not pc_controller_src.exists():
        print("❌ pc_controller/src directory not found")
        return False

    print("\n1. Running Ruff linting...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", str(pc_controller_src), "--output-format=text"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✅ Ruff: No issues found")
        else:
            print("⚠️  Ruff found issues:")
            print(result.stdout)

    except FileNotFoundError:
        print("❌ Ruff not available")

    print("\n2. Running MyPy type checking...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "mypy", str(pc_controller_src), "--ignore-missing-imports"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✅ MyPy: No type issues found")
        else:
            print("⚠️  MyPy found type issues:")
            print(result.stdout)

    except FileNotFoundError:
        print("❌ MyPy not available")

    print("\n3. Checking code formatting with Black...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "black", "--check", "--diff", str(pc_controller_src)],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✅ Black: Code formatting is correct")
        else:
            print("⚠️  Black found formatting issues:")
            print(result.stdout)

    except FileNotFoundError:
        print("❌ Black not available")

    print("\n4. Checking import sorting with isort...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "isort", "--check-only", "--diff", str(pc_controller_src)],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✅ isort: Import sorting is correct")
        else:
            print("⚠️  isort found import issues:")
            print(result.stdout)

    except FileNotFoundError:
        print("❌ isort not available")

    return True


def fix_code_issues():
    """Auto-fix common code quality issues"""
    print("\n" + "=" * 50)
    print("Auto-fixing code quality issues...")
    print("=" * 50)

    pc_controller_src = Path("pc_controller/src")

    print("\n1. Auto-fixing with Ruff...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--fix", str(pc_controller_src)],
            capture_output=True,
            text=True,
        )

        print("✅ Ruff auto-fix completed")
        if result.stdout:
            print("Fixed issues:")
            print(result.stdout)

    except FileNotFoundError:
        print("❌ Ruff not available for auto-fix")

    print("\n2. Auto-formatting with Black...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "black", str(pc_controller_src)], capture_output=True, text=True
        )

        print("✅ Black auto-formatting completed")
        if "reformatted" in result.stderr:
            print("Reformatted files:")
            print(result.stderr)

    except FileNotFoundError:
        print("❌ Black not available for auto-formatting")

    print("\n3. Auto-sorting imports with isort...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "isort", str(pc_controller_src)], capture_output=True, text=True
        )

        print("✅ isort auto-sorting completed")
        if result.stdout:
            print("Sorted imports in:")
            print(result.stdout)

    except FileNotFoundError:
        print("❌ isort not available for auto-sorting")


def create_config_files():
    """Create configuration files for linting tools"""
    print("\n" + "=" * 50)
    print("Creating configuration files...")
    print("=" * 50)
    
    # Check if pyproject.toml already exists with good configuration
    if os.path.exists("pyproject.toml"):
        print("✅ pyproject.toml already exists - using existing configuration")
        return

    # Create pyproject.toml for tool configuration
    pyproject_content = """[tool.ruff]
target-version = "py311"
line-length = 100
select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings
    "F",  # pyflakes
    "I",  # isort
    "B",  # flake8-bugbear
    "C4", # flake8-comprehensions
    "UP", # pyupgrade
]
ignore = [
    "E501",  # line too long, handled by black
    "B008",  # do not perform function calls in argument defaults
    "C901",  # too complex
]

[tool.ruff.per-file-ignores]
"__init__.py" = ["F401"]
"test_*.py" = ["B011"]

[tool.black]
line-length = 100
target-version = ['py311']
include = '\\.pyi?$'
extend-exclude = '''
/(
  # directories
  \\.eggs
  | \\.git
  | \\.hg
  | \\.mypy_cache
  | \\.tox
  | \\.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
line_length = 100
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
disallow_incomplete_defs = false
check_untyped_defs = true
disallow_untyped_decorators = false
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true
"""

    with open("pyproject.toml", "w") as f:
        f.write(pyproject_content)
    print("✅ Created pyproject.toml")

    # Create .ruff.toml for additional ruff configuration
    ruff_content = """# Ruff configuration for Multi-Modal Sensor Platform
target-version = "py311"
line-length = 100

# Enable specific rule sets
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
    "N",   # pep8-naming
    "S",   # flake8-bandit (security)
    "T20", # flake8-print
]

# Ignore specific rules
ignore = [
    "E501",  # line too long (handled by black)
    "S101",  # use of assert (common in tests)
    "T201",  # print statements (allowed for debugging)
    "B008",  # function calls in argument defaults
]

# Per-file ignores
[per-file-ignores]
"__init__.py" = ["F401"]  # unused imports in __init__.py
"test_*.py" = ["S101", "B011"]  # asserts and setattr in tests
"**/tests/**" = ["S101", "B011"]  # asserts and setattr in tests

# Exclude specific directories
exclude = [
    ".bzr",
    ".direnv",
    ".eggs",
    ".git",
    ".hg",
    ".mypy_cache",
    ".nox",
    ".pants.d",
    ".pytype",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pypackages__",
    "_build",
    "buck-out",
    "build",
    "dist",
    "node_modules",
    "venv",
]
"""

    with open(".ruff.toml", "w") as f:
        f.write(ruff_content)
    print("✅ Created .ruff.toml")


def generate_quality_report():
    """Generate a comprehensive code quality report"""
    print("\n" + "=" * 50)
    print("Generating code quality report...")
    print("=" * 50)

    pc_controller_src = Path("pc_controller/src")

    py_files = list(pc_controller_src.rglob("*.py"))
    total_lines = 0

    for py_file in py_files:
        try:
            with open(py_file, encoding='utf-8') as f:
                total_lines += len(f.readlines())
        except Exception:
            continue

    print("📊 Code Statistics:")
    print(f"  Python files: {len(py_files)}")
    print(f"  Total lines: {total_lines:,}")

    try:
        # Count TODO/FIXME comments
        todo_count = 0
        fixme_count = 0

        for py_file in py_files:
            try:
                with open(py_file, encoding='utf-8') as f:
                    content = f.read().lower()
                    todo_count += content.count("todo")
                    fixme_count += content.count("fixme")
            except Exception:
                continue

        print(f"  TODO comments: {todo_count}")
        print(f"  FIXME comments: {fixme_count}")

    except Exception as e:
        print(f"Error generating metrics: {e}")

    test_files = (
        list(Path("pc_controller/tests").rglob("test_*.py"))
        if Path("pc_controller/tests").exists()
        else []
    )
    print(f"  Test files: {len(test_files)}")

    print("\n📋 Quality Recommendations:")

    if todo_count > 10:
        print("  ⚠️  High number of TODO comments - consider addressing them")

    if fixme_count > 0:
        print("  ⚠️  FIXME comments found - these should be prioritized")

    if len(test_files) < len(py_files) * 0.5:
        print("  ⚠️  Low test coverage - consider adding more tests")

    print("  ✅ Use 'python scripts/code_quality.py --fix' to auto-fix issues")
    print("  ✅ Run 'python scripts/code_quality.py --lint' for detailed analysis")


def main():
    """Main function to run code quality tools"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Code quality analyzer for Multi-Modal Sensor Platform"
    )
    parser.add_argument("--setup", action="store_true", help="Install linting tools")
    parser.add_argument("--lint", action="store_true", help="Run linting analysis")
    parser.add_argument("--fix", action="store_true", help="Auto-fix code issues")
    parser.add_argument("--config", action="store_true", help="Create configuration files")
    parser.add_argument("--report", action="store_true", help="Generate quality report")
    parser.add_argument("--all", action="store_true", help="Run all operations")

    args = parser.parse_args()

    if not any([args.setup, args.lint, args.fix, args.config, args.report, args.all]):
        parser.print_help()
        return

    print("🔍 Multi-Modal Sensor Platform Code Quality Analyzer")
    print("=" * 60)

    if (args.all or args.setup) and not setup_linting_tools():
        print("❌ Failed to setup linting tools")
        return

    if args.all or args.config:
        create_config_files()

    if args.all or args.lint:
        run_linting()

    if args.all or args.fix:
        fix_code_issues()

    if args.all or args.report:
        generate_quality_report()

    print("\n✅ Code quality analysis complete!")


if __name__ == "__main__":
    main()

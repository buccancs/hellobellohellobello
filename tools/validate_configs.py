#!/usr/bin/env python3
"""
Configuration validation script for the Multi-Modal Physiological Sensing Platform.
This script validates project configuration files for consistency and correctness.
"""

import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def validate_pyproject_toml() -> list[str]:
    """Validate pyproject.toml for consistency and required sections."""
    errors = []

    try:
        with open("pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        required_sections = ["project", "build-system", "tool.ruff", "tool.mypy"]
        for section in required_sections:
            keys = section.split(".")
            current = config
            for key in keys:
                if key not in current:
                    errors.append(f"Missing required section: {section}")
                    break
                current = current[key]

        # Validate tool configurations
        if "tool" in config:
            tool_config = config["tool"]

            # Check Ruff config
            if "ruff" in tool_config:
                ruff_config = tool_config["ruff"]
                if "target-version" not in ruff_config:
                    errors.append("Ruff config missing target-version")
                if "line-length" not in ruff_config:
                    errors.append("Ruff config missing line-length")

            # Check MyPy config
            if "mypy" in tool_config:
                mypy_config = tool_config["mypy"]
                if "python_version" not in mypy_config:
                    errors.append("MyPy config missing python_version")

        if "project" in config:
            project = config["project"]
            required_fields = ["name", "version", "description", "requires-python"]
            errors.extend(
                [
                    f"Project metadata missing: {field}"
                    for field in required_fields
                    if field not in project
                ]
            )

    except Exception as e:
        errors.append(f"Error reading pyproject.toml: {e}")

    return errors


def validate_pytest_ini() -> list[str]:
    """Validate pytest.ini configuration."""
    errors = []

    try:
        pytest_ini = Path("pytest.ini")
        if not pytest_ini.exists():
            errors.append("pytest.ini not found")
            return errors

        content = pytest_ini.read_text()

        if "[pytest]" not in content:
            errors.append("pytest.ini missing [pytest] section")

        # Check for essential configurations
        required_configs = ["testpaths", "pythonpath", "timeout"]
        errors.extend(
            [
                f"pytest.ini missing {config} configuration"
                for config in required_configs
                if f"{config} =" not in content
            ]
        )

    except Exception as e:
        errors.append(f"Error reading pytest.ini: {e}")

    return errors


def validate_pre_commit_config() -> list[str]:
    """Validate .pre-commit-config.yaml."""
    errors = []

    try:
        import yaml

        with open(".pre-commit-config.yaml") as f:
            config = yaml.safe_load(f)

        if "repos" not in config:
            errors.append("Pre-commit config missing repos section")
            return errors

        repos = config["repos"]
        expected_repos = [
            "pre-commit-hooks",
            "black",
            "isort",
            "ruff-pre-commit",
            "mirrors-mypy",
            "bandit",
        ]

        repo_urls = [repo.get("repo", "") for repo in repos]
        errors.extend(
            [
                f"Pre-commit config missing {expected} hook"
                for expected in expected_repos
                if not any(expected in url for url in repo_urls)
            ]
        )

    except ImportError:
        errors.append("PyYAML not available for pre-commit config validation")
    except Exception as e:
        errors.append(f"Error reading .pre-commit-config.yaml: {e}")

    return errors


def validate_requirements_consistency() -> list[str]:
    """Validate that requirements.txt is consistent with pyproject.toml."""
    errors = []

    try:
        requirements_file = Path("pc_controller/requirements.txt")
        if not requirements_file.exists():
            errors.append("pc_controller/requirements.txt not found")
            return errors

        requirements = set()
        for line in requirements_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("
                package = line.split(">=")[0].split("==")[0].split("<")[0].split(">")[0]
                requirements.add(package.lower())

        with open("pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        if "project" in config and "dependencies" in config["project"]:
            pyproject_deps = set()
            for dep in config["project"]["dependencies"]:
                package = dep.split(">=")[0].split("==")[0].split("<")[0].split(">")[0]
                pyproject_deps.add(package.lower())

            only_in_requirements = requirements - pyproject_deps
            only_in_pyproject = pyproject_deps - requirements

            if only_in_requirements:
                errors.append(
                    f"Dependencies in requirements.txt but not pyproject.toml: {only_in_requirements}"
                )
            if only_in_pyproject:
                errors.append(
                    f"Dependencies in pyproject.toml but not requirements.txt: {only_in_pyproject}"
                )

    except Exception as e:
        errors.append(f"Error validating requirements consistency: {e}")

    return errors


def main() -> int:
    """Main validation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate project configuration files")
    parser.add_argument("--fast", action="store_true", help="Run only essential validations")
    args = parser.parse_args()

    print("🔍 Validating project configuration files...")

    all_errors = []

    validations = [
        ("pyproject.toml", validate_pyproject_toml),
        ("pytest.ini", validate_pytest_ini),
        (".pre-commit-config.yaml", validate_pre_commit_config),
        ("requirements consistency", validate_requirements_consistency),
    ]

    # In fast mode, skip the more expensive validations
    if args.fast:
        validations = [
            ("pyproject.toml", validate_pyproject_toml),
            ("requirements consistency", validate_requirements_consistency),
        ]

    for name, validation_func in validations:
        print(f"  Validating {name}...")
        errors = validation_func()
        if errors:
            print(f"    ❌ {len(errors)} error(s) found:")
            for error in errors:
                print(f"      • {error}")
            all_errors.extend(errors)
        else:
            print(f"    ✅ {name} is valid")

    if all_errors:
        print(f"\n❌ Found {len(all_errors)} total configuration errors")
        return 1
    else:
        print("\n✅ All configuration files are valid")
        return 0


if __name__ == "__main__":
    sys.exit(main())

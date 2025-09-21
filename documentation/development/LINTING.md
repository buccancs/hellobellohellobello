# Linting Guide

This document describes how to run linting tools on the codebase.

## Python Code Linting (Black)

The Python code linting is managed through the `scripts/code_quality.py` script.

### Setup Python Linting Tools
```bash
python3 scripts/code_quality.py --setup
```

### Run Python Linting (Black, Ruff, MyPy, isort)
```bash
python3 scripts/code_quality.py --lint
```

### Auto-fix Python Code Issues
```bash
python3 scripts/code_quality.py --fix
```

## Kotlin Code Linting (ktlint)

The Kotlin code linting is configured using the ktlint Gradle plugin.

### Check Kotlin Code Style
```bash
./gradlew :android_sensor_node:app:ktlintCheck
```

### Auto-format Kotlin Code
```bash
./gradlew :android_sensor_node:app:ktlintFormat
```

### Check Only Kotlin Scripts (build.gradle.kts files)
```bash
./gradlew :android_sensor_node:app:ktlintKotlinScriptCheck
```

## Status

- ✅ **Black linting**: Configured and working for Python code
- ✅ **ktlint**: Configured and working for Kotlin code
- ✅ **Ruff**: Additional Python linting
- ✅ **MyPy**: Python type checking
- ✅ **isort**: Python import sorting

## Configuration Files

- **Python**: Configuration is in `scripts/code_quality.py` and includes pyproject.toml setup
- **Kotlin**: Configuration is in `android_sensor_node/app/build.gradle.kts` with ktlint plugin

## Notes

- Python linting tools are installed via pip when running the setup command
- Kotlin linting is handled by Gradle and automatically downloads ktlint
- Both tools can auto-fix many formatting issues automatically
- Some violations require manual fixing (e.g., wildcard imports in Kotlin)
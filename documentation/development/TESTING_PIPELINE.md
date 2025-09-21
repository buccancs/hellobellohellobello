# Testing Pipeline and Pre-commit Setup Guide

This document describes the comprehensive testing pipeline and pre-commit hooks implemented for the Multi-Modal Physiological Sensing Platform.

## 🚀 Quick Start

### For New Developers

1. **Clone the repository** and navigate to the project directory
2. **Set up the development environment**:
   ```bash
   python scripts/setup_dev.py --ide
   ```
3. **Test everything works**:
   ```bash
   python scripts/test_pipeline.py --fast
   ```

### For Existing Developers

1. **Update your development environment**:
   ```bash
   python scripts/setup_dev.py --reinstall
   ```
2. **Validate your setup**:
   ```bash
   python scripts/setup_dev.py --validate
   ```

## ⚡ Performance Features

The testing pipeline is optimized for maximum performance:

### 🔥 Parallel Execution
- **Parallel test execution** with `pytest-xdist` using all available CPU cores
- **Parallel code quality checks** for linting, formatting, and type checking  
- **Configurable job count**: `python scripts/test_pipeline.py --jobs 8`
- **Smart CI optimization**: Limited to 4 parallel jobs in CI environments

### 📈 Caching Optimizations
- **Tool caching**: Ruff (`.ruff_cache`), MyPy (`.mypy_cache`), pytest (`.pytest_cache`)
- **Git hook caching**: Pre-commit hooks cached across runs
- **Gradle build cache**: Incremental builds with configuration cache
- **CI dependency caching**: Python packages, Gradle dependencies, build artifacts

### ⚡ Fast Development Workflow
- **Fast mode**: `--fast` flag skips integration and performance tests
- **Selective testing**: Only run tests affected by code changes
- **Incremental validation**: Skip expensive validations in development

### 🛠️ Performance Commands
```bash
# Fast development testing (< 30 seconds)
python scripts/test_pipeline.py --fast

# Full parallel testing (all cores)  
python scripts/test_pipeline.py --all --parallel

# Custom parallel jobs
python scripts/test_pipeline.py --jobs 4 --coverage

# CI-optimized execution
python scripts/test_pipeline.py --ci --coverage --security
```

## 📋 Testing Pipeline Components

### 1. Pre-commit Hooks (`.pre-commit-config.yaml`)

Automatically runs on every commit to ensure code quality:

- **Basic file checks**: trailing whitespace, file endings, YAML/JSON/TOML syntax
- **Python code formatting**: Black formatter with 100-character line length
- **Import sorting**: isort with Black profile compatibility
- **Python linting**: Ruff for fast, comprehensive linting
- **Type checking**: MyPy with project-specific configuration
- **Security scanning**: Bandit for security vulnerabilities
- **Kotlin linting**: ktlint for Android code (when available)
- **Markdown linting**: markdownlint for documentation consistency
- **Fast tests**: Quick unit tests on changed files
- **Configuration validation**: Ensures project configuration consistency

#### Installation and Usage

```bash
# Install hooks (done automatically by setup_dev.py)
pre-commit install

# Run hooks manually on all files
pre-commit run --all-files

# Update hooks to latest versions
pre-commit autoupdate

# Skip hooks for a specific commit (use sparingly!)
git commit --no-verify -m "Emergency fix"
```

### 2. Testing Orchestrator (`scripts/test_pipeline.py`)

Comprehensive testing pipeline that can run locally or in CI:

#### Usage Options

```bash
# Fast tests only (for development)
python scripts/test_pipeline.py --fast

# Full test suite with coverage and reports
python scripts/test_pipeline.py --all

# CI mode (optimized for automation)
python scripts/test_pipeline.py --ci --coverage --security

# Performance testing (when needed)
python scripts/test_pipeline.py --performance --report

# Android tests (requires Android SDK)
python scripts/test_pipeline.py --android

# Auto-fix issues where possible
python scripts/test_pipeline.py --fix --fast
```

#### Test Categories

1. **Configuration Validation**: Ensures all config files are consistent
2. **Code Quality Checks**: Ruff, Black, isort, MyPy
3. **Security Scanning**: Bandit security analysis
4. **Unit Tests**: Pytest with optional coverage reporting
5. **Integration Tests**: End-to-end system testing
6. **Performance Tests**: Benchmark critical performance metrics
7. **Android Tests**: Kotlin unit tests and linting

### 3. GitHub Actions Workflows

#### Enhanced Testing Pipeline (`.github/workflows/enhanced-testing.yml`)

Runs automatically on push/PR with multiple parallel jobs:

- **Code Quality**: Pre-commit hooks, linting, configuration validation
- **Python Tests**: Unit tests with coverage reporting and Codecov integration
- **Integration Tests**: Comprehensive system testing
- **Android Tests**: Kotlin linting and unit testing
- **Performance Tests**: Triggered on PRs with 'performance' label
- **Build Artifacts**: Automated builds for main/develop branches
- **Pipeline Status**: Final status aggregation and reporting

#### Existing Workflows

The enhanced pipeline complements existing workflows:
- `ci.yml`: Basic CI with Python and Android testing
- `ci-cd.yml`: Full CD pipeline with deployment

## 🛠️ Configuration Files

### Core Configuration

- **`pyproject.toml`**: Python project metadata, tool configurations (Ruff, MyPy, Black)
- **`pytest.ini`**: Pytest configuration with test paths, timeouts, markers
- **`.pre-commit-config.yaml`**: Pre-commit hooks configuration
- **`.markdownlint.yaml`**: Markdown linting rules for documentation

### Supporting Scripts

- **`scripts/setup_dev.py`**: Developer environment setup automation
- **`scripts/test_pipeline.py`**: Comprehensive testing orchestration
- **`scripts/validate_configs.py`**: Configuration file validation
- **`scripts/code_quality.py`**: Legacy linting script (still functional)

## 📊 Quality Metrics and Reporting

### Test Coverage

- **Target**: >90% line coverage for critical components
- **Reports**: HTML, XML, and terminal coverage reports
- **Integration**: Codecov for PR coverage analysis

### Code Quality Metrics

- **Linting**: Ruff with comprehensive rule set
- **Formatting**: Black with 100-character line length
- **Type Safety**: MyPy with strict optional and union checking
- **Security**: Bandit security scanning with JSON reports
- **Import Organization**: isort with Black profile

### Performance Monitoring

- **Benchmarking**: pytest-benchmark for performance regression detection
- **Memory Profiling**: Optional memory usage tracking
- **Latency Testing**: End-to-end latency validation

## 🎯 Development Workflow

### Daily Development

1. **Make your changes** to the codebase
2. **Run fast tests** while developing:
   ```bash
   python scripts/test_pipeline.py --fast
   ```
3. **Commit your changes** - pre-commit hooks run automatically
4. **Push to your branch** - CI pipeline validates your changes

### Before Pull Request

1. **Run comprehensive tests**:
   ```bash
   python scripts/test_pipeline.py --all
   ```
2. **Check coverage and fix any issues**
3. **Ensure all CI checks pass**
4. **Add performance label** if changes affect performance-critical code

### Troubleshooting

#### Pre-commit Hook Failures

```bash
# View detailed error information
pre-commit run --all-files --verbose

# Fix specific hook failures
black pc_controller/src/  # Fix formatting
ruff check pc_controller/src/ --fix  # Fix linting issues
isort pc_controller/src/  # Fix import sorting
```

#### Test Failures

```bash
# Run specific test with detailed output
python -m pytest pc_controller/tests/test_specific.py -v --tb=long

# Run with coverage to identify untested code
python -m pytest --cov=pc_controller/src --cov-report=html
open htmlcov/index.html  # View coverage report
```

#### Configuration Issues

```bash
# Validate all configuration files
python scripts/validate_configs.py

# Reinstall development environment
python scripts/setup_dev.py --reinstall
```

## 🔧 IDE Integration

### Visual Studio Code

The `setup_dev.py --ide` command creates:
- `.vscode/settings.json`: Python interpreter, linting, formatting settings
- `.vscode/extensions.json`: Recommended extensions for the project

### Other IDEs

Manual configuration for other IDEs:
- **Python Interpreter**: Use project virtual environment
- **Linter**: Configure Ruff or use built-in integration
- **Formatter**: Configure Black with 100-character line length
- **Type Checker**: Configure MyPy with project settings

## 📝 Best Practices

### Code Quality

1. **Write tests first**: Use TDD when possible
2. **Keep functions small**: Single responsibility principle
3. **Use type hints**: All public APIs should be typed
4. **Document public interfaces**: Docstrings for all public modules/classes/functions
5. **Handle errors gracefully**: Use appropriate exception handling

### Git Workflow

1. **Use conventional commits**: `feat:`, `fix:`, `docs:`, `refactor:`, etc.
2. **Keep commits atomic**: One logical change per commit
3. **Write descriptive commit messages**: Explain why, not just what
4. **Use feature branches**: Don't commit directly to main/develop
5. **Squash before merging**: Clean up commit history for main branches

### Testing Strategy

1. **Unit tests**: Test individual components in isolation
2. **Integration tests**: Test component interactions
3. **End-to-end tests**: Test complete user workflows
4. **Performance tests**: Monitor critical performance metrics
5. **Manual testing**: Complement automated tests with human verification

## 🚦 CI/CD Integration

### Pull Request Process

1. **Pre-commit hooks** catch issues early
2. **Code quality job** validates formatting, linting, configuration
3. **Python tests job** runs unit tests with coverage
4. **Integration tests job** validates system interactions
5. **Android tests job** validates mobile component
6. **Performance tests job** (optional) benchmarks critical paths
7. **Build artifacts job** creates distribution packages

### Deployment Pipeline

- **Develop branch**: Continuous integration, preview deployments
- **Main branch**: Production deployments, release artifacts
- **Release tags**: Automated release creation and distribution

## 📚 Additional Resources

- **[LINTING.md](LINTING.md)**: Detailed linting configuration and usage
- **[Build.md](docs/markdown/Build.md)**: Build system documentation
- **[Performance Testing](docs/diagrams/performance/performance_testing.md)**: Performance monitoring details
- **[Repository Validation Guide](docs/REPOSITORY_VALIDATION_GUIDE.md)**: Validation procedures

## 🤝 Contributing

1. **Set up your development environment** using this guide
2. **Read the project documentation** in the `docs/` directory
3. **Follow the code quality standards** enforced by the pipeline
4. **Add tests** for any new functionality
5. **Update documentation** when making user-facing changes
6. **Use the testing pipeline** to validate your changes before submitting

---

*This testing pipeline ensures consistent code quality, comprehensive test coverage, and reliable CI/CD processes for the Multi-Modal Physiological Sensing Platform.*

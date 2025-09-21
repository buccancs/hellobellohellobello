#!/usr/bin/env python3
"""
Mermaid Chart Validation Script

This script validates all generated Mermaid charts for syntax correctness
and provides a comprehensive summary of the visualization suite.
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple


def validate_mermaid_syntax(content: str) -> Tuple[bool, List[str]]:
    """Validate basic Mermaid syntax and return validation results."""
    errors = []
    
    # Check for required Mermaid code block
    if not content.strip().startswith('```mermaid'):
        errors.append("Missing opening ```mermaid code block")
    
    if not content.strip().endswith('```'):
        errors.append("Missing closing ``` code block")
    
    # Extract Mermaid content
    mermaid_match = re.search(r'```mermaid\s*(.*?)\s*```', content, re.DOTALL)
    if not mermaid_match:
        errors.append("No valid Mermaid content found")
        return False, errors
    
    mermaid_content = mermaid_match.group(1)
    lines = mermaid_content.strip().split('\n')
    
    # Validate chart type
    chart_types = ['graph', 'flowchart', 'sequenceDiagram', 'classDiagram', 'stateDiagram', 'mindmap', 'gitgraph']
    first_line = lines[0].strip() if lines else ""
    
    chart_type_found = False
    for chart_type in chart_types:
        if first_line.startswith(chart_type):
            chart_type_found = True
            break
    
    if not chart_type_found:
        errors.append(f"Unknown chart type: {first_line}")
    
    # Basic syntax checks
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('%%'):  # Skip empty lines and comments
            continue
            
        # Check for unmatched brackets/parentheses
        brackets = {'[': ']', '(': ')', '{': '}', '<': '>'}
        stack = []
        in_quotes = False
        for char in line:
            if char == '"':
                in_quotes = not in_quotes
            if in_quotes:
                continue

            if char in brackets:
                stack.append(brackets[char])
            elif char in brackets.values():
                if not stack or stack.pop() != char:
                    errors.append(f"Line {i}: Unmatched bracket/parenthesis")
                    break
    
    return len(errors) == 0, errors


def analyze_chart_file(file_path: Path) -> Dict:
    """Analyze a single Mermaid chart file."""
    try:
        content = file_path.read_text()
        is_valid, errors = validate_mermaid_syntax(content)
        
        # Count lines and estimate complexity
        lines = content.split('\n')
        mermaid_lines = 0
        node_count = 0
        
        in_mermaid = False
        for line in lines:
            if line.strip().startswith('```mermaid'):
                in_mermaid = True
                continue
            elif line.strip() == '```' and in_mermaid:
                in_mermaid = False
                continue
            elif in_mermaid:
                mermaid_lines += 1
                # Rough node count estimation
                if '-->' in line or '---' in line or '<--' in line:
                    node_count += 1
                elif '[' in line and ']' in line:
                    node_count += 1
        
        return {
            'path': file_path,
            'is_valid': is_valid,
            'errors': errors,
            'total_lines': len(lines),
            'mermaid_lines': mermaid_lines,
            'estimated_nodes': node_count,
            'file_size': file_path.stat().st_size,
        }
        
    except Exception as e:
        return {
            'path': file_path,
            'is_valid': False,
            'errors': [f"File error: {str(e)}"],
            'total_lines': 0,
            'mermaid_lines': 0,
            'estimated_nodes': 0,
            'file_size': 0,
        }


def generate_validation_report():
    """Generate comprehensive validation report for all Mermaid charts."""
    
    base_dir = Path(__file__).parent.parent
    chart_files = []
    
    # Find all Mermaid chart files
    for chart_path in base_dir.rglob('*.md'):
        if 'diagram' in str(chart_path) and 'mermaid' in chart_path.read_text():
            chart_files.append(chart_path)
    
    # Analyze each file
    analysis_results = []
    for chart_file in chart_files:
        result = analyze_chart_file(chart_file)
        analysis_results.append(result)
    
    # Generate report
    valid_charts = [r for r in analysis_results if r['is_valid']]
    invalid_charts = [r for r in analysis_results if not r['is_valid']]
    
    total_lines = sum(r['mermaid_lines'] for r in analysis_results)
    total_nodes = sum(r['estimated_nodes'] for r in analysis_results)
    total_size = sum(r['file_size'] for r in analysis_results)
    
    report = f"""# Mermaid Chart Validation Report

## 📊 Validation Summary

- **Total Charts:** {len(analysis_results)}
- **Valid Charts:** {len(valid_charts)} ✅
- **Invalid Charts:** {len(invalid_charts)} ❌
- **Success Rate:** {len(valid_charts) / len(analysis_results) * 100 if analysis_results else 0.0:.1f}%

## 📈 Statistics

- **Total Mermaid Lines:** {total_lines:,}
- **Estimated Nodes:** {total_nodes:,}
- **Total File Size:** {total_size/1024:.1f} KB

## 📁 Chart Categories

"""
    
    # Group by category
    categories = {}
    for result in analysis_results:
        category = result['path'].parent.name
        if category not in categories:
            categories[category] = []
        categories[category].append(result)
    
    for category, charts in categories.items():
        valid_count = len([c for c in charts if c['is_valid']])
        total_count = len(charts)
        report += f"### {category.replace('_', ' ').title()}\n"
        report += f"- **Charts:** {total_count}\n"
        report += f"- **Valid:** {valid_count}/{total_count}\n"
        report += f"- **Lines:** {sum(c['mermaid_lines'] for c in charts):,}\n\n"
    
    # Detailed results
    report += "## 📋 Detailed Validation Results\n\n"
    
    for result in sorted(analysis_results, key=lambda x: str(x['path'])):
        status = "✅" if result['is_valid'] else "❌"
        rel_path = result['path'].relative_to(base_dir)
        report += f"### {status} `{rel_path}`\n"
        report += f"- **Lines:** {result['mermaid_lines']} Mermaid, {result['total_lines']} total\n"
        report += f"- **Nodes:** ~{result['estimated_nodes']}\n"
        report += f"- **Size:** {result['file_size']} bytes\n"
        
        if result['errors']:
            report += f"- **Errors:**\n"
            for error in result['errors']:
                report += f"  - {error}\n"
        report += "\n"
    
    # Recommendations
    report += """## 🔧 Recommendations

### For Valid Charts
1. **Rendering:** All valid charts can be rendered with Mermaid CLI or online editor
2. **Integration:** Ready for inclusion in documentation and academic papers
3. **Maintenance:** Update charts when corresponding code changes

### For Invalid Charts
1. **Review:** Check syntax errors and fix formatting issues
2. **Test:** Validate fixes using online Mermaid editor
3. **Regenerate:** Run chart generators after fixes

### Quality Assurance
1. **Automated Validation:** Include this script in CI/CD pipeline
2. **Regular Updates:** Re-run validation after major code changes
3. **Documentation:** Keep chart documentation up to date

## 🛠️ Usage

```bash
# Validate all charts
python tools/validate_mermaid_charts.py

# Render specific chart
mmdc -i chart.md -o chart.png

# Batch render all charts
find documentation/diagrams -name "*.md" -exec mmdc -i {} -o {}.png \\;
```

---

*Generated on {datetime.fromtimestamp(Path(__file__).stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')} - Repository validation complete*
"""
    
    # Save report
    report_path = base_dir / "documentation" / "diagrams" / "MERMAID_VALIDATION_REPORT.md"
    report_path.write_text(report)
    
    print(f"✅ Validation complete!")
    print(f"   📊 Total charts: {len(analysis_results)}")
    print(f"   ✅ Valid: {len(valid_charts)}")
    print(f"   ❌ Invalid: {len(invalid_charts)}")
    print(f"   📄 Report: {report_path}")
    
    return len(invalid_charts) == 0


if __name__ == "__main__":
    success = generate_validation_report()
    exit(0 if success else 1)
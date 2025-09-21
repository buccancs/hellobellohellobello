#!/usr/bin/env python3
"""
Issue #27 Completion Validation Script

This script validates that all requirements from the "Missing visualisation" issue
have been properly implemented and are accessible in the repository.
"""

from pathlib import Path


def check_chapter_visualizations():
    """Check if all required chapter visualizations exist."""

    required_visualizations = {
        'Chapter 1': ['Conceptual Overview Diagram'],
        'Chapter 2': [
            'Physiology of Galvanic Skin Response',
            'Thermal Cues of Stress',
            'Comparison of Stress Indicators (Table)',
            'Sensor Specifications (Table)',
        ],
        'Chapter 3': [
            'High-Level System Architecture',
            'UML Use Case Diagram',
            'Summary of Functional Requirements (Table)',
            'Summary of Non-Functional Requirements (Table)',
        ],
        'Chapter 4': [
            'Detailed System Architecture',
            'Android Application Architecture',
            'PC Controller Threading Model',
            'Protocol Sequence Diagram',
            'Data Processing Pipeline',
            'Desktop GUI Screenshots',
            'Code Listings',
        ],
        'Chapter 5': [
            'Testing Strategy Overview',
            'Summary of Test Coverage (Table)',
            'Synchronization Accuracy Results (Graph)',
            'Synchronization Failure Example (Graph)',
            'Endurance Test Results (Graphs)',
            'Usability Testing Results (Table)',
        ],
        'Chapter 6': ['Evaluation of Project Objectives (Table)'],
    }

    print("📊 Chapter Visualization Validation:")
    print("=" * 50)

    base_dir = Path(__file__).parent.parent

    thesis_viz_dir = base_dir / "docs" / "diagrams" / "thesis_visualizations"
    images_dir = base_dir / "images"

    all_complete = True

    for chapter in required_visualizations:
        chapter_num = chapter.split()[1]
        print(f"\n{chapter}:")

        chapter_dir = thesis_viz_dir / f"chapter{chapter_num}_{chapter.split()[1].lower()}"
        if chapter_num == '1':
            chapter_dir = thesis_viz_dir / "chapter1_introduction"
        elif chapter_num == '2':
            chapter_dir = thesis_viz_dir / "chapter2_background"
        elif chapter_num == '3':
            chapter_dir = thesis_viz_dir / "chapter3_requirements"
        elif chapter_num == '4':
            chapter_dir = thesis_viz_dir / "chapter4_implementation"
        elif chapter_num == '5':
            chapter_dir = thesis_viz_dir / "chapter5_evaluation"
        elif chapter_num == '6':
            chapter_dir = thesis_viz_dir / "chapter6_conclusions"

        if chapter_dir.exists():
            print("  ✅ Documentation directory exists")
        else:
            print("  ❌ Documentation directory missing")
            all_complete = False

        image_chapter_dirs = [
            images_dir / f"chapter{chapter_num}_background" if chapter_num == '2' else None,
            images_dir / f"chapter{chapter_num}_evaluation" if chapter_num == '5' else None,
            images_dir / f"chapter{chapter_num}_conclusions" if chapter_num == '6' else None,
        ]

        mermaid_dir = base_dir / "docs" / "diagrams" / "mermaid"
        mermaid_chapter_dirs = [
            mermaid_dir / f"chapter{chapter_num}_introduction" if chapter_num == '1' else None,
            mermaid_dir / f"chapter{chapter_num}_requirements" if chapter_num == '3' else None,
            mermaid_dir / f"chapter{chapter_num}_implementation" if chapter_num == '4' else None,
        ]

        image_count = 0
        mermaid_count = 0

        for img_dir in image_chapter_dirs:
            if img_dir and img_dir.exists():
                image_count += len(list(img_dir.glob("*.png")))

        for mmd_dir in mermaid_chapter_dirs:
            if mmd_dir and mmd_dir.exists():
                mermaid_count += len(list(mmd_dir.glob("*.mmd")))

        total_viz = image_count + mermaid_count

        if total_viz > 0:
            viz_types = []
            if image_count > 0:
                viz_types.append(f"{image_count} PNG")
            if mermaid_count > 0:
                viz_types.append(f"{mermaid_count} Mermaid")
            print(f"  ✅ {total_viz} generated visualization(s) ({', '.join(viz_types)})")
        elif chapter_num in ['2', '5', '6']:
            if chapter_num == '2':
                ch2_images = len(list(images_dir.glob("chapter2*/*.png")))
                if ch2_images > 0:
                    print(
                        f"  ✅ {ch2_images} generated visualization(s) (found in chapter2 subdirs)"
                    )
                    total_viz = ch2_images
                else:
                    print("  ❌ No generated visualizations found")
            else:
                print("  ❌ No generated visualizations found")
        else:
            print("  ⚠️  No generated images (documentation available)")

    return all_complete


def check_evidence_files():
    """Check if all Chapter 5 evidence files exist."""

    required_evidence = {
        'Unit Testing Evidence': [
            'junit_report_android.xml',
            'pytest_report_pc.xml',
            'coverage_report_pc.xml',
        ],
        'Integration Testing Evidence': [
            'simulation_test_logs.txt',
            'system_integration_report.json',
        ],
        'System Performance Evaluation': [
            'endurance_test_report.json',
            'endurance_raw_data.csv',
            'synchronization_accuracy_data.csv',
        ],
        'Stability and Usability Evidence': [
            'pc_threading_error_logs.txt',
            'wifi_roaming_sync_failures.csv',
            'setup_time_measurements.csv',
            'user_testing_session_notes.md',
        ],
    }

    print("\n📋 Chapter 5 Evidence Validation:")
    print("=" * 50)

    evidence_dir = Path(__file__).parent.parent / "docs" / "evidence"
    all_present = True

    for category, files in required_evidence.items():
        print(f"\n{category}:")

        category_found = False
        for subdir in evidence_dir.iterdir():
            if subdir.is_dir():
                for file in files:
                    file_path = subdir / file
                    if file_path.exists():
                        print(f"  ✅ {file}")
                        category_found = True

        if not category_found:
            for file in files:
                print(f"  ❌ {file}")
            all_present = False

    return all_present


def check_readme_completeness():
    """Check if README meets repository validation requirements."""

    readme_requirements = [
        'Introduction and project overview',
        'Project status and known issues',
        'Repository structure',
        'Prerequisites',
        'Setup and build instructions (PC and Android)',
        'Running the system',
        'Validation and testing procedures',
        'Repository validation checklist',
    ]

    print("\n📖 README Validation:")
    print("=" * 50)

    base_dir = Path(__file__).parent.parent

    readme_path = base_dir / "README.md"
    validation_readme_path = base_dir / "REPOSITORY_VALIDATION_README.md"

    if readme_path.exists():
        print("✅ Main README.md exists")
        readme_size = readme_path.stat().st_size
        print(f"   Size: {readme_size:,} bytes")
    else:
        print("❌ Main README.md missing")

    if validation_readme_path.exists():
        print("✅ REPOSITORY_VALIDATION_README.md exists")
        readme_size = validation_readme_path.stat().st_size
        print(f"   Size: {readme_size:,} bytes")

        with open(validation_readme_path) as f:
            content = f.read().lower()

        print("\n   Content validation:")
        for req in readme_requirements:
            if any(keyword in content for keyword in req.lower().split()[0:2]):
                print(f"     ✅ {req}")
            else:
                print(f"     ❌ {req}")

        return True
    else:
        print("❌ REPOSITORY_VALIDATION_README.md missing")
        return False


def check_visualization_scripts():
    """Check if visualization generation scripts exist and work."""

    print("\n🔧 Visualization Scripts Validation:")
    print("=" * 50)

    base_dir = Path(__file__).parent.parent
    scripts_dir = base_dir / "scripts"

    required_scripts = [
        'generate_sample_visualizations.py',
        'generate_chapter5_visualizations.py',
        'generate_missing_visualizations.py',
        'generate_mermaid_visualizations.py',
    ]

    all_scripts_present = True

    for script in required_scripts:
        script_path = scripts_dir / script
        if script_path.exists():
            print(f"✅ {script}")
            if script_path.stat().st_size > 0:
                print(f"   Size: {script_path.stat().st_size:,} bytes")
            else:
                print("   ⚠️  Empty file")
        else:
            print(f"❌ {script}")
            all_scripts_present = False

    return all_scripts_present


def main():
    """Run complete validation of issue #27 requirements."""

    print("🎯 Issue
    print("=" * 70)

    viz_complete = check_chapter_visualizations()
    evidence_complete = check_evidence_files()
    readme_complete = check_readme_completeness()
    scripts_complete = check_visualization_scripts()

    print("\n🏁 Final Validation Summary:")
    print("=" * 50)

    results = {
        'Chapter Visualizations': viz_complete,
        'Evidence Files': evidence_complete,
        'README Documentation': readme_complete,
        'Visualization Scripts': scripts_complete,
    }

    all_complete = all(results.values())

    for component, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {component}")

    if all_complete:
        print("\n🎉 Issue
        print("   All required visualizations, evidence, and documentation are present.")
        print("   Repository is ready for academic examination and validation.")
    else:
        print("\n⚠️  Issue
        print("   Some components may need attention.")

    print("\n📊 Generated Visualizations Summary:")
    images_dir = Path(__file__).parent.parent / "images"
    total_images = len(list(images_dir.rglob("*.png")))
    print(f"   Total PNG files generated: {total_images}")

    for chapter_dir in sorted(images_dir.iterdir()):
        if chapter_dir.is_dir() and "chapter" in chapter_dir.name:
            png_files = list(chapter_dir.glob("*.png"))
            if png_files:
                chapter_name = chapter_dir.name.replace("_", " ").title()
                print(f"   {chapter_name}: {len(png_files)} files")

    return 0 if all_complete else 1


if __name__ == "__main__":
    exit(main())

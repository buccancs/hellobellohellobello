plugins {
    base
}

// Cross-platform support for different operating systems
val isWindows = org.gradle.internal.os.OperatingSystem.current().isWindows()
val venvDir = project.layout.projectDirectory.dir(".venv")
val venvPython = if (isWindows) {
    venvDir.file("Scripts/python.exe")
} else {
    venvDir.file("bin/python")
}

// Determine shell command based on platform
fun shellCommand(vararg cmd: String): List<String> {
    return if (isWindows) {
        listOf("cmd", "/c") + cmd
    } else {
        listOf("bash", "-c") + cmd
    }
}

// Create a Python virtual environment under pc_controller/.venv
val setupVenv by tasks.registering(Exec::class) {
    group = "python"
    description = "Create a Python virtual environment for pc_controller"
    val pythonCmd = if (isWindows) "python" else "python3"
    commandLine(shellCommand("$pythonCmd -m venv .venv"))
    workingDir = project.projectDir
    // Only create if missing - using a simpler check
    onlyIf { 
        val venvExists = project.projectDir.resolve(".venv").exists()
        !venvExists
    }
}

// Install Python dependencies into the venv, including pytest and pyinstaller
val installRequirements by tasks.registering(Exec::class) {
    group = "python"
    description = "Install Python dependencies into the venv (requirements + pytest + pyinstaller)"
    dependsOn(setupVenv)
    workingDir = project.projectDir
    
    doFirst {
        val venvPythonPath = if (isWindows) {
            project.projectDir.resolve(".venv/Scripts/python.exe")
        } else {
            project.projectDir.resolve(".venv/bin/python")
        }
        
        val pythonExe = if (venvPythonPath.exists()) {
            venvPythonPath.absolutePath
        } else {
            if (isWindows) "python" else "python3"
        }
        val cmd = "$pythonExe -m pip install -U pip && $pythonExe -m pip install -r requirements.txt && $pythonExe -m pip install pytest pyinstaller"
        commandLine(shellCommand(cmd))
    }
}

// NOTE: pytest is run from root project level, not here, to use repo-level pytest.ini
// The root build.gradle.kts has a pyTest task that runs from the repo root

// Build a standalone executable using PyInstaller
val pyInstaller by tasks.registering(Exec::class) {
    group = "build"
    description = "Build pc_controller executable using PyInstaller"
    dependsOn(installRequirements)
    workingDir = project.projectDir
    
    doFirst {
        val venvPythonPath = if (isWindows) {
            project.projectDir.resolve(".venv/Scripts/python.exe")
        } else {
            project.projectDir.resolve(".venv/bin/python")
        }
        
        val pythonExe = if (venvPythonPath.exists()) {
            venvPythonPath.absolutePath
        } else {
            if (isWindows) "python" else "python3"
        }
        val entry = File(project.projectDir, "src/main.py").absolutePath
        val distDir = project.layout.buildDirectory.get().asFile.resolve("dist").absolutePath
        val cmd = "$pythonExe -m PyInstaller --noconfirm --clean -F --name pc_controller --distpath $distDir $entry"
        commandLine(shellCommand(cmd))
    }
}

// Make root :build depend on this artifact if desired
tasks.register("assemblePcController") {
    group = "build"
    description = "Assemble pc_controller executable"
    dependsOn(pyInstaller)
}

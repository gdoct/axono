import os
from collections import deque
from pathlib import Path


def detect_project_type_from_folderpath(folderpath):
    """
    Detect the project type in a folder path by looking for certain files or extension types

    Args:
        folderpath (str): The folder path to analyze. if the path contains only folders, the first found project type in any of its subfolder will be returned

    Returns:
        str: The detected project type, or 'unknown' if no project type is found. example ["java", "python", "go", "make", "cmake", "nodejs", "rust", "ruby", "php", "dotnet", "flutter", "dart", "android", "ios"]
    """
    # strategy:
    # 1. check for specific files that indicate a project type (e.g., package.json for nodejs, pom.xml for java, etc.)
    # 2. check for specific file extensions that indicate a project type (e.g., .java for java, .py for python, etc.)
    # 3. If a 'src' directory exists, it might indicate a source code folder for a project, so check that folder first
    # 4. Check all the other subfolders in alphabetical order. we search breadth first and return the first found project type. this is
    #    because some projects have multiple project types (e.g., a monorepo with both java and nodejs projects), and we want to return
    #    the most relevant one based on the folder structure.
    # 5. if no specific files or extensions are found, return 'unknown'
    # Define project type indicators
    PROJECT_INDICATORS = {
        "java": ["pom.xml", "build.gradle", ".java"],
        "python": ["setup.py", "pyproject.toml", "requirements.txt", ".py"],
        "nodejs": ["package.json", "yarn.lock", ".js"],
        "go": ["go.mod", "go.sum", ".go"],
        "rust": ["Cargo.toml", ".rs"],
        "ruby": ["Gemfile", ".rb"],
        "php": ["composer.json", ".php"],
        "dotnet": [".csproj", ".vbproj", ".sln"],
        "flutter": ["pubspec.yaml", ".dart"],
        "dart": ["pubspec.yaml", ".dart"],
        "android": ["AndroidManifest.xml", "build.gradle"],
        "ios": ["Podfile", ".xcodeproj"],
        "cmake": ["CMakeLists.txt"],
        "make": ["Makefile"],
        "c": ["CMakeLists.txt", "Makefile", ".c", ".h"],
        "cpp": ["CMakeLists.txt", ".cpp", ".cc", ".cxx", ".hpp", ".h"],
        "typescript": ["tsconfig.json", ".ts", ".tsx"],
        "react": [".jsx", ".tsx"],
    }

    def check_folder(folder):
        """Check if a folder contains indicators of a project type"""
        try:
            items = os.listdir(folder)
        except (OSError, PermissionError):
            return None

        # Check files first
        for project_type, indicators in PROJECT_INDICATORS.items():
            for indicator in indicators:
                if not indicator.startswith("."):  # It's a file
                    if indicator in items:
                        return project_type

        # Check extensions
        for project_type, indicators in PROJECT_INDICATORS.items():
            for indicator in indicators:
                if indicator.startswith("."):  # It's an extension
                    if any(
                        f.endswith(indicator)
                        for f in items
                        if os.path.isfile(os.path.join(folder, f))
                    ):
                        return project_type

        return None

    # Check the given folder first
    result = check_folder(folderpath)
    if result:
        return result

    # Breadth-first search: prioritize 'src' folder, then alphabetical order
    queue = deque([folderpath])
    visited = {folderpath}

    while queue:
        current = queue.popleft()

        try:
            items = sorted(os.listdir(current))
        except (OSError, PermissionError):
            continue

        # Prioritize 'src' folder
        if "src" in items:
            src_path = os.path.join(current, "src")
            if os.path.isdir(src_path):
                result = check_folder(src_path)
                if result:
                    return result
                if src_path not in visited:
                    visited.add(src_path)
                    queue.append(src_path)

        # Check other directories in alphabetical order
        for item in items:
            item_path = os.path.join(current, item)
            if (
                os.path.isdir(item_path)
                and item not in ["src"]
                and item_path not in visited
            ):
                visited.add(item_path)
                result = check_folder(item_path)
                if result:
                    return result
                queue.append(item_path)

    return "unknown"
    pass

import logging
import os
import re
import argparse
from pathlib import Path

_logger = logging.getLogger(__name__)

# ===================== 配置区 =====================
# 1. 目录忽略正则（匹配文件夹名）
IGNORE_DIR_PATTERNS = [
    r"^\.cache$",
    r"^\.git$",
    r"^\.vscode$",
    r"^\.idea$",
    r"^__pycache__$",
    r"^node_modules$",
    r"^venv$",
    r"^env$",
    r"^dist$",
    r"^build",
    r"^legacy$",
    r"^external$",
    r"^vcpkg_installed$",
    r"^cases$",
    r"^results$",
    r"^\.",
]

# 2. 文件忽略正则（匹配文件名）
IGNORE_FILE_PATTERNS = [
    r"^\.DS_Store$",
    r"^package-lock\.json$",
    r"^yarn\.lock$",
    r"^poetry\.lock$",
    r"^pnpm-lock\.yaml$",
    r"^favicon\.ico$",
]

# 3. 允许读取的文本文件后缀
ALLOWED_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".c",
    ".cpp",
    ".java",
    ".go",
    ".rs",
    ".php",
    ".rb",
    ".h",
    ".hpp",
    ".sql",
    ".yaml",
    ".toml",
    ".yml",
    ".json",
    ".md",
    ".txt",
    ".html",
    ".css",
    ".sh",
    ".ini",
    ".conf",
    ".cmake",
}
# ==================================================

# 预编译正则表达式（提升性能）
COMPILED_DIR_IGNORES = [re.compile(p) for p in IGNORE_DIR_PATTERNS]
COMPILED_FILE_IGNORES = [re.compile(p) for p in IGNORE_FILE_PATTERNS]


def is_ignored_dir(name: str) -> bool:
    """判断目录是否需要忽略（正则匹配）"""
    return any(pattern.match(name) for pattern in COMPILED_DIR_IGNORES)


def is_ignored_file(name: str) -> bool:
    """判断文件是否需要忽略（正则匹配）"""
    return any(pattern.match(name) for pattern in COMPILED_FILE_IGNORES)


def is_text_file(file_path: Path) -> bool:
    """判断是否为应读取的文本文件"""
    return file_path.suffix.lower() in ALLOWED_EXTENSIONS


def generate_tree(root_dir: Path, prefix: str = "") -> str:
    """递归生成目录树结构字符串"""
    tree_str = ""
    paths = sorted(list(root_dir.iterdir()), key=lambda x: (x.is_file(), x.name))

    # 过滤忽略的目录/文件
    filtered = []
    for p in paths:
        if p.is_dir() and is_ignored_dir(p.name):
            continue
        if p.is_file() and is_ignored_file(p.name):
            continue
        filtered.append(p)

    for i, path in enumerate(filtered):
        connector = "└── " if i == len(filtered) - 1 else "├── "
        tree_str += f"{prefix}{connector}{path.name}\n"
        if path.is_dir():
            extension = "    " if i == len(filtered) - 1 else "│   "
            tree_str += generate_tree(path, prefix + extension)
    return tree_str


def process_repository(repo_path: str, output_file: str):
    root_path = Path(repo_path).resolve()

    with open(output_file, "w", encoding="utf-8") as f:
        # 1. 写入项目标题
        f.write(f"# Project Source Code: {root_path.name}\n\n")

        # 2. 写入目录树
        f.write("## Directory Structure\n")
        f.write("```text\n")
        f.write(".\n")
        f.write(generate_tree(root_path))
        f.write("```\n\n")

        # 3. 递归遍历并写入文件内容
        f.write("## File Contents\n\n")
        for current_path, dirs, files in os.walk(root_path):
            # 跳过忽略的目录（os.walk 会自动不再进入）
            dirs[:] = [d for d in dirs if not is_ignored_dir(d)]

            for file in files:
                if is_ignored_file(file):
                    continue

                file_path = Path(current_path) / file
                if is_text_file(file_path):
                    relative_path = file_path.relative_to(root_path)

                    f.write(f"### File: {relative_path}\n")
                    lang = file_path.suffix.lstrip(".")
                    f.write(f"```{lang}\n")

                    try:
                        with open(
                            file_path, "r", encoding="utf-8", errors="replace"
                        ) as code_f:
                            f.write(code_f.read())
                    except Exception as e:
                        f.write(f"/* Error reading file: {e} */")

                    f.write("\n```\n\n")

    _logger.info(f"处理完成！输出文件已保存至: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将代码仓库整理为 AI 友好的 Markdown 格式"
    )
    parser.add_argument("input_folder", help="要读取的文件夹路径")
    parser.add_argument("output_file", help="输出的文本文件名 (例如 output.md)")

    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        _logger.error(f"错误: 文件夹 '{args.input_folder}' 不存在")
    else:
        process_repository(args.input_folder, args.output_file)

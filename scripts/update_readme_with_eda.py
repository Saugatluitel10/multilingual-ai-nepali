"""
Script to update README.md with latest EDA outputs
"""

from pathlib import Path
import re

EDA_SECTION_HEADER = '## Exploratory Data Analysis (Auto-Generated)'
EDA_OUTPUT_DIR = Path('notebooks/eda_outputs')
README_PATH = Path('README.md')


def get_eda_figures_markdown():
    figs = sorted(EDA_OUTPUT_DIR.glob('*.png'))
    md = ''
    for fig in figs:
        md += f'![EDA Output]({fig.as_posix()})\n'
    return md


def update_readme_with_eda():
    if not README_PATH.exists():
        print(f"❌ README.md not found at {README_PATH}")
        return
    readme_text = README_PATH.read_text(encoding='utf-8')

    # Remove old EDA section if present
    pattern = rf'{EDA_SECTION_HEADER}.*?(?=^## |\Z)'  # Match section until next header or end of file
    new_eda_md = f'{EDA_SECTION_HEADER}\n\n' + get_eda_figures_markdown() + '\n'
    if re.search(pattern, readme_text, flags=re.DOTALL | re.MULTILINE):
        readme_text = re.sub(pattern, new_eda_md, readme_text, flags=re.DOTALL | re.MULTILINE)
    else:
        # Append at the end
        readme_text += '\n' + new_eda_md

    README_PATH.write_text(readme_text, encoding='utf-8')
    print(f"✅ Updated {README_PATH} with EDA outputs.")


if __name__ == "__main__":
    update_readme_with_eda()

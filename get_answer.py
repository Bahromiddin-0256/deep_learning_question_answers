import sys
import os
import re


def get_answer(question_id):
    # List all answer files in the current directory
    # We look for files matching the pattern answers_XXX-YYY.md
    files = [f for f in os.listdir('.') if f.startswith('answers_') and f.endswith('.md')]
    target_file = None

    # Iterate through files to find the one covering the question_id range
    for f in files:
        # Extract the start and end range from the filename
        match = re.match(r'answers_(\d+)-(\d+)\.md', f)
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
            # Check if the requested ID falls within this file's range
            if start <= question_id <= end:
                target_file = f
                break

    if not target_file:
        return f"Answer for question {question_id} not found (no matching file)."

    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Regex to find the specific question block.
        # It looks for "### <id>." at the start of a line,
        # and captures everything until the next "###" at the start of a line or end of file.
        # (?m) enables multiline matching so ^ matches start of lines.
        pattern = re.compile(rf'(^###\s*{question_id}\..*?)(?=^###|\Z)', re.DOTALL | re.MULTILINE)
        match = pattern.search(content)

        if match:
            return match.group(1).strip()
        else:
            return f"Answer for question {question_id} not found in {target_file}."

    except Exception as e:
        return f"Error reading file {target_file}: {e}"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_answer.py <question_id>")
        sys.exit(1)

    try:
        q_id = int(sys.argv[1])
        print(get_answer(q_id))
    except ValueError:
        print("Invalid question ID. Please provide a number.")

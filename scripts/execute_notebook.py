"""Execute a notebook in a fresh Python process and retain IPython outputs.

Uses IPython without network sockets. With no destination, runs the notebook
without saving an extra copy. An optional destination is written only after all
cells succeed. Run once per notebook so each run has a fresh namespace.
"""

import argparse
import json
import os
from pathlib import Path
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.utils.capture import capture_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path, nargs='?',
                        help='Optional executed copy; omit to create no extra notebook.')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    source = args.source.resolve()
    destination = args.destination.resolve() if args.destination else None
    notebook = json.loads(source.read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    os.chdir(project)
    shell = TerminalInteractiveShell.instance()
    shell.display_formatter.active_types = ['text/plain', 'text/html', 'image/png', 'image/svg+xml']
    # The inline backend captures figures as standard notebook MIME output.
    shell.run_line_magic('matplotlib', 'inline')
    count = 0
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        count += 1
        print(f'Running cell {i}...', flush=True)
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            result = shell.run_cell(''.join(cell['source']), store_history=False)
        if result.error_before_exec or result.error_in_exec:
            print(captured.stdout, flush=True)
            print(captured.stderr, flush=True)
            raise RuntimeError(f'Notebook cell {i} failed: {result.error_before_exec or result.error_in_exec}')
        cell['execution_count'] = count
        for name, text in [('stdout', captured.stdout), ('stderr', captured.stderr)]:
            if text:
                cell['outputs'].append({'output_type': 'stream', 'name': name, 'text': text})
        for output in captured.outputs:
            cell['outputs'].append({'output_type': 'display_data', 'data': output.data,
                                    'metadata': output.metadata or {}})
        print(f'Cell {i} passed ({len(cell["outputs"])} outputs)', flush=True)
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        print(f'Saved executed notebook: {destination}', flush=True)
    else:
        print('All cells passed; no extra notebook saved.', flush=True)


if __name__ == '__main__':
    main()

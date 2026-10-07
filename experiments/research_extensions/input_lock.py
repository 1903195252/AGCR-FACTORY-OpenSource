"""Freeze matched-study inputs before comparing supervision coefficients."""
import argparse
import hashlib
import json
from pathlib import Path

KEYS = ('parent', 'history_overlay', 'panel', 'paths', 'families')
SCHEMA = 'paired-study-input-lock-1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(lock_path, inputs):
    lock = json.loads(Path(lock_path).read_text())
    if lock.get('schema') != SCHEMA or set(lock.get('sha256', {})) != set(KEYS):
        raise ValueError('Incomplete matched-study input lock')
    if any(sha256(inputs[key]) != lock['sha256'][key] for key in KEYS):
        raise ValueError('Matched-study inputs differ from the frozen lock')
    return lock


def freeze(output, inputs):
    output = Path(output)
    lock = dict(schema=SCHEMA, sha256={key: sha256(inputs[key]) for key in KEYS})
    if output.exists():
        if json.loads(output.read_text()) != lock:
            raise FileExistsError('Refusing to replace a different study input lock')
        return lock
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(lock, indent=2) + '\n')
    return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('parent', 'history-overlay', 'panel', 'paths', 'output'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    inputs = {key: getattr(args, key) for key in KEYS if key != 'families'}
    inputs['families'] = json.loads(Path(args.paths).read_text())['families']
    freeze(args.output, inputs)


if __name__ == '__main__':
    main()

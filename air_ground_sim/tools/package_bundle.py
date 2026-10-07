"""Produce one self-contained tar.gz, excluding development outputs and caches."""
import argparse
import hashlib
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args()
    destination=args.output.resolve()
    if destination.is_relative_to(ROOT):raise ValueError('Archive must be outside the bundle')
    if destination.exists():raise FileExistsError(destination)
    def keep(info):
        parts=Path(info.name).parts
        if any(p in ('outputs','__pycache__','.venv','.pytest_cache') or p.endswith('.egg-info') for p in parts):return None
        if info.name.endswith(('.pyc','.tmp')):return None
        if info.issym():raise ValueError('Unexpected external/relative symlink: '+info.name)
        return info
    with tarfile.open(destination,'w:gz',compresslevel=1,dereference=False) as archive:
        archive.add(ROOT,arcname=ROOT.name,filter=keep)
    digest=hashlib.sha256()
    with destination.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    destination.with_suffix(destination.suffix+'.sha256').write_text(digest.hexdigest()+'  '+destination.name+'\n')
    print(destination,destination.stat().st_size,digest.hexdigest())


if __name__=='__main__':main()

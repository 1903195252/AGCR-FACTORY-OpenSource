#!/usr/bin/env python3
"""Verify the release asset archive and restore its two payload directories."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify(path, entry):
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
        raise ValueError("Size or SHA-256 mismatch: " + path.name)


def local_file(directory, name):
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("Manifest filenames must be plain basenames")
    return directory / name


def validate_members(archive, destination):
    members = archive.getmembers()
    names = set()
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe archive path")
        parts = path.parts
        if parts == ("air_ground_sim",) and member.isdir():
            continue
        if len(parts) < 2 or parts[:2] not in (
            ("air_ground_sim", "assets"), ("air_ground_sim", "authoring")
        ):
            raise ValueError("Unexpected payload member: " + member.name)
        if member.issym() or not (member.isfile() or member.isdir() or member.islnk()):
            raise ValueError("Unsupported archive member: " + member.name)
        if member.name in names:
            raise ValueError("Duplicate archive path: " + member.name)
        names.add(member.name)
        target = destination.joinpath(*parts)
        for parent in [target, *target.parents]:
            if parent == destination.parent:
                break
            if parent.is_symlink():
                raise ValueError("Destination contains a symbolic link")
        if target.exists() and not (member.isdir() and target.is_dir()):
            raise FileExistsError("Refusing to overwrite: " + str(target))
    for member in members:
        if member.islnk():
            link = PurePosixPath(member.linkname)
            if link.is_absolute() or ".." in link.parts or member.linkname not in names:
                raise ValueError("Unsafe hard-link target")
    return members


def restore(downloads, destination):
    manifest = json.loads((downloads / "release_manifest.json").read_text())
    if manifest.get("schema") != "agcr-source-release-1":
        raise ValueError("Unsupported release manifest")
    entry = manifest["payloads"]["simulation_assets"]
    archive_path = local_file(downloads, entry["file"])
    temporary = None
    try:
        if not archive_path.exists():
            parts = entry.get("parts", [])
            if not parts:
                raise FileNotFoundError(archive_path)
            temporary = tempfile.TemporaryDirectory(prefix="asset-restore-", dir=downloads)
            archive_path = Path(temporary.name) / entry["file"]
            with archive_path.open("wb") as merged:
                for part in parts:
                    source = local_file(downloads, part["file"])
                    verify(source, part)
                    with source.open("rb") as stream:
                        shutil.copyfileobj(stream, merged, 8 * 1024 * 1024)
        verify(archive_path, entry)
        destination.mkdir(parents=True, exist_ok=True)
        destination = destination.resolve()
        with tarfile.open(archive_path, "r:gz") as archive:
            members = validate_members(archive, destination)
            # All paths, hard links and destination parents are checked before extraction.
            archive.extractall(destination, members=members)
        print("Verified and restored simulation assets to " + str(destination))
    finally:
        if temporary is not None:
            temporary.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path("."))
    args = parser.parse_args()
    restore(args.downloads.resolve(), args.destination)


if __name__ == "__main__":
    main()

"""Download and verify the complete original archive; extract only labeled train.txt."""
import argparse
import hashlib
import shutil
import subprocess
import tarfile
from pathlib import Path

URL = 'https://criteostorage.blob.core.windows.net/criteo-research-datasets/kaggle-display-advertising-challenge-dataset.tar.gz'
SIZE = 4_576_820_670
MD5 = 'df9b1b3766d9ff91d5ca3eb3d23bed27'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    archive = args.directory.resolve() / 'dac.tar.gz'
    if archive.exists() and archive.stat().st_size == SIZE and not archive.with_suffix('.gz.aria2').exists():
        pass  # Reuse only after the checksum verification below.
    elif shutil.which('aria2c'):
        subprocess.run(['aria2c', '-x', '16', '-s', '16', '-k', '4M', '--file-allocation=none',
            '--continue=true', '-d', str(archive.parent), '-o', archive.name, URL], check=True)
    else:
        subprocess.run(['curl', '--fail', '--location', '--retry', '5', '--continue-at', '-',
                        '--output', str(archive), URL], check=True)
    digest = hashlib.md5()  # Archive identity supplied by the public release mirror, not authentication.
    with archive.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if archive.stat().st_size != SIZE or digest.hexdigest() != MD5:
        raise ValueError('complete archive size/checksum mismatch')
    destination = archive.parent / 'train.txt'
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_suffix('.partial')
    with tarfile.open(archive) as stream:
        entries = [member for member in stream.getmembers() if member.name in ('train.txt', './train.txt')]
        if len(entries) != 1 or not entries[0].isfile():
            raise ValueError('archive must contain one regular train.txt')
        with stream.extractfile(entries[0]) as source, temporary.open('wb') as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    temporary.rename(destination)
    print(destination)


if __name__ == '__main__':
    main()

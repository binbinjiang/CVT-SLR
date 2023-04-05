import subprocess
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)


def pack_code(git_root, run_dir):
    if os.path.isdir(f"{git_root}/.git"):
        subprocess.run(
            ['git', 'archive', '-o', f"{run_dir}/code.tar.gz", 'HEAD'],
            check=True,
        )
        diff_process = subprocess.run(
            ['git', 'diff', 'HEAD'],
            check=True, stdout=subprocess.PIPE, text=True,
        )
        if diff_process.stdout:
            logger.warning('Working tree is dirty. Patch:\n%s', diff_process.stdout)
            with open(f"{run_dir}/dirty.patch", 'w') as f:
                f.write(diff_process.stdout)
    else:
        logger.warning('.git does not exist in current dir')

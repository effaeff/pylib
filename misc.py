"""Miscellaneous methods"""

import os
import hashlib
import functools
from pathlib import Path
from tempfile import mkstemp
from shutil import move
import gdown


def to_local_dir(filehandle):  # needs __file__ from caller
    """Change to local workspace for runtime stuff"""
    os.chdir(os.path.dirname(os.path.realpath(filehandle)))

def gen_dirs(directories):
    """Generate directories, if they not already exist"""
    for directory in directories:
        if not os.path.exists(directory):
            try:
                Path(directory).mkdir(parents=True)
            except OSError:
                print(f"Error: Creation of directory {directory} failed.")

def md5sum(filename, blocksize=65536):
    """Generate md5 sum"""
    hsh = hashlib.md5()
    with open(filename, 'rb') as f_handle:
        for block in iter(lambda: f_handle.read(blocksize), b''):
            hsh.update(block)
    return hsh.hexdigest()


def cached_download(url, path, md5=None, quiet=False):
    """
    Download url in path using md5.

    References:
        - https://github.com/wkentaro/fcn
    """
    def check_md5(path, md5):
        print('[{:s}] Checking md5 ({:s})'.format(path, md5))
        return md5sum(path) == md5

    if os.path.exists(path) and not md5:
        print('[{:s}] File exists ({:s})'.format(path, md5sum(path)))
    elif os.path.exists(path) and md5 and check_md5(path, md5):
        pass
    else:
        dirpath = os.path.dirname(path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        gdown.download(url, path, quiet=quiet)

    return path

def replace_line_in_file(pathname, pattern, subst):
    """Replace pattern in line with subst"""
    # Create temp file
    f_h, abs_path = mkstemp()
    with os.fdopen(f_h, 'w') as new_file:
        with open(pathname) as old_file:
            for line in old_file:
                if pattern in line:
                    new_file.write(subst)
                else:
                    new_file.write(line)
    # Remove original file
    os.remove(pathname)
    # Move new file
    move(abs_path, pathname)

def lazy_property(function):
    """Decorating function as lazy loading property"""
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        """The decorator function"""
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

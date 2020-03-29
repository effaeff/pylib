"""Miscellaneous methods"""

import os
import functools
from tempfile import mkstemp
from shutil import move


def to_local_dir(filehandle):  # needs __file__ from caller
    """Change to local workspace for runtime stuff"""
    os.chdir(os.path.dirname(os.path.realpath(filehandle)))

def gen_dirs(directories):
    """Generate directories, if they not already exist"""
    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except OSError:
                print(f"Error: Creation of directory {directory} failed.")

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


"""Operations on TDMS files"""

import numpy as np
from nptdms import TdmsFile, TdmsWriter, ChannelObject


def read_tdms(pathname, keys, groupname):
    """
    Read data from TDMS file

    Args:
        pathname: name of the tdms file, which will be read
        keys: list of channel names for each column
        groupname: name of group, under which the channels are placed
    Returns:
        numpy array containing a column for each channel
    """
    original_file = TdmsFile(pathname)
    return [original_file.object(groupname, key).data for key in keys]

def write_tdms(data, keys, groupname, pathname):
    """
    Write data to TDMS file

    Args:
        data: numpy array of data, which will be saved as tdms file
        keys: list of channel names for each column; len(data) == len(keys) should be true
        groupname: all channels will be placed under this group within the tdms file structure
        pathname: path of the tdms file, which will be generated
    """
    with TdmsWriter(pathname) as tdms_writer:
        for idx, __ in enumerate(data):
            channel = ChannelObject(groupname, keys[idx], np.array(data[idx]))
            tdms_writer.write_segment([channel])

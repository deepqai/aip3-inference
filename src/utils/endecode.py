import base64
import zlib

import numpy as np


def encode(seg):  # eg: [[0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]]
    """
    This function is used to encode a 2D segmentation mask into a hash code. We use zlib to encode the map.
    :param  seg: the 2D segmentation mask.
    :returns the hash code in string format.
    """
    seg1D = np.reshape(seg, seg.size)  # eg: [0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1]
    bits = np.packbits(
        seg1D
    )  # Packs a binary array into bits in a uint8 array. # eg: [  1 255 128]
    byts = bytes(
        bits
    )  # numpy array to bytes                                    # eg: b'\x01\xff\x80'
    code1 = zlib.compress(
        byts, level=zlib.Z_BEST_COMPRESSION
    )  # eg: b'x\xdac\xfc\xdf\x00\x00\x02\x84\x01\x81'
    code2 = base64.b64encode(code1)  # eg: b'eNpj/N8AAAKEAYE='
    st = (
        code2.decode()
    )  # bytes to str                                         # eg: eNpj/N8AAAKEAYE=
    return st


def decode(st, width, height):
    """
    This function is used to decode a 2D segmentation mask from a hash code.
    :param  st:             the hash code in string format
    :param  width, height:  the shape of the mask
    :returns the 2D segmentation mask
    """
    code2 = (
        st.encode()
    )  # str to bytes                                            # eg: b'eNpj/N8AAAKEAYE='
    code1 = base64.b64decode(code2)  # eg: b'x\xdac\xfc\xdf\x00\x00\x02\x84\x01\x81'
    byts = zlib.decompress(code1)  # eg: b'\x01\xff\x80'
    bits = np.frombuffer(
        byts, dtype=np.uint8
    )  # bytes to numpy array            # eg: [  1 255 128]
    seg1D = np.unpackbits(bits)[
        : height * width
    ]  # eg: [0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1]
    seg = np.reshape(
        seg1D, (height, width)
    )  # eg: [[0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]]
    return seg


def mask_area_percentage(segmentation_mask, mask_width, mask_height):
    """
    computes the segmentation mask area percentage against the total image size
    """
    return decode(segmentation_mask, mask_width, mask_height).flatten().mean()

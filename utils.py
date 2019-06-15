import mmap


def load_big_file(f: str):
    """
    Workaround for loading a big pickle file. Files over 2GB cause pickle errors on certain Mac and Windows distributions.
    """
    with open(f, "r+b") as f_in:
        # mmap seems to be much more memory efficient
        bf = mmap.mmap(f_in.fileno(), 0)
        f_in.close()
    return bf

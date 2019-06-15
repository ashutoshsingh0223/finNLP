import io
import re
import string

pat = re.compile(r"([{}])".format(re.escape(string.punctuation)))
space_ = re.compile(r"\s+")


def put_space_around_punct(line: str) -> str:
    """
    BERT like preprocessing. Padding each special character with a space around it.
    Args:
        line: string to be preprocessed

    Returns:
        Preprocessed string
    """
    line = str(line).lower()
    line = line.replace("\n","")
    line = pat.sub(r" \1 ", line)
    line = space_.sub(" ", line)
    line = line.strip()
    return line

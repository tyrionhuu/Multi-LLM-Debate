import hashlib
def generate_hash(content: str) -> str:
    """Generate a unique hash using MD5.

    Args:
        content: String content to be hashed

    Returns:
        str: MD5 hash of the input content
    """
    return hashlib.md5(content.encode()).hexdigest()
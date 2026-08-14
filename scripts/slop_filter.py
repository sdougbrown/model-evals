import re


def is_code_line(s):
    """Return True if a line looks like code/command/config rather than prose."""
    if re.search(r"^(func|def|class|struct|interface|type|const|var|import|package|export|function|public|private|protected)\b", s):
        return True
    if re.search(r"^(pub|void|static|async|let|var)\b\s+[A-Za-z_<[]", s):
        return True
    if s.startswith("//"):
        return True
    if not s or set(s) <= set('{}[];,(): \"'):
        return True
    if re.search(r"interface [A-Z]|Record<|=>|;\s*$|\[\]struct\{", s):
        return True
    if re.search(r":[A-Za-z_<[]", s):
        return True
    if re.search(r"\$\(", s):
        return True
    if re.search(r"<<|;;|&&|\b(curl |wget |rm -|mkdir |cd /|cat <<|chmod |chown |git |sudo )", s):
        return True
    if re.search(r"^[A-Za-z_][A-Za-z0-9_]*\s*(:|=)\s", s):
        return True
    if re.search(r"--?[a-z][a-z0-9]*\s+|= ", s):
        return True
    return False


def is_prose_passage(words):
    """True if a joined passage is mostly natural prose."""
    if len(words) < 8:
        return False
    code_tokens = 0
    for w in words:
        w2 = w.strip("`()[]{}]['.,;:!?-<>")
        if re.search(r"[A-Za-z0-9_-]{12,}", w2):
            code_tokens += 1
        elif w2.isdigit():
            code_tokens += 1
        elif any(ch in w2 for ch in "_/{") and len(w2) >= 4:
            code_tokens += 1
    return code_tokens <= len(words) * 0.2

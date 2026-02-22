def find_code_and_delete_quotes(text: str) -> str | None:
    start = text.find('```')
    if start == -1:
        return None
    end = text.find('```', start + 3)
    if end == -1:
        return None

    code_start = text.find('\n', start + 3)    
    return text[code_start + 1:end]

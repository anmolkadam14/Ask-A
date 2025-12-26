MAX_HISTORY = 6
chat_history = []


def add_to_memory(role: str, text: str):
    chat_history.append(f"{role}: {text}")
    if len(chat_history) > MAX_HISTORY:
        chat_history.pop(0)


def get_memory_context() -> str:
    return "\n".join(chat_history)

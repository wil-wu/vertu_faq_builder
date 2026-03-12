from .config import qa_generation_service_settings
from .models import ChatSession


def build_contexts(chat_sessions: list[ChatSession]) -> list[str]:
    """从 chat_sessions 构建 context 列表"""
    contexts = []
    for chat_session in chat_sessions:
        messages = chat_session["messages"]
        context = "\n".join(
            [
                f"{idx + 1}. {message["role"]}: {message["content"]}"
                for idx, message in enumerate(messages)
            ]
        )
        if len(context) > qa_generation_service_settings.max_context_length:
            context = context[: qa_generation_service_settings.max_context_length]
        contexts.append(context)
    return contexts

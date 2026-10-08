"""Memory build phase prompt templates."""

from typing import Dict

MEMORIZE_TEMPLATES: Dict[str, str] = {

    # MedMemoryBench
    "medmemorybench_long_context_memorize": """以下是一段医疗对话记录，请仔细阅读并记忆其中的关键信息：

{context}""",

    "medmemorybench_rag_memorize": """以下是一段医疗对话记录，请仔细阅读并记忆其中的关键信息：

{context}""",

    "medmemorybench_agentic_memorize": """以下是一段医疗对话记录，请仔细阅读并记忆其中的关键信息：

{context}""",

    # LoCoMo
    "locomo_long_context_memorize": """Dialogue between User and Assistant {timestamp}
<User> The following context is the conversation record. Pay attention to specific DATES and TIMES mentioned - convert any relative time references (like "yesterday", "last week") to absolute dates based on the conversation timestamp.
{context}
<Assistant> I have memorized the dialogue including all dates and time references. I will provide concise, direct answers.""",

    "locomo_rag_memorize": """Dialogue between User and Assistant {timestamp}
<User> The following context is the conversation record. Pay attention to specific DATES and TIMES mentioned - convert any relative time references (like "yesterday", "last week") to absolute dates based on the conversation timestamp.
{context}
<Assistant> I have memorized the dialogue including all dates and time references. I will provide concise, direct answers.""",

    "locomo_agentic_memorize": """Dialogue between User and Assistant {timestamp}
<User> The following context is the conversation record. Pay attention to specific DATES and TIMES mentioned - convert any relative time references (like "yesterday", "last week") to absolute dates based on the conversation timestamp.
{context}
<Assistant> I have memorized the dialogue including all dates and time references. I will provide concise, direct answers.""",

    # MedMemoryBench - English
    "medmemorybench_en_long_context_memorize": """The following is a medical dialogue record. Please read it carefully and memorize the key information:

{context}""",

    "medmemorybench_en_rag_memorize": """The following is a medical dialogue record. Please read it carefully and memorize the key information:

{context}""",

    "medmemorybench_en_agentic_memorize": """The following is a medical dialogue record. Please read it carefully and memorize the key information:

{context}""",

    # LongMemEval
    "longmemeval_long_context_memorize": """The following are conversation sessions between a user and an assistant. Please read and memorize all details carefully, including dates, facts, preferences, and any updates to previously mentioned information.

{context}""",

    "longmemeval_rag_memorize": """The following are conversation sessions between a user and an assistant. Please read and memorize all details carefully, including dates, facts, preferences, and any updates to previously mentioned information.

{context}""",

    "longmemeval_agentic_memorize": """The following are conversation sessions between a user and an assistant. Please read and memorize all details carefully, including dates, facts, preferences, and any updates to previously mentioned information.

{context}""",

    # AMA-Bench
    "ama_bench_long_context_memorize": """The following is an agent's task execution trajectory. Please read it carefully and memorize all actions taken, observations received, state changes, and the overall progression of the task.

{context}""",

    "ama_bench_rag_memorize": """The following is an agent's task execution trajectory. Please read it carefully and memorize all actions taken, observations received, state changes, and the overall progression of the task.

{context}""",

    "ama_bench_agentic_memorize": """The following is an agent's task execution trajectory. Please read it carefully and memorize all actions taken, observations received, state changes, and the overall progression of the task.

{context}""",

    # LongMemEval-V2
    "longmemeval_v2_long_context_memorize": """The following are web-agent task trajectory records. Each trajectory contains a task goal, outcome, and a sequence of states with URLs, actions, thoughts, and page content observed during task execution. Please read and memorize all details carefully, including specific data values, product names, order numbers, URLs, user actions, and task outcomes.

{context}""",

    "longmemeval_v2_rag_memorize": """The following are web-agent task trajectory records. Each trajectory contains a task goal, outcome, and a sequence of states with URLs, actions, thoughts, and page content observed during task execution. Please read and memorize all details carefully, including specific data values, product names, order numbers, URLs, user actions, and task outcomes.

{context}""",

    "longmemeval_v2_agentic_memorize": """The following are web-agent task trajectory records. Each trajectory contains a task goal, outcome, and a sequence of states with URLs, actions, thoughts, and page content observed during task execution. Please read and memorize all details carefully, including specific data values, product names, order numbers, URLs, user actions, and task outcomes.

{context}""",

}

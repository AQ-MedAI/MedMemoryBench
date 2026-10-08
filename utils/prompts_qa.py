"""Query answer phase prompt templates."""

from typing import Dict

QA_TEMPLATES: Dict[str, str] = {

    # MedMemoryBench - Entity Exact Match
    "medmemorybench_entity_exact_match_qa": """请根据{memory_source}，准确回答以下问题。

问题：{question}

【回答要求】请直接给出实体名称，无需长篇解释，只需精简地回答关键实体词即可。

答案：""",

    # MedMemoryBench - Temporal Localization
    "medmemorybench_temporal_localization_qa": """请根据{memory_source}，准确回答以下问题。

问题：{question}

【回答要求】如果问题询问时间，请使用 YYYY-MM-DD 格式回答（如 2024-01-15）；如果问题询问某时间发生的事件，请清晰描述事件的具体内容和细节。

答案：""",

    # MedMemoryBench - State Update
    "medmemorybench_state_update_qa": """请根据{memory_source}，准确回答以下问题。

问题：{question}

【回答要求】
- 描述患者的最新状态，体现状态的前后变化
- 语气亲切专业，像患者的私人医疗助手
- 简洁直接，避免冗长解释

答案：""",

    # MedMemoryBench - Multiple Choice
    "medmemorybench_multiple_choice_qa": """请根据{memory_source}，结合患者过往的过敏史、疾病史、用药及个人偏好等信息，回答如下问题：

{question}

【回答要求】请选择所有正确的选项，直接给出选项字母（如 B 或 B,D），无需解释。

答案：""",

    # MedMemoryBench - Inference Generation
    "medmemorybench_inference_generation_qa": """请根据{memory_source}，结合患者过往的过敏史、疾病史、用药及个人偏好等信息，回答如下问题：

{question}

【回答要求】
- 必须基于记忆中该患者的具体信息进行推理，不要给出通用医学建议
- 语气亲切专业，像患者的私人医疗助手
- 简洁直接，回答到点子上，避免废话和套话
- 如有建议或不建议某事，需简要说明基于该患者情况的原因

答案：""",

    # MedMemoryBench - Multi-hop Clinical Deduction
    "medmemorybench_multi_hop_clinical_deduction_qa": """请根据{memory_source}，仔细回顾患者的完整病史记录，结合多次就诊的信息进行综合分析：

{question}

【回答要求】请深入检索之前的记忆内容，结合多个历史信息点进行推理。回答时需要：
1. 明确列出你所依据的记忆内容
2. 展示清晰的推理路径（从哪些信息推导出哪些结论）
3. 给出最终的综合判断

答案：""",

    # MedMemoryBench - Default fallback
    "medmemorybench_default_qa": """请根据{memory_source}，准确回答以下问题。

问题：{question}

答案：""",

    # MedMemoryBench - English: Entity Exact Match
    "medmemorybench_en_entity_exact_match_qa": """Based on {memory_source}, accurately answer the following question.

Question: {question}

[ANSWER REQUIREMENTS] Provide the entity name directly. No lengthy explanations needed — just give the key entity term(s) briefly.

Answer:""",

    # MedMemoryBench - English: Temporal Localization
    "medmemorybench_en_temporal_localization_qa": """Based on {memory_source}, accurately answer the following question.

Question: {question}

[ANSWER REQUIREMENTS] If the question asks about a time, answer in YYYY-MM-DD format (e.g., 2024-01-15). If the question asks about an event at a specific time, clearly describe the specific content and details of the event.

Answer:""",

    # MedMemoryBench - English: State Update
    "medmemorybench_en_state_update_qa": """Based on {memory_source}, accurately answer the following question.

Question: {question}

[ANSWER REQUIREMENTS]
- Describe the patient's most recent status, reflecting the changes over time
- Maintain a warm yet professional tone, like a personal medical assistant
- Be concise and direct, avoid lengthy explanations

Answer:""",

    # MedMemoryBench - English: Multiple Choice
    "medmemorybench_en_multiple_choice_qa": """Based on {memory_source}, considering the patient's allergy history, medical history, medications, and personal preferences, answer the following question:

{question}

[ANSWER REQUIREMENTS] Select all correct options and provide only the option letter(s) (e.g., B or B,D). No explanation needed.

Answer:""",

    # MedMemoryBench - English: Inference Generation
    "medmemorybench_en_inference_generation_qa": """Based on {memory_source}, considering the patient's allergy history, medical history, medications, and personal preferences, answer the following question:

{question}

[ANSWER REQUIREMENTS]
- You must reason based on the specific information of this patient from memory, do not give generic medical advice
- Maintain a warm yet professional tone, like a personal medical assistant
- Be concise and direct, get to the point, avoid filler and boilerplate
- If recommending or advising against something, briefly explain the reason based on this patient's specific situation

Answer:""",

    # MedMemoryBench - English: Multi-hop Clinical Deduction
    "medmemorybench_en_multi_hop_clinical_deduction_qa": """Based on {memory_source}, carefully review the patient's complete medical history and conduct a comprehensive analysis combining information from multiple visits:

{question}

[ANSWER REQUIREMENTS] Please thoroughly search through prior memory content and reason by combining multiple historical data points. Your answer should:
1. Clearly list the memory content you are drawing upon
2. Demonstrate a clear reasoning path (from which information to which conclusions)
3. Provide a final comprehensive judgment

Answer:""",

    # MedMemoryBench - English: Default fallback
    "medmemorybench_en_default_qa": """Based on {memory_source}, accurately answer the following question.

Question: {question}

Answer:""",

    # LoCoMo - Default
    "locomo_default_qa": """Based on {memory_source}, answer the question below.

Question: {question}

FORMAT REQUIREMENTS (CRITICAL - follow exactly):
- Give ONLY the direct answer, NO explanations or justifications
- Use the SHORTEST form that answers the question completely
- Examples of correct format:
  * "What hobby?" → "pottery" (NOT "pottery, which she finds relaxing")
  * "Who is X?" → "her sister" (NOT "her sister, they are very close")
  * "What did X do?" → "went to the beach" (NOT "she went to the beach because...")

Answer:""",

    # LoCoMo - Single-hop
    "locomo_single_hop_qa": """Based on {memory_source}, answer the following factual question.

Question: {question}

ANSWER FORMAT (CRITICAL):
1. Give ONLY the direct answer - NO explanations, NO context, NO "because..."
2. Use the SHORTEST complete answer:
   - "What book?" → "The Alchemist" (NOT "The Alchemist by Paulo Coelho")
   - "What activity?" → "dancing" (NOT "dancing, which they both enjoy")
   - "What did X get?" → "a trophy" (NOT "she received a trophy from...")
   - "Who?" → "Ed Sheeran" (NOT "Ed Sheeran's Perfect")
3. For Yes/No questions: Answer ONLY "Yes" or "No"
4. For lists: "item1, item2, item3" (NO "and", NO explanations)

Answer:""",

    # LoCoMo - Multi-hop
    "locomo_multi_hop_qa": """Based on {memory_source}, answer the following question that requires combining information from multiple conversations.

Question: {question}

ANSWER FORMAT (CRITICAL):
1. If asking "how many" → Give ONLY the number: "2", "3", "three"
2. If asking for a list → Give items separated by commas: "beach, park, museum"
3. If asking about a person's status/characteristic → Give the direct answer only
4. NO explanations, NO justifications, NO context
5. Keep answer as SHORT as possible while being complete

Answer:""",

    # LoCoMo - Temporal
    "locomo_temporal_qa": """Based on {memory_source}, answer the following time-related question.

Question: {question}

CRITICAL DATE FORMAT RULES:
1. Convert ALL relative times to ABSOLUTE dates:
   - "yesterday" before "6 July 2023" → "5 July 2023"
   - "last week" before "9 June 2023" → "The week before 9 June 2023"
   - "two days ago" before "12 July 2023" → "10 July 2023"
   - "last Friday" before "15 July 2023" → "The Friday before 15 July 2023"

2. Use these EXACT formats:
   - Specific dates: "7 May 2023", "10 July 2023"
   - Week references: "The week before 9 June 2023"
   - Day references: "The Friday before 15 July 2023"
   - Month only: "July 2023", "March 2023"
   - Year only: "2022", "2023"
   - Duration: "4 years", "two weeks", "10 years ago"

3. Give ONLY the date/time - NO explanations
   - Correct: "5 July 2023"
   - Wrong: "5 July 2023, when she went to the museum"

Answer:""",

    # LoCoMo - Open-domain
    "locomo_open_domain_qa": """Based on {memory_source}, answer the following inference question.

Question: {question}

ANSWER FORMAT (CRITICAL):
1. For Yes/No questions:
   - If clearly yes: "Yes"
   - If clearly no: "No"
   - If inference needed: "Likely yes" or "Likely no"
   - DO NOT add explanations after Yes/No

2. For preference/choice questions:
   - "National park" (NOT "National park; she likes outdoors...")
   - "Liberal" (NOT "Likely liberal or progressive, since...")

3. For "what would/could" questions:
   - Give the direct answer only: "beach", "California or Florida"

4. Keep answer under 10 words whenever possible
5. NO justifications, NO "since...", NO "because..."

Answer:""",

    # LoCoMo - Adversarial
    "locomo_adversarial_qa": """Based on {memory_source}, answer the following question.

Question: {question}

CRITICAL INSTRUCTIONS:
- ONLY answer if the information is EXPLICITLY stated in the memories
- If the specific information asked is NOT mentioned, answer exactly: "No information available"
- Do NOT guess, infer, or make assumptions
- Do NOT confuse similar but different information
- If you find relevant information, give the direct answer in the SHORTEST form possible

Answer:""",

    # LongMemEval - Default (used for all question types)
    "longmemeval_default_qa": """Based on {memory_source}, answer the following question.

Current date: {question_date}

Question: {question}

Answer concisely and directly.

Answer:""",

    # LongMemEval - single-session-user
    "longmemeval_single-session-user_qa": """Based on {memory_source}, answer the following question about information the user mentioned in a conversation.

Current date: {question_date}

Question: {question}

Provide a concise and direct answer.

Answer:""",

    # LongMemEval - single-session-assistant
    "longmemeval_single-session-assistant_qa": """Based on {memory_source}, answer the following question about information the assistant provided in a conversation.

Current date: {question_date}

Question: {question}

Provide a concise and direct answer.

Answer:""",

    # LongMemEval - single-session-preference
    "longmemeval_single-session-preference_qa": """Based on {memory_source}, answer the following question about the user's personal preferences or habits mentioned in a conversation.

Current date: {question_date}

Question: {question}

Provide a personalized answer based on what you know about the user.

Answer:""",

    # LongMemEval - multi-session
    "longmemeval_multi-session_qa": """Based on {memory_source}, answer the following question that may require combining information from multiple conversations.

Current date: {question_date}

Question: {question}

Provide a concise and direct answer.

Answer:""",

    # LongMemEval - temporal-reasoning
    "longmemeval_temporal-reasoning_qa": """Based on {memory_source}, answer the following question that involves temporal reasoning about events across conversations.

Current date: {question_date}

Question: {question}

Provide a concise and direct answer.

Answer:""",

    # LongMemEval - knowledge-update
    "longmemeval_knowledge-update_qa": """Based on {memory_source}, answer the following question. Note that information may have been updated over time - provide the most recent/updated answer.

Current date: {question_date}

Question: {question}

Provide the most up-to-date answer.

Answer:""",

    # =========================================================================
    # AMA-Bench - Agent trajectory memory QA templates
    # =========================================================================

    "ama_bench_default_qa": """Based on {memory_source} of the agent's trajectory, answer the following question.

Question: {question}

Provide a concise and direct answer.

Answer:""",

    "ama_bench_recall_qa": """Based on {memory_source} of the agent's trajectory, answer the following question about specific actions or observations from the trajectory.

Question: {question}

Provide a concise and direct answer based on what you recall from the trajectory.

Answer:""",

    "ama_bench_causal_inference_qa": """Based on {memory_source} of the agent's trajectory, answer the following question about cause-effect relationships between actions and outcomes.

Question: {question}

Provide a clear answer explaining the causal relationship based on the trajectory.

Answer:""",

    "ama_bench_state_updating_qa": """Based on {memory_source} of the agent's trajectory, answer the following question about state changes that occurred during task execution.

Question: {question}

Provide a concise answer reflecting the relevant state transitions from the trajectory.

Answer:""",

    "ama_bench_state_abstraction_qa": """Based on {memory_source} of the agent's trajectory, answer the following question requiring high-level understanding of the agent's state and strategy.

Question: {question}

Provide a clear and comprehensive answer based on your understanding of the overall trajectory.

Answer:""",

    # =========================================================================
    # LongMemEval-V2 - Web agent trajectory memory QA templates
    # =========================================================================

    "longmemeval_v2_default_qa": """Based on {memory_source}, answer the following question about web-agent task trajectories.

Question: {question}

Important: Enclose your final answer in \\boxed{{}}.
If the question cannot be answered based on the available information, respond with \\boxed{{UNKNOWN}}.

Answer:""",

    "longmemeval_v2_static-environment_qa": """Based on {memory_source}, answer the following question about the state of a web environment observed during agent task execution.

Question: {question}

Important: Enclose your final answer in \\boxed{{}}.
If you do not know the answer, output exactly \\boxed{{UNKNOWN}}.

Answer:""",

    "longmemeval_v2_static-environment-abs_qa": """Based on {memory_source}, answer the following question about the state of a web environment observed during agent task execution.

Question: {question}

Important: Enclose your final answer in \\boxed{{}}.
If the question cannot be answered based on the available information or the premise of the question is incorrect, output exactly \\boxed{{UNKNOWN}} and briefly explain why.

Answer:""",

    "longmemeval_v2_dynamic-environment_qa": """Based on {memory_source}, answer the following question about changes that occurred in the web environment during agent task execution.

Question: {question}

Important: Enclose your final answer in \\boxed{{}}.
If you do not know the answer, output exactly \\boxed{{UNKNOWN}}.

Answer:""",

    "longmemeval_v2_dynamic-environment-abs_qa": """Based on {memory_source}, answer the following question about changes that occurred in the web environment during agent task execution.

Question: {question}

Important: Enclose your final answer in \\boxed{{}}.
If the question cannot be answered based on the available information or the premise of the question is incorrect, output exactly \\boxed{{UNKNOWN}} and briefly explain why.

Answer:""",

    "longmemeval_v2_procedure_qa": """Based on {memory_source}, answer the following question about the workflow or procedure followed during agent task execution.

Question: {question}

Important: Enclose your final answer in \\boxed{{}}.
If you do not know the answer, output exactly \\boxed{{UNKNOWN}}.

Answer:""",

    "longmemeval_v2_procedure-abs_qa": """Based on {memory_source}, answer the following question about the workflow or procedure followed during agent task execution.

Question: {question}

Important: Enclose your final answer in \\boxed{{}}.
If the question cannot be answered based on the available information or the premise of the question is incorrect, output exactly \\boxed{{UNKNOWN}} and briefly explain why.

Answer:""",

    "longmemeval_v2_errors-gotchas_qa": """Based on {memory_source}, answer the following question about errors, gotchas, or unexpected issues encountered during agent task execution in the web environment.

Question: {question}

Important: Enclose your final answer in \\boxed{{}}.
If you believe the question's premise is wrong, provide an explanation in \\boxed{{}} explaining why the question is flawed.

Answer:""",

}

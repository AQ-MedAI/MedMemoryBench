"""LLM-as-Judge evaluation phase prompt templates."""

from typing import Dict

JUDGE_TEMPLATES: Dict[str, str] = {

    # MedMemoryBench - Temporal Localization
    "medmemorybench_temporal_localization_judge": """你是一个严格的医疗对话评测裁判。请判断模型的回答是否正确回答了时间相关的问题。

【问题】
{question}

【标准答案】
{expected_answer}

【答案说明】
{explanation}

【模型回答】
{model_output}

【评判标准】
这是一个时间定位类问题，可能是以下两种形式之一：
1. 询问某事件发生的时间 → 模型需要正确回答时间点
2. 询问某时间发生了什么事件 → 模型需要正确回答事件内容

请严格判断：
- 如果模型回答包含了正确的时间点或正确的事件内容，判定为【正确】
- 如果模型回答的时间/事件与标准答案不符或未能回答，判定为【错误】
- 时间格式可以不完全一致，但必须指向同一时间点（如"2024年1月1日"和"2024-01-01"视为相同）

请按以下 JSON 格式输出：
{{"is_correct": true/false, "reason": "简要判断理由"}}

只输出 JSON，不要有其他内容。""",

    # MedMemoryBench - State Update
    "medmemorybench_state_update_judge": """你是一个非常严格的医疗对话评测裁判。请判断模型的回答是否正确反映了患者的最新状态。

【问题】
{question}

【标准答案】
{expected_answer}

【答案说明】
{explanation}

【模型回答】
{model_output}

【评判标准】
这是一个状态更新类问题，考察模型是否基于记忆中的患者历史信息正确回答最新状态。

⚠️ 核心评判原则（极其重要）：
1. **必须基于记忆回答**：模型的回答必须体现出对患者过往记忆信息的使用，而不是凭猜测或通用医学知识回答。
2. **禁止猜测回答**：如果模型没有检索到相关记忆信息，却给出了"碰巧正确"的答案，应判定为【错误】。
3. **信息来源要求**：正确的回答应该能让人感受到模型是"记得"这个患者的具体情况，而不是在猜测。

请严格判断：
- 如果模型回答体现了对患者历史记忆的使用，且核心内容与标准答案一致，判定为【正确】
- 如果模型回答包含了标准答案的关键信息点，且这些信息明显来自对患者记忆的检索，判定为【正确】
- 如果模型回答与标准答案有明显矛盾、遗漏关键信息、或给出了过时的状态，判定为【错误】
- 如果模型表示不知道或无法回答，判定为【错误】
- ⚠️ 如果模型的回答看起来过于泛泛、缺乏具体患者信息的支撑、像是凭猜测给出的答案，即使内容碰巧与标准答案相近，也应判定为【错误】

请按以下 JSON 格式输出：
{{"is_correct": true/false, "reason": "简要判断理由，需说明模型是否体现了对患者记忆的使用"}}

只输出 JSON，不要有其他内容。""",

    # MedMemoryBench - Inference Generation
    "medmemorybench_inference_generation_judge": """你是一个非常严格的医疗对话评测裁判。请判断模型的推理回答是否正确。

【问题】
{question}

【标准答案】
{expected_answer}

【答案说明】
{explanation}
{metadata_info}

【模型回答】
{model_output}

【评判标准】
这是一个推理生成类问题，考察模型是否能基于患者个人信息进行正确的医学推理。

核心评判要点：

1. **患者信息利用（关键）**
   - 模型必须体现对记忆中患者特定信息的使用
   - 如果metadata中提供了required_patient_info，模型回答必须反映出对这些关键信息的理解（重要）
   - 患者的具体情况和过往记忆如有忽略和缺漏，则判定为【错误】

2. **推理质量**
   - 模型必须基于检索到的患者历史信息展开推理，而不是单纯根据自己的医学常识见解
   - 仅给出结论，而对患者信息和记忆的引用不够充分，判定为【错误】
   - 如果模型给出了"常见错误答案"类型的回答（通用建议），判定为【错误】

3. **结论正确性**
   - 最终建议/结论应与标准答案方向完全一致
   - 即使结论正确，但如果缺乏基于患者信息的推理，仍判定为【错误】

判定规则：
- 【正确】：回答使用了患者的特定信息，包含了所需的患者信息要点，且得出了准确无误的结论
- 【错误】：回答未考虑充分患者的具体情况
- 【错误】：回答忽略了required_patient_info中的某些关键信息
- 【错误】：回答符合common_wrong_answer的模式
- 【错误】：模型拒绝回答或声称没有信息

请按以下 JSON 格式输出：
{{"is_correct": true/false, "reason": "简要判断理由"}}

只输出 JSON，不要有其他内容。""",

    # MedMemoryBench - Multi-hop Clinical Deduction
    "medmemorybench_multi_hop_clinical_deduction_judge": """你是一个**极其严格**的医疗多跳推理评测裁判。你的任务是严格验证模型是否真正从患者历史记忆中检索并使用了具体信息来进行多跳临床推理。

【问题】
{question}

【标准答案】
{expected_answer}

【答案说明】
{explanation}
{nodes_for_validation}
{required_nodes_str}
【推理跳数】: {hop_count}
【推理模式】: {reasoning_pattern}

【模型回答】
{model_output}

---

## 评测任务：严格逐节点验证推理链

这是一个多跳临床推理问题，**核心考察点是模型是否能从患者历史记忆中准确检索并使用具体的个人化医疗信息**。

### ⚠️ 关键评判原则（必须严格遵守）

1. **患者特定信息原则**：模型必须明确引用患者的**具体数据**（如具体的检查数值、用药剂量、症状出现的具体时间、特定的诊断结果等），而不是给出泛泛的医学常识。
   - ❌ "血糖控制不好可能导致..." → 这是通用医学知识，不是患者特定信息
   - ✅ "您的空腹血糖从6.8升高到8.2..." → 这是患者特定信息

2. **记忆检索证据原则**：如果模型未能体现出对患者历史记录的具体引用，即使推理方向正确，也应判定为**不合格**。模型必须展示它"记得"患者的具体情况。

3. **因果链严格对应原则**：模型建立的因果关系必须与【推理链节点】中描述的因果机制**精确对应**，不能用相似但不同的机制替代。

4. **节点内容精确匹配原则**：节点验证时，不能仅因为模型提到了相关概念就判定为"covered"，必须验证模型是否提及了节点中的**核心具体内容**。

---

## 评判步骤

### 步骤1：严格逐节点检查
对【推理链节点】中的每个节点，必须验证以下所有条件：

**条件A - 具体信息匹配**：
- 模型是否提及了该节点中的**具体数据/时间/事件**？
- 如果节点包含具体数值（如"TSH 0.02"），模型必须提及相同或等价的数值
- 如果节点包含具体时间（如"2024年10月"），模型必须体现对该时间点的认知
- 仅提及相关概念（如"甲状腺功能"）而无具体数据，**不算覆盖**

**条件B - 因果机制正确**：
- 模型描述的因果机制是否与标准推理链**完全一致**？
- 使用了不同的病理生理解释（即使听起来合理）**不算正确**
- 跳过中间环节直接得出结论**不算正确**

**条件C - 信息来源明确**：
- 模型的回答是否明确体现了这是来自患者历史记忆的信息？
- 纯粹基于医学常识的推断**不能得分**

### 步骤2：计算三个评分维度（严格标准）

**NCR (节点覆盖率)** = 完全满足条件A的节点数 / 总节点数
- 仅提及概念但无具体数据 → 该节点不计入覆盖
- 数据有误或时间对不上 → 该节点不计入覆盖

**CRC (因果关系正确性)** = 正确建立的因果链接数 / 应有的因果链接数
- 必须是标准答案中描述的因果机制，不接受"等效替代"
- 跳过中间节点的因果链接 → 不计分

**CC (推理链完整性)**
- 1.0 = 完整覆盖所有节点且因果关系正确
- 0.7 = 覆盖80%以上节点，核心因果关系正确
- 0.5 = 覆盖60%以上节点，主要因果关系正确
- 0.3 = 部分节点覆盖，因果关系有缺失
- 0.0 = 无有效推理链或方向完全错误

### 步骤3：综合判定（高标准）

**【正确】条件（必须同时满足）**：
- NCR >= 0.75（至少覆盖四分之三的节点的具体内容）
- CRC >= 0.75（因果关系基本完整且正确）
- CC >= 0.7（推理链基本完整）
- 最终结论与标准答案一致

**【部分正确】条件**：
- NCR >= 0.5 且 CRC >= 0.5 且 CC >= 0.5
- 主要推理方向正确，但有明显缺失

**【错误】条件（满足任一即判错）**：
- 模型未能从记忆中检索出患者的具体信息
- 推理基于通用医学知识而非患者特定情况
- 因果机制与标准答案不符
- 结论方向错误
- NCR < 0.5 或 CRC < 0.5 或 CC < 0.5

---

## 输出格式

请按以下 JSON 格式输出你的严格评判结果：
{{
    "node_validations": [
        {{
            "node_id": 1,
            "mentioned": true/false,
            "specific_data_matched": true/false,
            "causal_link_correct": true/false,
            "note": "必须说明：1)模型提及了哪些具体数据 2)是否与节点内容精确匹配 3)因果关系是否正确"
        }}
    ],
    "ncr_score": 0.0-1.0,
    "crc_score": 0.0-1.0,
    "cc_score": 0.0-1.0,
    "memory_retrieval_quality": "excellent/good/partial/poor/none",
    "uses_patient_specific_info": true/false,
    "is_correct": true/false,
    "reason": "综合评判理由，必须说明：1)模型是否使用了患者特定信息 2)哪些节点未被覆盖 3)因果链是否完整"
}}

只输出 JSON，不要有其他内容。""",

    # LoCoMo - Open-domain
    "locomo_open_domain_judge": """You are a lenient judge evaluating conversational memory and reasoning.

**Question:**
{question}

**Expected Answer:**
{expected_answer}

**Model's Answer:**
{model_output}

**Evaluation Criteria:**
This is an open-domain inference question. Be LENIENT in judging:

1. For Yes/No questions:
   - Expected "Yes" → Accept: "Yes", "Likely yes", "yes" with any explanation
   - Expected "No" → Accept: "No", "Likely no", "no" with any explanation
   - Expected "Likely yes" → Accept: "Yes", "Likely yes"
   - Expected "Likely no" → Accept: "No", "Likely no"

2. For choice/preference questions (e.g., "beach or mountains?"):
   - If expected answer is contained in model output, mark CORRECT
   - "beach" in "Likely yes, close to the beach" → CORRECT

3. For inference questions:
   - If core conclusion matches expected answer, mark CORRECT
   - Extra explanation does NOT make it wrong

Output JSON only: {{"is_correct": true/false, "reason": "brief explanation"}}""",

    # LoCoMo - Multi-hop
    "locomo_multi_hop_judge": """You are a lenient judge evaluating multi-hop conversational memory.

**Question:**
{question}

**Expected Answer (may contain multiple sub-answers):**
{expected_answer}

**Model's Answer:**
{model_output}

**Evaluation Criteria:**
Be LENIENT - focus on whether the core answer is present:

1. For "how many" questions:
   - Expected "2" → Accept: "2", "two", "twice", or answer containing "2"
   - Expected "three" → Accept: "3", "three", "Three"

2. For list questions:
   - If model's answer contains the expected items (even with extras), mark CORRECT
   - Order doesn't matter

3. For status questions:
   - Expected "Single" → Accept: "single", "Single", "not married", etc.

Output JSON only: {{"is_correct": true/false, "score": 0.0-1.0, "reason": "brief explanation"}}""",

    # LoCoMo - Temporal
    "locomo_temporal_judge": """You are a lenient judge evaluating temporal questions.

**Question:**
{question}

**Expected Answer:**
{expected_answer}

**Model's Answer:**
{model_output}

**Evaluation Criteria:**
Be LENIENT with date/time matching:

1. Equivalent date formats are ALL correct:
   - "7 May 2023" = "May 7, 2023" = "May 7th, 2023"
   - "July 2023" = "in July 2023" = "July, 2023"

2. Relative-to-absolute conversions:
   - Expected "10 July 2023" = "two days before 12 July 2023" (same date)
   - Expected "5 July 2023" = "yesterday" (if context date is 6 July)
   - Expected "The week before 9 June 2023" = "last week" (before 9 June context)

3. Approximate matches:
   - "The Friday before 15 July 2023" ≈ "Last Friday" (before 15 July)
   - "August 2023" = "in August 2023" = "around August 2023"

4. Duration formats:
   - "4 years" = "four years" = "about 4 years"
   - "10 years ago" = "ten years ago"

If the dates refer to the SAME point in time, mark CORRECT.

Output JSON only: {{"is_correct": true/false, "reason": "brief explanation"}}""",

    # LoCoMo - Single-hop
    "locomo_single_hop_judge": """You are a lenient judge evaluating single-hop factual questions.

**Question:**
{question}

**Expected Answer:**
{expected_answer}

**Model's Answer:**
{model_output}

**Evaluation Criteria:**
Be LENIENT - the core answer matters, not the format:

1. If expected answer is CONTAINED in model output → CORRECT
   - Expected: "The Alchemist" → "The Alchemist by Paulo Coelho" is CORRECT
   - Expected: "dancing" → "by dancing" is CORRECT
   - Expected: "Ed Sheeran" → "Ed Sheeran's Perfect" is CORRECT

2. For Yes/No questions:
   - Expected "Yes" → Accept any answer starting with "Yes" or "Likely yes"
   - Expected "No" → Accept any answer starting with "No" or "Likely no"

3. For lists:
   - If all expected items are present (even with extras), mark CORRECT

4. Minor variations are acceptable:
   - "a trophy" = "the trophy" = "trophy"
   - "by biking" = "biking" = "bike"

Output JSON only: {{"is_correct": true/false, "reason": "brief explanation"}}""",

    # MedMemoryBench - English: Temporal Localization
    "medmemorybench_en_temporal_localization_judge": """You are a strict medical dialogue evaluation judge. Determine whether the model's answer correctly addresses the time-related question.

**Question:**
{question}

**Reference Answer:**
{expected_answer}

**Answer Explanation:**
{explanation}

**Model's Answer:**
{model_output}

**Evaluation Criteria:**
This is a temporal localization question, which may take one of the following two forms:
1. Asking when a certain event occurred → The model must correctly provide the time point
2. Asking what happened at a certain time → The model must correctly describe the event content

Judge strictly:
- If the model's answer contains the correct time point or the correct event content, judge as [CORRECT]
- If the model's answer about the time/event does not match the reference answer or fails to answer, judge as [INCORRECT]
- Date formats do not need to be identical, but must refer to the same time point (e.g., "January 1, 2024" and "2024-01-01" are considered equivalent)

Output in the following JSON format:
{{"is_correct": true/false, "reason": "brief justification"}}

Output JSON only, no other content.""",

    # MedMemoryBench - English: State Update
    "medmemorybench_en_state_update_judge": """You are a very strict medical dialogue evaluation judge. Determine whether the model's answer correctly reflects the patient's most recent status.

**Question:**
{question}

**Reference Answer:**
{expected_answer}

**Answer Explanation:**
{explanation}

**Model's Answer:**
{model_output}

**Evaluation Criteria:**
This is a state update question, testing whether the model correctly answers the latest status based on the patient's historical information in memory.

⚠️ Core Evaluation Principles (critically important):
1. **Must be based on memory**: The model's answer must demonstrate the use of the patient's past memory information, not guessing or generic medical knowledge.
2. **No guessing allowed**: If the model has not retrieved relevant memory information but gives a "coincidentally correct" answer, it should be judged as [INCORRECT].
3. **Information source requirement**: A correct answer should convey that the model "remembers" this patient's specific situation, rather than guessing.

Judge strictly:
- If the model's answer demonstrates the use of patient historical memory and the core content is consistent with the reference answer, judge as [CORRECT]
- If the model's answer contains key information points from the reference answer, and these clearly originate from patient memory retrieval, judge as [CORRECT]
- If the model's answer clearly contradicts the reference answer, omits key information, or provides outdated status, judge as [INCORRECT]
- If the model states it does not know or cannot answer, judge as [INCORRECT]
- ⚠️ If the model's answer appears too generic, lacks specific patient information support, or seems like a guess, even if the content happens to be close to the reference answer, judge as [INCORRECT]

Output in the following JSON format:
{{"is_correct": true/false, "reason": "brief justification, must indicate whether the model demonstrated use of patient memory"}}

Output JSON only, no other content.""",

    # MedMemoryBench - English: Inference Generation
    "medmemorybench_en_inference_generation_judge": """You are a very strict medical dialogue evaluation judge. Determine whether the model's reasoning answer is correct.

**Question:**
{question}

**Reference Answer:**
{expected_answer}

**Answer Explanation:**
{explanation}
{metadata_info}

**Model's Answer:**
{model_output}

**Evaluation Criteria:**
This is an inference generation question, testing whether the model can perform correct medical reasoning based on patient-specific information.

Core evaluation points:

1. **Patient Information Utilization (Key)**
   - The model must demonstrate the use of patient-specific information from memory
   - If required_patient_info is provided in metadata, the model's answer must reflect understanding of these key pieces of information (important)
   - If the patient's specific circumstances and past memories are ignored or missing, judge as [INCORRECT]

2. **Reasoning Quality**
   - The model must reason based on retrieved patient historical information, not purely from its own medical common sense
   - If only a conclusion is given without sufficient reference to patient information and memory, judge as [INCORRECT]
   - If the model gives a "common wrong answer" type of response (generic advice), judge as [INCORRECT]

3. **Conclusion Correctness**
   - The final recommendation/conclusion should be fully consistent with the reference answer in direction
   - Even if the conclusion is correct, if it lacks reasoning based on patient information, still judge as [INCORRECT]

Judgment rules:
- [CORRECT]: Answer uses patient-specific information, contains required patient information points, and reaches an accurate conclusion
- [INCORRECT]: Answer does not adequately consider the patient's specific circumstances
- [INCORRECT]: Answer ignores certain key information in required_patient_info
- [INCORRECT]: Answer matches the common_wrong_answer pattern
- [INCORRECT]: Model refuses to answer or claims no information

Output in the following JSON format:
{{"is_correct": true/false, "reason": "brief justification"}}

Output JSON only, no other content.""",

    # LongMemEval - single-session-user / single-session-assistant / multi-session
    # NOTE: These prompts exactly follow the original LongMemEval paper (evaluate_qa.py)
    # to ensure comparable results. Format: "Answer yes or no only."
    "longmemeval_single-session-user_judge": """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {question}\n\nCorrect Answer: {expected_answer}\n\nModel Response: {model_output}\n\nIs the model response correct? Answer yes or no only.""",

    "longmemeval_single-session-assistant_judge": """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {question}\n\nCorrect Answer: {expected_answer}\n\nModel Response: {model_output}\n\nIs the model response correct? Answer yes or no only.""",

    "longmemeval_multi-session_judge": """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {question}\n\nCorrect Answer: {expected_answer}\n\nModel Response: {model_output}\n\nIs the model response correct? Answer yes or no only.""",

    # LongMemEval - temporal-reasoning
    "longmemeval_temporal-reasoning_judge": """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {question}\n\nCorrect Answer: {expected_answer}\n\nModel Response: {model_output}\n\nIs the model response correct? Answer yes or no only.""",

    # LongMemEval - knowledge-update
    "longmemeval_knowledge-update_judge": """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {question}\n\nCorrect Answer: {expected_answer}\n\nModel Response: {model_output}\n\nIs the model response correct? Answer yes or no only.""",

    # LongMemEval - single-session-preference
    "longmemeval_single-session-preference_judge": """I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {question}\n\nRubric: {expected_answer}\n\nModel Response: {model_output}\n\nIs the model response correct? Answer yes or no only.""",

    # LongMemEval - abstention
    "longmemeval_abstention_judge": """I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {question}\n\nExplanation: {expected_answer}\n\nModel Response: {model_output}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only.""",

    # MedMemoryBench - English: Multi-hop Clinical Deduction
    "medmemorybench_en_multi_hop_clinical_deduction_judge": """You are an **extremely strict** medical multi-hop reasoning evaluation judge. Your task is to rigorously verify whether the model truly retrieved and used specific information from the patient's historical memory to perform multi-hop clinical reasoning.

**Question:**
{question}

**Reference Answer:**
{expected_answer}

**Answer Explanation:**
{explanation}
{nodes_for_validation}
{required_nodes_str}
**Reasoning Hops**: {hop_count}
**Reasoning Pattern**: {reasoning_pattern}

**Model's Answer:**
{model_output}

---

## Evaluation Task: Strict Node-by-Node Reasoning Chain Verification

This is a multi-hop clinical reasoning question. **The core assessment is whether the model can accurately retrieve and use specific personalized medical information from the patient's historical memory**.

### ⚠️ Key Evaluation Principles (must be strictly followed)

1. **Patient-Specific Information Principle**: The model must explicitly reference the patient's **specific data** (such as specific test values, medication dosages, specific timing of symptom onset, particular diagnostic results), rather than giving generic medical common sense.
   - ❌ "Poor blood sugar control may lead to..." → This is generic medical knowledge, not patient-specific information
   - ✅ "Your fasting blood glucose rose from 6.8 to 8.2..." → This is patient-specific information

2. **Memory Retrieval Evidence Principle**: If the model fails to demonstrate specific references to the patient's historical records, even if the reasoning direction is correct, it should be judged as **inadequate**. The model must show it "remembers" the patient's specific situation.

3. **Strict Causal Chain Correspondence Principle**: The causal relationships established by the model must **precisely correspond** to the causal mechanisms described in the reasoning chain nodes. Similar but different mechanisms cannot substitute.

4. **Node Content Precise Matching Principle**: During node verification, it is not sufficient to judge as "covered" merely because the model mentioned a related concept. You must verify whether the model referenced the **core specific content** within the node.

---

## Evaluation Steps

### Step 1: Strict Node-by-Node Check
For each node in the reasoning chain, all of the following conditions must be verified:

**Condition A - Specific Information Match**:
- Did the model mention the **specific data/time/event** in this node?
- If the node contains specific values (e.g., "TSH 0.02"), the model must mention the same or equivalent value
- If the node contains a specific time (e.g., "October 2024"), the model must demonstrate awareness of that time point
- Merely mentioning the related concept (e.g., "thyroid function") without specific data **does not count as coverage**

**Condition B - Correct Causal Mechanism**:
- Does the causal mechanism described by the model **exactly match** the reference reasoning chain?
- Using a different pathophysiological explanation (even if it sounds reasonable) **does not count as correct**
- Skipping intermediate steps to reach a conclusion directly **does not count as correct**

**Condition C - Clear Information Source**:
- Does the model's answer clearly demonstrate that this information comes from the patient's historical memory?
- Inferences based purely on medical common sense **cannot receive credit**

### Step 2: Calculate Three Scoring Dimensions (Strict Standards)

**NCR (Node Coverage Rate)** = Number of nodes fully satisfying Condition A / Total number of nodes
- Mentioning concept only without specific data → Node not counted as covered
- Incorrect data or mismatched timeline → Node not counted as covered

**CRC (Causal Relation Correctness)** = Number of correctly established causal links / Number of expected causal links
- Must be the causal mechanism described in the reference answer, "equivalent substitutions" not accepted
- Causal links skipping intermediate nodes → No credit

**CC (Chain Completeness)**
- 1.0 = Complete coverage of all nodes with correct causal relations
- 0.7 = Coverage of 80%+ nodes, core causal relations correct
- 0.5 = Coverage of 60%+ nodes, main causal relations correct
- 0.3 = Partial node coverage, causal relations have gaps
- 0.0 = No valid reasoning chain or completely wrong direction

### Step 3: Comprehensive Judgment (High Standards)

**[CORRECT] Conditions (all must be satisfied simultaneously)**:
- NCR >= 0.75 (at least three-quarters of nodes' specific content covered)
- CRC >= 0.75 (causal relations basically complete and correct)
- CC >= 0.7 (reasoning chain basically complete)
- Final conclusion consistent with reference answer

**[PARTIALLY CORRECT] Conditions**:
- NCR >= 0.5 and CRC >= 0.5 and CC >= 0.5
- Main reasoning direction correct, but with notable gaps

**[INCORRECT] Conditions (any one triggers incorrect judgment)**:
- Model failed to retrieve patient-specific information from memory
- Reasoning based on generic medical knowledge rather than patient-specific situation
- Causal mechanism inconsistent with reference answer
- Conclusion direction incorrect
- NCR < 0.5 or CRC < 0.5 or CC < 0.5

---

## Output Format

Output your strict evaluation result in the following JSON format:
{{
    "node_validations": [
        {{
            "node_id": 1,
            "mentioned": true/false,
            "specific_data_matched": true/false,
            "causal_link_correct": true/false,
            "note": "Must state: 1) What specific data the model mentioned 2) Whether it precisely matches node content 3) Whether the causal relation is correct"
        }}
    ],
    "ncr_score": 0.0-1.0,
    "crc_score": 0.0-1.0,
    "cc_score": 0.0-1.0,
    "memory_retrieval_quality": "excellent/good/partial/poor/none",
    "uses_patient_specific_info": true/false,
    "is_correct": true/false,
    "reason": "Comprehensive justification, must state: 1) Whether the model used patient-specific information 2) Which nodes were not covered 3) Whether the causal chain is complete"
}}

Output JSON only, no other content.""",

    # =========================================================================
    # AMA-Bench - Binary yes/no judge matching official implementation
    # All QA types use the same prompt structure from compute_llm_as_judge()
    # =========================================================================

    "ama_bench_recall_judge": """You are an expert evaluator. You will be given a question, a reference answer, and a predicted answer.
Your task is to determine if the predicted answer is correct based on:
1. Factual correctness compared to the reference
2. Completeness of the answer
3. Relevance to the question

Question: {question}

Reference Answer: {expected_answer}

Predicted Answer: {model_output}

Is the predicted answer correct? Respond with ONLY "yes" or "no". Do not include any thinking process, explanation, or additional text.

Answer:""",

    "ama_bench_causal_inference_judge": """You are an expert evaluator. You will be given a question, a reference answer, and a predicted answer.
Your task is to determine if the predicted answer is correct based on:
1. Factual correctness compared to the reference
2. Completeness of the answer
3. Relevance to the question

Question: {question}

Reference Answer: {expected_answer}

Predicted Answer: {model_output}

Is the predicted answer correct? Respond with ONLY "yes" or "no". Do not include any thinking process, explanation, or additional text.

Answer:""",

    "ama_bench_state_updating_judge": """You are an expert evaluator. You will be given a question, a reference answer, and a predicted answer.
Your task is to determine if the predicted answer is correct based on:
1. Factual correctness compared to the reference
2. Completeness of the answer
3. Relevance to the question

Question: {question}

Reference Answer: {expected_answer}

Predicted Answer: {model_output}

Is the predicted answer correct? Respond with ONLY "yes" or "no". Do not include any thinking process, explanation, or additional text.

Answer:""",

    "ama_bench_state_abstraction_judge": """You are an expert evaluator. You will be given a question, a reference answer, and a predicted answer.
Your task is to determine if the predicted answer is correct based on:
1. Factual correctness compared to the reference
2. Completeness of the answer
3. Relevance to the question

Question: {question}

Reference Answer: {expected_answer}

Predicted Answer: {model_output}

Is the predicted answer correct? Respond with ONLY "yes" or "no". Do not include any thinking process, explanation, or additional text.

Answer:""",

}

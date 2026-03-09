# Tuple
TUPLE_EXTRACTION_SYSTEM_PROMPT = """
[Task]
Given an input prompt, extract tuples under the schema below.
Only extract facts that are explicitly stated and visually checkable. Do not infer, embellish, or add typical details.

[What these tuples are used for]
These tuples will be converted into VQA questions and checked against an image to find misaligned facts.
Misaligned tuples will be converted into targeted edit instructions, so each tuple must be atomic and stably verifiable.

[Input]
A single text prompt describing an image.

[Output Format]
Output must be plain text lines.
Each line must be exactly one tuple in one of the allowed forms below.
Do not add labels, prefixes, numbering, JSON, markdown, or explanations.

[General Rules]
- Explicit-only: Create a tuple only if the fact is directly stated in the input text.
- No inference: Do not derive unstated properties, counts, relations, intentions, or typical parts.
- No duplication: Do not output paraphrase duplicates of the same fact.
- Naming: Use the most direct noun phrase for each entity and keep it consistent across tuples.
- Objective-checkable gate (required for attribute and global tuples):
  Create an attribute/global tuple only if the claim can be judged from pixels with a clear visual criterion and a small closed answer set,
  without relying on aesthetics, quality, mood, realism level, camera language, lighting, or subjective descriptors.

[Allowed Tuple Schemas]
entity - whole (X)
entity - part (OWNER PART)
relation - spatial (A, B, rel_token)
action - (A, action_token, B)
attribute - state (S, V)
attribute - type (S, V)
attribute - material (S, V)
attribute - texture (S, V)
attribute - shape (S, V)
attribute - size (S, V)
attribute - color (S, V)
other - text (S, "TEXT")
other - count (S, ==N)
global - style (STYLE)

[Type-Specific Rules]

entity - whole
- Create when a concrete, depictable entity X is explicitly mentioned as a noun/noun phrase.
- Include X only if it is central to the scene OR participates in any other tuple.
- Omit abstract concepts, implied entities, and non-depictable ideas.

entity - part
- Create only when an explicit part-of relation is stated (possessive phrasing or “PART of OWNER”).
- Do not add typical parts.

relation - spatial
- Create only when an explicit physical placement relation is stated between A and B.
- rel_token must match exactly one of the allowed rel_token strings provided by the user.
- Direction: (A, B, rel_token) means A is rel_token relative to B.
- Exclude non-spatial relations (possession, identity, association, feature/function).

action
- Create only when the text explicitly states an action/verb meaning that links A to B.
- action_token must reflect the explicit verb meaning in the text.
- If the link is mere co-occurrence, accompaniment, or underspecified connection, omit.

attribute (state|type|material|texture|shape|size|color)
- Create only when the attribute is explicitly stated AND passes the Objective-checkable gate.
- type: Only categorical identity labels that define what S is, not evaluative labels.
- color: Only literal color descriptors used as factual identification, not aesthetic or comparative color language.
- size: Only when expressed as an explicit measurement or as a comparison with an explicit reference object in the text.
- shape: Only when the term denotes a well-defined form with clear boundaries and stable visual criteria.
- material: Only physical substance claims with stable visual criteria; omit claims that depend on inference or perceived quality.
- texture: Only surface-pattern or surface-structure claims that are visually identifiable with stable criteria; omit purely aesthetic surface descriptions.
- state: Only discrete states with unambiguous visual markers; omit ambiguous or interpretive states.

other - text
- Create only when the exact displayed text string is explicitly provided.
- TEXT must match exactly the string in the prompt.
- If text presence is mentioned without the exact string, omit.

other - count
- Create only when an exact integer N is explicitly stated.
- Do not infer counts from plurals or vague quantifiers.

global - style
- Create only when STYLE is explicitly stated and passes the Objective-checkable gate.
- STYLE must be a discrete, visually identifiable rendering modality with stable criteria.
- Omit style language that encodes aesthetics, quality, realism level, resolution, detail intensity, mood, camera, or lighting.

[Output Constraints]
- Output only tuple lines in the exact formats above.
- No extra whitespace lines.
""".strip()

############################ Step 1, 3 Naive Reward ############################
REASONGEN_R1_TEMPLATE = """You are given a text prompt: \"{prompt}\" 
Below is one generated image:
<image>

1. Describe the image thoroughly (objects, colors, layout, etc.), do not be affected by the prompt.
2. Identify key visual elements and instructions from the prompt.
3. Evaluate how well the image follows the prompt:
   - Are all required elements present?
   - Are object counts, colors, and positions accurate?

Be extremly strict and precise:
Only if the image matches the prompt perfectly, respond with: \\boxed{{1}}
Otherwise, respond with: \\boxed{{0}}

Reason before your final boxed answer. Only one number should appear inside the box.
""".strip()

############################ Step 1, 3 Fine-Graine Reward ############################
TASK1_TASK3_IMAGE_GENERATOR_SYSTEM_PROMPT_TEMPLATE = r"""
You are a VQA assistant. The user provides a single image and multiple questions in the following exact input format:

[IMAGE]:
<input image here>

[QUESTIONS]:
<id> | <question>
<id> | <question>

Rules:
1) Visual-only: decide from what is visible. No typicality/context inference and no external verification.
2) YES gate: answer "Yes" only if the Reason cites at least one specific visible part/structure AND its location (e.g., "wheels under the fuselage"). If you cannot cite this, answer "No".
3) Visibility gating: for attributes, the entity must be visible; for relations, BOTH entities must be visible; otherwise Answer must be "No" and the Reason must mention what is not visible.
4) Scope: do not add attributes/states not asked.
5) Consistency: the Answer must be forced by the Reason.

Relation Rules:

Frame:
- All relations use the camera perspective.

2D relations:
- A left/right B: A must be clearly left/right of B.
- A above/below B: A must be above/below B.
- A (on the) top of B means the same as A above B.
- A (on the) bottom of B means the same as A below B (NOT inside/underside; no contact required).

Proximity (NO overlap):
- A (on the) side of / next to / near B: A and B must NOT overlap.
- side of / next to: very close. near: close but can be farther.

3D relations:
- A in front of / behind / hidden by B: overlap NOT required; be as close as possible; slight overlap allowed.
- Both A and B must remain visible (do not make either fully invisible).
- A in front of B: A appears closer to camera than B.
- A behind B / A hidden by B: A appears farther from the camera than B, so B appears in front of A.

Output only the required lines, in order, with no extra text or blank lines.

[Input]
Image:
<image>

Questions:
{questions}
""".strip()

############################ Step 2 Summarize Reward ############################
TASK2_COMPARISON_SUMMARIZE_SYSTEM_PROMPT = r"""
[Role]
You are an evaluator for canonical prompt summarization used for tuple decomposition, VQA verification, and edit-instruction generation.

[Goal]
Given a long descriptive prompt and its summarization, evaluate whether the summarization is a short canonical prompt that preserves only explicitly stated, tuple-extractable facts that can be verified via VQA.
The summarization must omit any content that cannot be expressed as valid tuples under the tuple schema, and must not introduce any new facts.

[Input]
The user provides two fields in plain text:
PROMPT: <long prompt>
SUMMARY: <short summary>

[VQA-checkable Definition (Schema-locked)]
A fact is VQA-checkable only if all are true:
- It is explicitly stated in the long prompt.
- It can be converted into at least one valid tuple under the tuple schema.
- It can be verified from the image with a clear visual criterion and a closed answer set.
- It does not rely on aesthetics, quality, mood, realism level, camera language, lighting, or subjective judgment.

[Tuple Schema (allowed facts)]
The summary should preserve only facts that fit one of these tuple types:

entity - whole (X)
- X is a concrete, depictable entity explicitly mentioned as a noun/noun phrase.
- Include X only if it is central OR participates in any other tuple.

entity - part (OWNER PART)
- Only if an explicit part-of relation is stated (possessive phrasing or “PART of OWNER”).
- Do not add typical parts.

relation - spatial (A, B, rel_token)
- Only if a physical placement relation is explicitly stated between A and B.
- Direction: (A, B, rel_token) means A is rel_token relative to B.
- Exclude non-spatial relations (possession, identity, association, feature/function).

action - (A, action_token, B)
- Only if an explicit verb/action meaning links A to B.
- If the link is mere co-occurrence or underspecified, omit.

attribute - state (S, V)
attribute - type (S, V)
attribute - material (S, V)
attribute - texture (S, V)
attribute - shape (S, V)
attribute - size (S, V)
attribute - color (S, V)
- Only if explicitly stated AND objectively checkable as a factual visual claim.
- state: discrete states with clear visual markers; omit interpretive states.
- type: categorical identity labels; omit evaluative labels.
- material: physical substance claims with stable visual criteria; omit inference-based material guesses.
- texture: surface pattern/structure claims with stable criteria; omit aesthetic surface descriptions.
- shape: well-defined forms with clear boundaries; omit vague shape language.
- size: only explicit measurements or explicit comparisons with a stated reference object.
- color: literal color descriptors used as factual identification; omit aesthetic or comparative color language.

other - text (S, "TEXT")
- Only if the exact displayed text string is explicitly provided.
- If text presence is mentioned without the exact string, omit.

other - count (S, ==N)
- Only if an exact integer N is explicitly stated.
- Do not infer counts from plurals or vague quantifiers.

global - style (STYLE)
- Only if explicitly stated AND objectively checkable as a discrete rendering modality.
- Omit style language that encodes aesthetics, quality, realism level, resolution, detail intensity, mood, camera, or lighting.

[Keep]
- Only facts that satisfy the Schema-locked VQA-checkable definition.
- Only entities required to express preserved relations/actions/attributes/text/count.

[Remove]
- Any fact that cannot be expressed as valid tuples under the schema.
- Any subjective, aesthetic, quality, vibe, mood, camera, lighting, realism-level language.
- Any implied facts, typical parts, inferred relations, inferred counts.

[IMPORTANT Facts Definition (Schema-locked)]
Only the following may be listed as missing_fact:
- A main entity required to form any valid tuple above.
- An explicitly stated, identity-defining attribute that maps to an allowed attribute tuple above.
- An explicitly stated exact integer count that maps to other - count.
- An explicitly stated primary spatial relation that maps to relation - spatial.
- An explicitly stated literal text string that maps to other - text.

[Error Types]
Use only these tags, ordered by severity (most severe first):
hallucinated_fact
missing_main_entity
missing_main_relation
missing_key_attribute
count_error
too_verbose
not_canonical

Tag criteria:
- hallucinated_fact: SUMMARY includes any fact not explicitly stated in PROMPT, or any fact not representable under the tuple schema.
- missing_main_entity: SUMMARY omits a required entity needed to express preserved tuple-extractable facts.
- missing_main_relation: SUMMARY omits an explicitly stated primary spatial relation representable as relation - spatial.
- missing_key_attribute: SUMMARY omits an explicitly stated, identity-defining attribute representable as an attribute tuple.
- count_error: SUMMARY invents/changes/infers an integer count, or omits an explicitly stated integer count.
- too_verbose: SUMMARY includes non-essential content beyond schema-locked, tuple-extractable facts.
- not_canonical: SUMMARY uses phrasing that harms tuple extraction (ambiguous references, bundled facts, inconsistent naming).

[Scoring]
Initialize score = 1.00
- hallucinated_fact: -0.50
- missing_main_entity: -0.50 each
- missing_main_relation: -0.20
- missing_key_attribute: -0.20 each
- count_error: -0.20
- too_verbose: -0.10
- not_canonical: -0.10
Clamp score to [0, 1], round to 2 decimals.

[Output Format]
Output plain text only, with the following exact sections and order:

RATIONALE: <a short explanation of why this score was assigned, referencing only the applicable errors and key omissions/additions>
ERROR: <comma-separated tags or NONE>
DEDUCTION: <explicit arithmetic showing applied penalties and the final score>
SCORE: \boxed{<score>}

Rules:
- Do not output JSON.
- Do not output explanations beyond the required fields.
- Each missing_fact and extra_fact bullet must describe exactly one fact in one short sentence.
""".strip()

############################ Step 2 Tuple Comparision Reward ############################
TASK2_COMPARISON_TUPLE_SYSTEM_PROMPT = r"""
[Role]
You are an evaluator for the tuple-decomposition stage in a pipeline that converts prompts into tuples, then into VQA checks, then into targeted edit instructions.

[Goal]
Given GT (ground truth) tuples and PRED (predicted) tuples, evaluate how accurately PRED reproduces the GT tuple set required for downstream VQA and edits.
Use schema-locked matching. Score using F1 to reflect both missing tuples and extra tuples.

[Input]
Plain text with two sections:

GT:
<one tuple per line>

PRED:
<one tuple per line>

Do not assume JSON. Do not assume ids.

[Tuple Schema (Schema-locked)]
A tuple line must be exactly one of the following forms:

entity - whole (X)
- X is a concrete, depictable entity explicitly mentioned as a noun/noun phrase.
- Include X only if it is central OR participates in any other tuple.

entity - part (OWNER PART)
- Only if an explicit part-of relation is stated (possessive phrasing or “PART of OWNER”).
- Do not add typical parts.

relation - spatial (A, B, rel_token)
- Only if a physical placement relation is explicitly stated between A and B.
- Direction: (A, B, rel_token) means A is rel_token relative to B.
- Exclude non-spatial relations (possession, identity, association, feature/function).

action - (A, action_token, B)
- Only if an explicit verb/action meaning links A to B.
- If the link is mere co-occurrence or underspecified, omit.

attribute - state (S, V)
attribute - type (S, V)
attribute - material (S, V)
attribute - texture (S, V)
attribute - shape (S, V)
attribute - size (S, V)
attribute - color (S, V)
- Only if explicitly stated AND objectively checkable as a factual visual claim.
- state: discrete states with clear visual markers; omit interpretive states.
- type: categorical identity labels; omit evaluative labels.
- material: physical substance claims with stable visual criteria; omit inference-based material guesses.
- texture: surface pattern/structure claims with stable criteria; omit aesthetic surface descriptions.
- shape: well-defined forms with clear boundaries; omit vague shape language.
- size: only explicit measurements or explicit comparisons with a stated reference object.
- color: literal color descriptors used as factual identification; omit aesthetic or comparative color language.

other - text (S, "TEXT")
- Only if the exact displayed text string is explicitly provided.
- If text presence is mentioned without the exact string, omit.

other - count (S, ==N)
- Only if an exact integer N is explicitly stated.
- Do not infer counts from plurals or vague quantifiers.

global - style (STYLE)
- Only if explicitly stated AND objectively checkable as a discrete rendering modality.
- Omit style language that encodes aesthetics, quality, realism level, resolution, detail intensity, mood, camera, or lighting.


[Schema Validity]
A PRED line that does not conform to one of the schema forms above is schema-invalid.
Schema-invalid PRED lines are always counted as wrong and cannot match any GT tuple.

[VQA-checkable Gate (Schema-locked)]
For attribute and global - style, treat subjective, aesthetic, mood, lighting, camera, realism-level, or quality language as invalid for this task.
If such language appears as an attribute value or style, count the tuple as schema-invalid.

[Normalization and Canonicalization]
Before matching, normalize both GT and PRED tuple lines using only the rules below.

General:
- Trim leading/trailing whitespace.
- Collapse internal runs of whitespace to a single space.
- Standardize punctuation spacing so separators and commas are not affected by extra spaces.

Entity surface forms (X, OWNER, PART, A, B, S):
- Lowercase.
- Remove leading articles.
- Singularize common noun heads so singular/plural variants normalize to the same form.
- Normalize simple inflections so obvious morphological variants normalize to the same base form.
- Keep the core noun phrase meaning; do not rewrite into a different entity.

Attribute values, action_token, and STYLE:
- Lowercase.
- Normalize simple inflections so obvious morphological variants normalize to the same base form.
- Collapse clearly equivalent wording into a single canonical form when the factual meaning is unchanged and VQA-checkable under the schema.

Spatial relations (rel_token):
- Lowercase.

Text and counts:
- other - text: preserve the exact characters inside quotes; do not apply synonyming, singularization, or case folding to quoted TEXT.
- other - count: normalize N to an integer value when possible; treat equivalent integer forms as equal.

No other paraphrase matching is allowed beyond the normalization rules above.

[Matching Rules]
- Match tuples by comparing their fully normalized canonical tuple lines.
- One-to-one matching: each GT tuple can match at most one PRED tuple.
- Perform matching greedily in PRED order: each PRED tuple matches the first unmatched equivalent GT tuple.
- Duplicates in PRED: if multiple PRED tuples canonicalize to the same GT tuple, only one can be matched; the rest are extra.
- Argument order is strict for ordered tuples. Do not swap arguments.

[Counts]
Let:
- total_gt = number of GT lines
- total_pred = number of PRED lines
- schema_invalid = number of PRED lines that are schema-invalid
- matched = number of PRED lines that match a unique GT line after normalization
- wrong = total_pred - matched


Compute:
- Precision = matched / total_pred, if total_pred > 0 else 0
- Recall = matched / total_gt, if total_gt > 0 else 0
- F1 = 2 * Precision * Recall / (Precision + Recall), if Precision + Recall > 0 else 0

Edge case:
- If total_gt == 0 and total_pred == 0, set Precision = 1, Recall = 1, F1 = 1.

[Score]
SCORE is F1, clamped to [0, 1] and rounded to 2 decimals.
acceptable is YES if SCORE >= 0.70, else NO.

[Output Format]
Output plain text only, with the following exact sections and order:

RATIONALE: <brief explanation mentioning only matched vs missing/extra/schema_invalid at a high level; do not list tuples>
DEDUCTION: <explicit arithmetic showing total_gt, total_pred, schema_invalid, matched, Precision, Recall, and F1 calculation>
SCORE: \boxed{<F1>}

Rules:
- Do not output JSON.
- Do not include per-tuple lists.
- Keep RATIONALE brief and factual.
""".strip()

############################ Step 2 VQA Accuracy Reward ############################
TASK2_HALLUCINATION_CHECK_SYSTEM_PROMPT = r"""
[Role]
You are an evaluator for the VQA-verification stage in a pipeline that uses semantic tuples to check image alignment and produce targeted edit instructions.

[Goal]
Given an image, a list of semantic tuples, and model-produced rationales that end with a binary label, evaluate whether each rationale-label pair is a valid, image-grounded VQA judgment for its corresponding tuple.
Your final output is an aggregate accuracy score over all pairs.

[Input]
The user provides three sections in plain text (NOT JSON):

IMAGE:
<an image is provided in the conversation>

TUPLES:
<multiline; each line: id | <tuple>>

ANSWERS:
<multiline; each line: <rationale text> Answer: Yes/No>

Constraints:
- TUPLES lines and ANSWERS lines are aligned by order: the i-th tuple corresponds to the i-th answer line.
- Each answer line must end with exactly "Answer: Yes" or exactly "Answer: No".
- The rationale is the text before "Answer:".
- If the number of ANSWERS lines differs from the number of TUPLES lines, mark all unmatched items as wrong due to format_error.

[Tuple Schema (Schema-locked)]
A tuple line must be exactly one of the following forms:

entity - whole (X)
- X is a concrete, depictable entity explicitly mentioned as a noun/noun phrase.
- Include X only if it is central OR participates in any other tuple.

entity - part (OWNER PART)
- Only if an explicit part-of relation is stated (possessive phrasing or “PART of OWNER”).
- Do not add typical parts.

relation - spatial (A, B, rel_token)
- Only if a physical placement relation is explicitly stated between A and B.
- Direction: (A, B, rel_token) means A is rel_token relative to B.
- Exclude non-spatial relations (possession, identity, association, feature/function).

action - (A, action_token, B)
- Only if an explicit verb/action meaning links A to B.
- If the link is mere co-occurrence or underspecified, omit.

attribute - state (S, V)
attribute - type (S, V)
attribute - material (S, V)
attribute - texture (S, V)
attribute - shape (S, V)
attribute - size (S, V)
attribute - color (S, V)
- Only if explicitly stated AND objectively checkable as a factual visual claim.
- state: discrete states with clear visual markers; omit interpretive states.
- type: categorical identity labels; omit evaluative labels.
- material: physical substance claims with stable visual criteria; omit inference-based material guesses.
- texture: surface pattern/structure claims with stable criteria; omit aesthetic surface descriptions.
- shape: well-defined forms with clear boundaries; omit vague shape language.
- size: only explicit measurements or explicit comparisons with a stated reference object.
- color: literal color descriptors used as factual identification; omit aesthetic or comparative color language.

other - text (S, "TEXT")
- Only if the exact displayed text string is explicitly provided.
- If text presence is mentioned without the exact string, omit.

other - count (S, ==N)
- Only if an exact integer N is explicitly stated.
- Do not infer counts from plurals or vague quantifiers.

global - style (STYLE)
- Only if explicitly stated AND objectively checkable as a discrete rendering modality.
- Omit style language that encodes aesthetics, quality, realism level, resolution, detail intensity, mood, camera, or lighting.


[Tuple Semantics]
Treat each tuple as a claim about the image. The final label must indicate whether the claim is true in the image.
Respect tuple direction and arguments exactly for ordered tuples (relations/actions). Quoted text must be exact. Exact counts must be exact.

[Evaluation Rules]
For each i-th pair (tuple[i], answer[i]):

A pair is CORRECT if and only if ALL conditions hold:
1) format_ok:
   - answer[i] ends with exactly "Answer: Yes" or "Answer: No".
2) relevant:
   - the rationale addresses the exact semantics of tuple[i] (same entities, same attribute/relation/action, same direction, same quoted text, same exact count meaning).
3) grounded:
   - the rationale cites at least one concrete, observable visual cue from the image that supports the claim.
4) non_hallucinated:
   - the rationale does not invent specific visual evidence not visible in the image and does not contradict the image.
5) logically_aligned:
   - the rationale supports its own final Yes/No without internal contradiction.
6) label_correct:
   - the final "Answer: Yes/No" matches what is actually true in the image for tuple[i].

A pair is WRONG if any condition fails. Use these failure categories internally to decide correctness:
- format_error: missing/incorrect "Answer: Yes/No" ending, or missing/unmatched line due to length mismatch.
- unrelated: rationale does not address the tuple semantics.
- insufficient_evidence: rationale provides no concrete observable cue, or relies on guessing/typicality/prior knowledge.
- hallucination: rationale asserts specific visual evidence not present or contradicts the image.
- label_contradiction: rationale conflicts with its own final label.
- label_incorrect: label is wrong for the image and tuple.

[Scoring]
Let:
- total = number of TUPLES lines
- num_correct = number of CORRECT pairs
- num_wrong = total - num_correct
- accuracy = num_correct / total, with accuracy = 0 if total == 0
Round accuracy to 2 decimals.

[Output Format]
Output plain text only, with the following exact sections and order:

RATIONALE: <brief summary of the dominant failure modes if any (format vs relevance vs grounding vs hallucination vs label correctness), without listing individual items>
DEDUCTION: <explicit arithmetic showing total, num_correct, num_wrong, and accuracy computation>
SCORE: \boxed{<accuracy>}

Rules:
- Do not output JSON.
- Do not output per-item lists.
- Keep RATIONALE brief and factual.
""".strip()


############################ Step 2 VQA Feedback Reward ############################
TASK3_FEEDBACK_REWARD_SYSTEM_PROMPT = r"""
[Role]
You are an evaluator for the feedback-generation stage in a pipeline that uses semantic tuples and VQA verification to locate image–prompt misalignment and produce targeted edit instructions.

[Goal]
Given the original prompt, semantic tuples, VQA results, and a generated multi-step feedback (edit plan), assign a reward that reflects whether the feedback will:
- Fix only the misaligned facts (those labeled Answer: No) so they become satisfied (Yes),
- Preserve already-aligned facts (those labeled Answer: Yes) without breaking them,
- Move the image toward better alignment with the original prompt and tuple set,
while remaining grounded, non-hallucinated, atomic, and actionable.

[Target Policy]
- The feedback must propose edits only to address tuples whose VQA label is Answer: No.
- The feedback must not propose edits that would plausibly flip any Answer: Yes tuple into No.
- If the feedback modifies any content that is evidence for an Answer: Yes tuple, it is a non-regression violation unless it explicitly preserves that tuple’s semantics.

[Input]
Plain text with four sections:

PROMPT:
<original long prompt>

TUPLES:
<id | tuple>

VQA_RESULTS:
<id | Answer: Yes/No | rationale>

FEEDBACK:
<multi-step feedback/edit plan text>

Constraints:
- Every tuple id in TUPLES appears exactly once in VQA_RESULTS.
- VQA_RESULTS are the only allowed grounding source; do not assume additional image facts not supported by VQA rationales.

[Tuple Schema (Schema-locked)]
A tuple line must be exactly one of the following forms:

entity - whole (X)
- X is a concrete, depictable entity explicitly mentioned as a noun/noun phrase.
- Include X only if it is central OR participates in any other tuple.

entity - part (OWNER PART)
- Only if an explicit part-of relation is stated (possessive phrasing or “PART of OWNER”).
- Do not add typical parts.

relation - spatial (A, B, rel_token)
- Only if a physical placement relation is explicitly stated between A and B.
- Direction: (A, B, rel_token) means A is rel_token relative to B.
- Exclude non-spatial relations (possession, identity, association, feature/function).

action - (A, action_token, B)
- Only if an explicit verb/action meaning links A to B.
- If the link is mere co-occurrence or underspecified, omit.

attribute - state (S, V)
attribute - type (S, V)
attribute - material (S, V)
attribute - texture (S, V)
attribute - shape (S, V)
attribute - size (S, V)
attribute - color (S, V)
- Only if explicitly stated AND objectively checkable as a factual visual claim.
- state: discrete states with clear visual markers; omit interpretive states.
- type: categorical identity labels; omit evaluative labels.
- material: physical substance claims with stable visual criteria; omit inference-based material guesses.
- texture: surface pattern/structure claims with stable criteria; omit aesthetic surface descriptions.
- shape: well-defined forms with clear boundaries; omit vague shape language.
- size: only explicit measurements or explicit comparisons with a stated reference object.
- color: literal color descriptors used as factual identification; omit aesthetic or comparative color language.

other - text (S, "TEXT")
- Only if the exact displayed text string is explicitly provided.
- If text presence is mentioned without the exact string, omit.

other - count (S, ==N)
- Only if an exact integer N is explicitly stated.
- Do not infer counts from plurals or vague quantifiers.

global - style (STYLE)
- Only if explicitly stated AND objectively checkable as a discrete rendering modality.
- Omit style language that encodes aesthetics, quality, realism level, resolution, detail intensity, mood, camera, or lighting.

[Schema-locked Tuple Semantics]
Treat each tuple as a factual constraint:
- entity / part / spatial relation / action direction / attribute type / quoted TEXT / exact count must be preserved exactly.
- Quoted text must match exactly if FEEDBACK proposes displayed text.
- Exact counts must match exactly if FEEDBACK proposes counts.

[Evaluation Criteria]
Assign reward by judging FEEDBACK against these requirements:

1) No-only Targeting (required)
- FEEDBACK must address all and only the tuples with Answer: No.
- Penalize if it proposes changes unrelated to any No-labeled tuple.

2) Non-regression (required)
- FEEDBACK must not introduce changes that plausibly break any Answer: Yes tuple.
- Penalize if it changes entities/relations/attributes/text/counts that are already verified as Yes, or adds conflicting constraints.

3) Prompt/Tuple Alignment (required)
- Proposed edits must move the image toward satisfying the original PROMPT as encoded by the tuples.
- Penalize if it introduces new entities/attributes/text/counts/relations not present in PROMPT/TUPLES, or if it shifts semantics away from tuple meaning.

4) Grounding / Non-hallucination
- FEEDBACK must be justified by the VQA_RESULTS for No tuples and must not invent unseen evidence.
- Penalize if it cites specifics not supported by VQA rationales or adds speculative details.

5) Atomicity / Actionability
- Each edit step should be specific enough to be executed and should map cleanly to one tuple-level correction.
- Penalize if steps are vague, bundled, or dominated by aesthetics/lighting/style instructions not needed to satisfy tuples.

[Scoring]
Initialize score = 1.00

Major penalties:
- targets_yes_or_breaks_yes (non-regression violation): -0.50 each distinct Yes tuple plausibly affected
- hallucinated_edit_fact (introduces facts not in PROMPT/TUPLES or unsupported by VQA): -0.50 each distinct invented fact
- wrong_text_or_count (proposed text/count mismatches tuple): -0.50 each

Moderate penalties:
- misses_no_tuple (fails to address a No-labeled tuple): -0.20 each
- edits_not_linked_to_no (edit content not tied to any No tuple): -0.20 each distinct edit chunk
- semantics_mismatch (edit changes tuple meaning or direction): -0.20 each

Minor penalties:
- too_verbose (excessive style/lighting/quality instructions not required for tuple satisfaction): -0.10
- not_actionable (too vague to execute or not decomposable into tuple fixes): -0.10

Clamp score to [0, 1], round to 2 decimals.

[Output Format]
Output plain text only, with the following exact sections and order:

RATIONALE: <brief, factual summary emphasizing: does it fix No-only, avoid breaking Yes, and stay aligned to PROMPT/TUPLES; no long lists>
DEDUCTION: <explicit arithmetic listing applied penalties and final score>
SCORE: \boxed{<score>}

Rules:
- Do not output JSON.
- Do not output per-step or per-tuple detailed diagnostics beyond a brief summary.
""".strip()

INTEGRATED_PIPELINE_REWARD_SYSTEM_PROMPT = r"""
[Role]
You are an integrated reward judge for a prompt-alignment pipeline that operates as:
PROMPT -> SUMMARY -> TUPLE_DECOMPOSITION -> VQA -> FEEDBACK.
You output a single final reward score measuring end-to-end usefulness for generating targeted edits that improve prompt-image alignment.

[Goal]
Given the pipeline artifacts, assign one score in [0,1] that reflects whether:
- SUMMARY is a faithful, canonical reduction of PROMPT that keeps only tuple-extractable, VQA-checkable facts.
- PRED_TUPLES are a correct decomposition of SUMMARY (and therefore PROMPT) under the schema.
- VQA_RESULTS correctly evaluate each tuple against the image, with grounded rationales and correct Yes/No labels.
- FEEDBACK uses VQA_RESULTS to propose edits that turn No-labeled tuples into Yes-labeled tuples without breaking Yes-labeled tuples, and stays aligned to PROMPT/SUMMARY semantics.

[Input]
Plain text with these sections:

IMAGE:
<image provided in the conversation>

PROMPT:
<original prompt text>

SUMMARY:
<summary text produced from PROMPT>

PRED_TUPLES:
<one tuple per line; decomposition of SUMMARY>

VQA_RESULTS:
<one line per tuple, aligned by order with PRED_TUPLES; each ends with exactly "Answer: Yes" or "Answer: No">

FEEDBACK:
<multi-step edit feedback text derived from VQA_RESULTS>
or exactly the single line:
No need to generate feedback.

Constraints:
- PRED_TUPLES and VQA_RESULTS must have the same number of lines and be aligned by order.

[Tuple Schema (Schema-locked)]
A tuple line must be exactly one of the following forms:

entity - whole (X)
- X is a concrete, depictable entity explicitly mentioned as a noun/noun phrase.
- Include X only if it is central OR participates in any other tuple.

entity - part (OWNER PART)
- Only if an explicit part-of relation is stated (possessive phrasing or “PART of OWNER”).
- Do not add typical parts.

relation - spatial (A, B, rel_token)
- Only if a physical placement relation is explicitly stated between A and B.
- Direction: (A, B, rel_token) means A is rel_token relative to B.
- Exclude non-spatial relations (possession, identity, association, feature/function).

action - (A, action_token, B)
- Only if an explicit verb/action meaning links A to B.
- If the link is mere co-occurrence or underspecified, omit.

attribute - state (S, V)
attribute - type (S, V)
attribute - material (S, V)
attribute - texture (S, V)
attribute - shape (S, V)
attribute - size (S, V)
attribute - color (S, V)
- Only if explicitly stated AND objectively checkable as a factual visual claim.
- state: discrete states with clear visual markers; omit interpretive states.
- type: categorical identity labels; omit evaluative labels.
- material: physical substance claims with stable visual criteria; omit inference-based material guesses.
- texture: surface pattern/structure claims with stable criteria; omit aesthetic surface descriptions.
- shape: well-defined forms with clear boundaries; omit vague shape language.
- size: only explicit measurements or explicit comparisons with a stated reference object.
- color: literal color descriptors used as factual identification; omit aesthetic or comparative color language.

other - text (S, "TEXT")
- Only if the exact displayed text string is explicitly provided.
- If text presence is mentioned without the exact string, omit.

other - count (S, ==N)
- Only if an exact integer N is explicitly stated.
- Do not infer counts from plurals or vague quantifiers.

global - style (STYLE)
- Only if explicitly stated AND objectively checkable as a discrete rendering modality.
- Omit style language that encodes aesthetics, quality, realism level, resolution, detail intensity, mood, camera, or lighting.

[VQA-checkable (Schema-locked)]
A fact is eligible only if it can be represented by a valid tuple under the schema and judged from the image with clear visual criteria.
Exclude subjective/aesthetic/mood/lighting/camera/quality language from SUMMARY, tuples, VQA rationales, and FEEDBACK as primary claims.

[Evaluation Procedure]
Compute four stage scores in [0,1]:
S1 = PROMPT -> SUMMARY
S2 = SUMMARY -> PRED_TUPLES
S3 = PRED_TUPLES -> VQA_RESULTS (vs IMAGE)
S4 = VQA_RESULTS -> FEEDBACK

Geometric mean aggregation:
- Let eps = 0.05
- FinalScore = (max(S1, eps) * max(S2, eps) * max(S3, eps) * max(S4, eps)) ** 0.25

[Stage 1: PROMPT -> SUMMARY (S1)]
SUMMARY should keep only tuple-extractable, VQA-checkable facts explicitly stated in PROMPT and add nothing new.
Compute S1 as:

S1 = 1.00
- hallucination: If SUMMARY adds any semantic fact not explicitly stated in PROMPT: S1 -= 0.50
- missing_core: If any tuple-extractable core facts are missing (missing_core > 0): S1 -= 0.50
- not_canonical: If SUMMARY is not canonical (includes subjective/aesthetic/mood/lighting/camera/quality claims, or uses bundled/ambiguous phrasing): S1 -= 0.40
Clamp S1 to [0,1].

[Stage 2: SUMMARY -> PRED_TUPLES (S2)]
Let T be the number of PRED_TUPLES lines.
If T == 0, set S2 = 0.00.
Otherwise compute S2 using only rate-based penalties and strictly syntax-based schema validity.

Schema-invalid definition (syntax-only):
- A tuple line is schema-invalid if and only if its raw text line does NOT match one of the allowed tuple schemas exactly in this prompt.
- Do NOT use semantic judgment to decide schema validity. Do NOT mark a tuple schema-invalid because an entity seems odd, non-central, or undesirable.
- rel_token must be a short spatial preposition phrase, not a full sentence.
- attribute values that are subjective or not VQA-checkable (e.g., aesthetic/quality/vibe terms) make the tuple schema-invalid.

Count terms:
- invalid_count = number of schema-invalid tuple lines (syntax-only).
- invalid_rate = invalid_count / T.

Unsupported-by-summary (tuple hallucination):
- halluc_count = number of schema-valid tuples whose factual claim is not supported by SUMMARY (i.e., introduces a new entity/relation/attribute/action/text/count not stated in SUMMARY).
- halluc_rate = halluc_count / T.

Missing-from-summary decomposition:
- Let summary_fact_count = number of explicit tuple-extractable facts stated in SUMMARY.
  Only count facts that can be represented by a valid tuple under the schema.
- Let missing_fact_count = number of those SUMMARY facts that are not represented by any schema-valid tuple in PRED_TUPLES.
- For spatial facts, a missing_fact occurs when SUMMARY states a spatial relation but PRED_TUPLES contains no corresponding relation - spatial tuple for it.
- missing_summary_rate = missing_fact_count / max(1, summary_fact_count).

Canonicality:
- For support/missing checks, normalize entity surface forms by lowercasing and treating singular/plural variants as equivalent.
- not_canonical_flag = 1 only if the same entity is referred to by genuinely different names across tuples (not singular/plural variants),
  or ordered arguments are ambiguous in a way that harms downstream VQA/editing; otherwise 0.

Compute:
S2 = 1.00
S2 -= 0.80 * missing_summary_rate
S2 -= 0.70 * invalid_rate
S2 -= 0.60 * halluc_rate
S2 -= 0.50 * not_canonical_flag
Clamp S2 to [0,1].
All tuple types are treated as equally important.

Stage 2 RATIONALE sentence requirement:
- Mention which rates drove S2 using the names: invalid_rate, halluc_rate, missing_summary_rate, not_canonical_flag.


[Stage 3: PRED_TUPLES -> VQA_RESULTS (S3)]
If the number of VQA_RESULTS lines does not equal T, set S3 = 0.00.
Else if T == 0, set S3 = 0.00.
Else:

For each aligned pair (tuple[i], vqa[i]), mark it CORRECT only if all hold:
- format_ok: ends with exactly "Answer: Yes" or "Answer: No"
- relevant: addresses the exact tuple semantics (entities, attribute/relation/action, direction, quoted text, exact count)
- grounded: cites at least one concrete observable cue from IMAGE
- non_hallucinated: does not invent evidence or contradict IMAGE
- logically_aligned: rationale supports its own label
- label_correct: label matches what is true in IMAGE for the tuple; "Answer: No" is CORRECT when the tuple’s claim is not satisfied in the image (i.e., the required entity/attribute/relation/action/text/count specified by the tuple is absent or not met)

Let C be the number of CORRECT pairs.
S3 = C / T
Clamp S3 to [0,1].

[Stage 4: VQA_RESULTS -> FEEDBACK (S4)]
Let num_no be the number of tuples labeled "Answer: No".
Let num_yes be the number of tuples labeled "Answer: Yes".
Let T be the total number of tuples.

Case A: FEEDBACK is exactly "No need to generate feedback."
- If num_no == 0: S4 = 1.00
- If num_no > 0: S4 = 0.00

Case B: FEEDBACK contains edit steps.

Define:
- no_fix_rate = (Number of No-labeled tuples clearly addressed by FEEDBACK) / max(1, num_no)
- yes_break_rate = (Number of Yes-labeled tuples plausibly broken by FEEDBACK) / max(1, num_yes)

AlignScore (predicted post-edit PROMPT alignment):
- For each tuple i, define predicted_post_label[i] using FEEDBACK:
  - If FEEDBACK explicitly enforces tuple[i] (i.e., would plausibly make it true), set predicted_post_label[i] = Yes.
  - Else if FEEDBACK explicitly contradicts tuple[i] (i.e., would plausibly make it false), set predicted_post_label[i] = No.
  - Else set predicted_post_label[i] = the original VQA label for tuple[i].
- align_score = (Number of tuples with predicted_post_label == Yes) / max(1, T)

- fb_not_actionable_flag = 1 if FEEDBACK is too vague or too bundled to map to tuple-level fixes, else 0

Compute:
S4 = 1.00
S4 -= 0.90 * (1 - no_fix_rate)
S4 -= 0.90 * (1 - align_score)
S4 -= 0.80 * yes_break_rate
S4 -= 0.50 * fb_not_actionable_flag
Clamp S4 to [0,1].

[Stage 4 Clarification]
- A No-labeled tuple counts as addressed only if FEEDBACK proposes an edit that would plausibly make that tuple true.
- yes_break_rate measures plausible regressions of previously Yes-labeled tuples.
- AlignScore evaluates predicted post-edit tuple satisfaction using the rule-based predicted_post_label defined above.
- If FEEDBACK changes the scene semantics away from PROMPT, treat align_score as 0.00.

[Output JSON Only]
Return exactly ONE JSON object and nothing else.

The JSON object MUST contain exactly these 6 top-level keys and NO others:
s1, s2, s3, s4, final_score, dominant_stage

Each stage value must be an object with exactly the required keys below (no extras):

s1 keys: hallucination, missing_core, not_canonical, issue, deduction, score
s2 keys: invalid_rate, halluc_rate, missing_summary_rate, not_canonical, issue, deduction, score
s3 keys: correct_num, correct_den, dominant_failure, issue, deduction, score
s4 keys: num_no, no_fix_rate, yes_break_rate, fb_not_actionable_flag, align_score, issue, deduction, score

Formatting constraints:
- All scores/rates are floats in [0,1] rounded to 2 decimals.
- correct_num and correct_den are integers.
- dominant_failure must be one of: format, relevance, grounding, hallucination, label, none
- issue must be a single short clause (no multi-sentence explanation).
- deduction must be a single short string containing the exact arithmetic used to compute the stage score from the reported fields.

dominant_stage must be one of: s1, s2, s3, s4 and should be the stage with the lowest score (break ties by earliest stage).
""".strip()

## LEGACY
INTEGRATED_PIPELINE_REWARD_SYSTEM_PROMPT_LEGACY = r"""
[Role]
You are an integrated reward judge for a prompt-alignment pipeline that operates as:
PROMPT -> SUMMARY -> TUPLE_DECOMPOSITION -> VQA -> FEEDBACK.
You output a single final reward score measuring end-to-end usefulness for generating targeted edits that improve prompt-image alignment.

[Goal]
Given the pipeline artifacts, assign one score in [0,1] that reflects whether:
- SUMMARY is a faithful, canonical reduction of PROMPT that keeps only tuple-extractable, VQA-checkable facts.
- PRED_TUPLES are a correct decomposition of SUMMARY (and therefore PROMPT) under the schema.
- VQA_RESULTS correctly evaluate each tuple against the image, with grounded rationales and correct Yes/No labels.
- FEEDBACK uses VQA_RESULTS to propose edits that turn No-labeled tuples into Yes-labeled tuples without breaking Yes-labeled tuples, and stays aligned to PROMPT/SUMMARY semantics.

[Input]
Plain text with these sections:

IMAGE:
<image provided in the conversation>

PROMPT:
<original prompt text>

SUMMARY:
<summary text produced from PROMPT>

PRED_TUPLES:
<one tuple per line; decomposition of SUMMARY>

VQA_RESULTS:
<one line per tuple, aligned by order with PRED_TUPLES; each ends with exactly "Answer: Yes" or "Answer: No">

FEEDBACK:
<multi-step edit feedback text derived from VQA_RESULTS>
or exactly the single line:
No need to generate feedback.

Constraints:
- PRED_TUPLES and VQA_RESULTS must have the same number of lines and be aligned by order.

[Tuple Schema (Schema-locked)]
A tuple line must be exactly one of the following forms:

entity - whole (X)
- X is a concrete, depictable entity explicitly mentioned as a noun/noun phrase.
- Include X only if it is central OR participates in any other tuple.

entity - part (OWNER PART)
- Only if an explicit part-of relation is stated (possessive phrasing or “PART of OWNER”).
- Do not add typical parts.

relation - spatial (A, B, rel_token)
- Only if a physical placement relation is explicitly stated between A and B.
- Direction: (A, B, rel_token) means A is rel_token relative to B.
- Exclude non-spatial relations (possession, identity, association, feature/function).

action - (A, action_token, B)
- Only if an explicit verb/action meaning links A to B.
- If the link is mere co-occurrence or underspecified, omit.

attribute - state (S, V)
attribute - type (S, V)
attribute - material (S, V)
attribute - texture (S, V)
attribute - shape (S, V)
attribute - size (S, V)
attribute - color (S, V)
- Only if explicitly stated AND objectively checkable as a factual visual claim.
- state: discrete states with clear visual markers; omit interpretive states.
- type: categorical identity labels; omit evaluative labels.
- material: physical substance claims with stable visual criteria; omit inference-based material guesses.
- texture: surface pattern/structure claims with stable criteria; omit aesthetic surface descriptions.
- shape: well-defined forms with clear boundaries; omit vague shape language.
- size: only explicit measurements or explicit comparisons with a stated reference object.
- color: literal color descriptors used as factual identification; omit aesthetic or comparative color language.

other - text (S, "TEXT")
- Only if the exact displayed text string is explicitly provided.
- If text presence is mentioned without the exact string, omit.

other - count (S, ==N)
- Only if an exact integer N is explicitly stated.
- Do not infer counts from plurals or vague quantifiers.

global - style (STYLE)
- Only if explicitly stated AND objectively checkable as a discrete rendering modality.
- Omit style language that encodes aesthetics, quality, realism level, resolution, detail intensity, mood, camera, or lighting.

[VQA-checkable (Schema-locked)]
A fact is eligible only if it can be represented by a valid tuple under the schema and judged from the image with clear visual criteria.
Exclude subjective/aesthetic/mood/lighting/camera/quality language from SUMMARY, tuples, VQA rationales, and FEEDBACK as primary claims.

[Evaluation Procedure]
Compute four stage scores in [0,1]:
S1 = PROMPT -> SUMMARY
S2 = SUMMARY -> PRED_TUPLES
S3 = PRED_TUPLES -> VQA_RESULTS (vs IMAGE)
S4 = VQA_RESULTS -> FEEDBACK

Geometric mean aggregation:
- Let eps = 0.05
- FinalScore = (max(S1, eps) * max(S2, eps) * max(S3, eps) * max(S4, eps)) ** 0.25

[Stage 1: PROMPT -> SUMMARY (S1)]
SUMMARY should keep only tuple-extractable, VQA-checkable facts explicitly stated in PROMPT and add nothing new.
Compute S1 as:

S1 = 1.00
- If SUMMARY adds any semantic fact not explicitly stated in PROMPT: S1 -= 0.50
- Let missing_core_rate = (Number of missing tuple-extractable core facts) / max(1, Number of tuple-extractable core facts in PROMPT)
  Then S1 -= 0.80 * missing_core_rate
- If SUMMARY is not canonical (includes subjective/aesthetic/mood/lighting/camera/quality claims, or uses bundled/ambiguous phrasing): S1 -= 0.40
Clamp S1 to [0,1].

[Stage 2: SUMMARY -> PRED_TUPLES (S2)]
Let T be the number of PRED_TUPLES lines.
If T == 0, set S2 = 0.00.
Otherwise compute S2 using only rate-based penalties and strictly syntax-based schema validity.

Schema-invalid definition (syntax-only):
- A tuple line is schema-invalid if and only if its raw text line does NOT match one of the allowed tuple schemas exactly in this prompt.
- Do NOT use semantic judgment to decide schema validity. Do NOT mark a tuple schema-invalid because an entity seems odd, non-central, or undesirable.
- rel_token must be a short spatial preposition phrase, not a full sentence.
- attribute values that are subjective or not VQA-checkable (e.g., aesthetic/quality/vibe terms) make the tuple schema-invalid.

Count terms:
- invalid_count = number of schema-invalid tuple lines (syntax-only).
- invalid_rate = invalid_count / T.

Unsupported-by-summary (tuple hallucination):
- halluc_count = number of schema-valid tuples whose factual claim is not supported by SUMMARY (i.e., introduces a new entity/relation/attribute/action/text/count not stated in SUMMARY).
- halluc_rate = halluc_count / T.

Missing-from-summary decomposition:
- Let summary_fact_count = number of explicit tuple-extractable facts stated in SUMMARY.
  Only count facts that can be represented by a valid tuple under the schema.
- Let missing_fact_count = number of those SUMMARY facts that are not represented by any schema-valid tuple in PRED_TUPLES.
- For spatial facts, a missing_fact occurs when SUMMARY states a spatial relation but PRED_TUPLES contains no corresponding relation - spatial tuple for it.
- missing_summary_rate = missing_fact_count / max(1, summary_fact_count).

Canonicality:
- For support/missing checks, normalize entity surface forms by lowercasing and treating singular/plural variants as equivalent.
- not_canonical_flag = 1 only if the same entity is referred to by genuinely different names across tuples (not singular/plural variants),
  or ordered arguments are ambiguous in a way that harms downstream VQA/editing; otherwise 0.

Compute:
S2 = 1.00
S2 -= 0.80 * missing_summary_rate
S2 -= 0.70 * invalid_rate
S2 -= 0.60 * halluc_rate
S2 -= 0.50 * not_canonical_flag
Clamp S2 to [0,1].
All tuple types are treated as equally important.

Stage 2 RATIONALE sentence requirement:
- Mention which rates drove S2 using the names: invalid_rate, halluc_rate, missing_summary_rate, not_canonical_flag.


[Stage 3: PRED_TUPLES -> VQA_RESULTS (S3)]
If the number of VQA_RESULTS lines does not equal T, set S3 = 0.00.
Else if T == 0, set S3 = 0.00.
Else:

For each aligned pair (tuple[i], vqa[i]), mark it CORRECT only if all hold:
- format_ok: ends with exactly "Answer: Yes" or "Answer: No"
- relevant: addresses the exact tuple semantics (entities, attribute/relation/action, direction, quoted text, exact count)
- grounded: cites at least one concrete observable cue from IMAGE
- non_hallucinated: does not invent evidence or contradict IMAGE
- logically_aligned: rationale supports its own label
- label_correct: label matches what is true in IMAGE for the tuple; "Answer: No" is CORRECT when the tuple’s claim is not satisfied in the image (i.e., the required entity/attribute/relation/action/text/count specified by the tuple is absent or not met)

Let C be the number of CORRECT pairs.
S3 = C / T
Clamp S3 to [0,1].

[Stage 4: VQA_RESULTS -> FEEDBACK (S4)]
Let num_no be the number of tuples labeled "Answer: No".
Let num_yes be the number of tuples labeled "Answer: Yes".
Let T be the total number of tuples.

Case A: FEEDBACK is exactly "No need to generate feedback."
- If num_no == 0: S4 = 1.00
- If num_no > 0: S4 = 0.00

Case B: FEEDBACK contains edit steps.

Define:
- no_fix_rate = (Number of No-labeled tuples clearly addressed by FEEDBACK) / max(1, num_no)
- yes_break_rate = (Number of Yes-labeled tuples plausibly broken by FEEDBACK) / max(1, num_yes)

AlignScore (predicted post-edit PROMPT alignment):
- For each tuple i, define predicted_post_label[i] using FEEDBACK:
  - If FEEDBACK explicitly enforces tuple[i] (i.e., would plausibly make it true), set predicted_post_label[i] = Yes.
  - Else if FEEDBACK explicitly contradicts tuple[i] (i.e., would plausibly make it false), set predicted_post_label[i] = No.
  - Else set predicted_post_label[i] = the original VQA label for tuple[i].
- align_score = (Number of tuples with predicted_post_label == Yes) / max(1, T)

- fb_not_actionable_flag = 1 if FEEDBACK is too vague or too bundled to map to tuple-level fixes, else 0

Compute:
S4 = 1.00
S4 -= 0.90 * (1 - no_fix_rate)
S4 -= 0.90 * (1 - align_score)
S4 -= 0.80 * yes_break_rate
S4 -= 0.50 * fb_not_actionable_flag
Clamp S4 to [0,1].

[Stage 4 Clarification]
- A No-labeled tuple counts as addressed only if FEEDBACK proposes an edit that would plausibly make that tuple true.
- yes_break_rate measures plausible regressions of previously Yes-labeled tuples.
- AlignScore evaluates predicted post-edit tuple satisfaction using the rule-based predicted_post_label defined above.
- If FEEDBACK changes the scene semantics away from PROMPT, treat align_score as 0.00.

[Output JSON Only]
Return exactly ONE JSON object and nothing else.

The JSON object MUST contain exactly these 6 top-level keys and NO others:
s1, s2, s3, s4, final_score, dominant_stage

Each stage value must be an object with exactly the required keys below (no extras):

s1 keys: hallucination, missing_core_rate, not_canonical, issue, deduction, score
s2 keys: invalid_rate, halluc_rate, missing_summary_rate, not_canonical, issue, deduction, score
s3 keys: correct_num, correct_den, dominant_failure, issue, deduction, score
s4 keys: num_no, no_fix_rate, yes_break_rate, fb_not_actionable_flag, align_score, issue, deduction, score

Formatting constraints:
- All scores/rates are floats in [0,1] rounded to 2 decimals.
- correct_num and correct_den are integers.
- dominant_failure must be one of: format, relevance, grounding, hallucination, label, none
- issue must be a single short clause (no multi-sentence explanation).
- deduction must be a single short string containing the exact arithmetic used to compute the stage score from the reported fields.

dominant_stage must be one of: s1, s2, s3, s4 and should be the stage with the lowest score (break ties by earliest stage).
""".strip()
import re
import torch
import spacy

_TUPLE_SCHEMA_PATTERNS = [
    re.compile(r'^entity\s*-\s*whole\s*\([^,)]+\)$'),                          # single entity, no comma
    re.compile(r'^entity\s*-\s*part\s*\([^,)]+\)$'),                           # flat string, no comma
    re.compile(r'^relation\s*-\s*spatial\s*\([^,)]+,\s*[^,)]+,\s*[^,)]+\)$'),  # (A, B, rel_token)
    re.compile(r'^action\s*-\s*\([^,)]+,\s*[^,)]+,\s*[^,)]+\)$'),             # (A, action_token, B)
    re.compile(r'^attribute\s*-\s*state\s*\([^,)]+,\s*[^,)]+\)$'),
    re.compile(r'^attribute\s*-\s*type\s*\([^,)]+,\s*[^,)]+\)$'),
    re.compile(r'^attribute\s*-\s*material\s*\([^,)]+,\s*[^,)]+\)$'),
    re.compile(r'^attribute\s*-\s*texture\s*\([^,)]+,\s*[^,)]+\)$'),
    re.compile(r'^attribute\s*-\s*shape\s*\([^,)]+,\s*[^,)]+\)$'),
    re.compile(r'^attribute\s*-\s*size\s*\([^,)]+,\s*[^,)]+\)$'),
    re.compile(r'^attribute\s*-\s*color\s*\([^,)]+,\s*[^,)]+\)$'),
    re.compile(r'^other\s*-\s*text\s*\([^,)]+,\s*[^,)]+\)$'),                 # (S, TEXT), comma 필수
    re.compile(r'^other\s*-\s*count\s*\([^,)]+,\s*==\d+\)$'),
    re.compile(r'^global\s*-\s*style\s*\([^,)]+\)$'),                          # single style token, no comma
]

class FormattingEvaluatorV2:
    def __init__(self):
        # 4단계 구조를 위한 새로운 패턴 정의
        self.SECOND_PATTERN = "Second, Decompose summarize"
        self.THIRD_PATTERN = "Third, Verify that the decomposed elements align with the image."
        self.FOURTH_PATTERN = "Fourth, Generate corrective feedback."
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    def _split_text_into_parts(self, text):
        """
        전체 텍스트를 4단계로 분리합니다.
        Part 1: Summarize
        Part 2: Decompose summarize (Tuples)
        Part 3: Verify (Answers)
        Part 4: Feedback
        """
        m2 = re.search(re.escape(self.SECOND_PATTERN), text)
        m3 = re.search(re.escape(self.THIRD_PATTERN), text)
        m4 = re.search(re.escape(self.FOURTH_PATTERN), text)

        if m2 and m3 and m4:
            # 1단계: 시작 ~ Second 전까지
            part1 = text[:m2.start()].strip()
            # 2단계: Second ~ Third 전까지
            part2 = text[m2.start():m3.start()].strip().replace(self.SECOND_PATTERN, "").strip()
            # 3단계: Third ~ Fourth 전까지
            part3 = text[m3.start():m4.start()].strip().replace(self.THIRD_PATTERN, "").strip()
            # 4단계: Fourth ~ 끝까지
            part4 = text[m4.start():].strip().replace(self.FOURTH_PATTERN, "").strip()
            return part1, part2, part3, part4
        
        return None, None, None, None

    def _parse_part2(self, text_block):
        """기존 Decompose(Tuple) 파싱 로직 동일 (Part 2 전달용)"""
        if not text_block: return []
        lines = [line for line in text_block.split('\n') if line.strip()]
        parsed_lines = []
        for line in lines:
            try:
                num, contents = line.split('|', 1)
                parsed_lines.append((int(num.strip()), contents.strip()))
            except (ValueError, IndexError):
                continue
        return parsed_lines

    def _extract_answer_paragraphs(self, text_block):
        """기존 Verify(Yes/No) 추출 로직 동일 (Part 3 전달용)"""
        if not text_block: return []
        pattern = r'.*?Answer: (?:Yes|No)'
        paragraphs = re.findall(pattern, text_block, re.DOTALL)
        return [p.strip() for p in paragraphs]

    def _get_answer_from_paragraph(self, paragraph):
        match = re.search(r'Answer: (Yes|No)$', paragraph)
        return match.group(1) if match else None

    def check_all_answers_positive(self, paragraphs: list[str]) -> bool:
        """모든 답변이 Yes인지 확인"""
        answers = [self._get_answer_from_paragraph(para) for para in paragraphs]
        answers_int = [1 if ans.lower() == "yes" else 0 for ans in answers if ans is not None]
        return 1.0 if all(ans == 1 for ans in answers_int) and len(answers_int) > 0 else 0.0

    def _calculate_metrics(self, gt_part2, pred_part2, gt_answers, pred_paragraphs):
        """
        핵심 메트릭 계산 로직 (변경 없음)
        gt_part2/pred_part2: 파싱된 튜플 리스트
        gt_answers/pred_paragraphs: 질문/답변 문단 리스트
        """
        metrics = {}

        # 1. Part 1(Decompose) 내용 정확도
        gt_contents = {self._normalize_content(self.nlp, content) for _, content in gt_part2}
        pred_contents = {self._normalize_content(self.nlp, content) for _, content in pred_part2}
        correct_matches = len(gt_contents.intersection(pred_contents))
        total_gt = len(gt_part2)
        metrics['part1_accuracy'] = correct_matches / total_gt if total_gt > 0 else 0.0

        # 2. Part 2(Verify) 답변 정확도
        correct_answers = 0
        pred_content_to_index = {content: i for i, (_, content) in enumerate(pred_part2)}
        for gt_index, (_, gt_content) in enumerate(gt_part2):
            if gt_content in pred_content_to_index:
                pred_para_index = pred_content_to_index[gt_content]
                if pred_para_index < len(pred_paragraphs) and gt_index < len(gt_answers):
                    gt_ans = self._get_answer_from_paragraph(gt_answers[gt_index])
                    pred_ans = self._get_answer_from_paragraph(pred_paragraphs[pred_para_index])
                    if gt_ans and gt_ans == pred_ans:
                        correct_answers += 1
        
        metrics['part2_accuracy'] = correct_answers / total_gt if total_gt > 0 else 0.0
        metrics['part2_accuracy_only_matching'] = correct_answers / correct_matches if correct_matches > 0 else 0.0
        
        # 3. 형식 및 일관성 점수
        metrics['internal_consistency_ok'] = 1 if len(pred_part2) == len(pred_paragraphs) else 0
        metrics['part1_length_match_ok'] = 1 if len(gt_part2) == len(pred_part2) else 0

        return {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in metrics.items()}

    def _calculate_metrics_for_reward(self, gt_part2, pred_part2, pred_paragraphs):
        """
        정확도(Accuracy)와 형식 점수(0/1)를 함께 계산합니다.
        """
        metrics = {}
        
        # Part 1: F1 score
        gt_contents = {self._normalize_content(self.nlp, content) for _, content in gt_part2}
        pred_contents = {self._normalize_content(self.nlp, content) for _, content in pred_part2}
        correct_matches = len(gt_contents.intersection(pred_contents))
        total_gt = len(gt_part2)

        precision = correct_matches / len(pred_part2) if len(pred_part2) > 0 else 0.0
        recall = correct_matches / total_gt if total_gt > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics['task2_part2_accuracy'] = f1_score
        
        metrics['task2_internal_consistency_ok'] = 1 if (len(pred_part2) == len(pred_paragraphs)) and (len(pred_part2) != 0) and (len(pred_paragraphs) != 0) else 0

        final_metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
        
        return final_metrics

    def check_tuple_schema_ok(self, parsed_tuples: list) -> bool:
        """각 tuple content가 사전 정의된 스키마 중 하나와 일치하는지 검사.

        parsed_tuples: _parse_part2()가 반환한 (index, content) 리스트
        하나라도 스키마와 다르거나 리스트가 비어 있으면 False.
        """
        if not parsed_tuples:
            return False
        return all(
            any(p.match(content) for p in _TUPLE_SCHEMA_PATTERNS)
            for _, content in parsed_tuples
        )

    def check_feedback_step_format(self, feedback_text: str) -> bool:
        """각 non-empty 줄이 Step N:으로 시작하고 1부터 연속인지 확인"""
        if not feedback_text or not feedback_text.strip():
            return False
        lines = [l.strip() for l in feedback_text.split('\n') if l.strip()]
        step_re = re.compile(r'^Step\s*(\d+)\s*:', re.IGNORECASE)
        step_numbers = []
        for line in lines:
            m = step_re.match(line)
            if not m:
                return False
            step_numbers.append(int(m.group(1)))
        return step_numbers == list(range(1, len(step_numbers) + 1))

    def _normalize_content(self, nlp, text):
        if not text:
            return ""
        
        doc = nlp(text.lower())
        # 각 토큰의 기본형(lemma)을 추출하여 공백 없이 결합
        normalized = "".join([token.lemma_.strip() for token in doc if not token.is_space])
        return normalized


class FormattingEvaluatorV3:
    """
    V3 format evaluator.

    Response structure (no summarize step):
      <N | tuple lines>
      Second, Verify that the decomposed elements align with the image.
      <verification paragraphs ending with Answer: Yes/No>
      Third, Generate corrective feedback.
      <step-by-step feedback or 'No need to generate feedback.'>
    """

    SECOND_PATTERN = "Second, Verify that the decomposed elements align with the image."
    THIRD_PATTERN = "Third, Generate corrective feedback."

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    def _split_text_into_parts(self, text: str):
        """Returns (decompose, verify, feedback) or (None, None, None) if structure is missing."""
        m2 = re.search(re.escape(self.SECOND_PATTERN), text)
        m3 = re.search(re.escape(self.THIRD_PATTERN), text)
        if not (m2 and m3 and m2.start() < m3.start()):
            return None, None, None
        decompose = text[:m2.start()].strip()
        verify = text[m2.end():m3.start()].strip()
        feedback = text[m3.end():].strip()
        return decompose, verify, feedback

    def _parse_tuples(self, text_block: str) -> list:
        """Parse 'N | content' lines → [(int, str), ...]"""
        if not text_block:
            return []
        parsed = []
        for line in text_block.split('\n'):
            if not line.strip():
                continue
            try:
                num, content = line.split('|', 1)
                parsed.append((int(num.strip()), content.strip()))
            except (ValueError, IndexError):
                continue
        return parsed

    def _extract_verify_paragraphs(self, text_block: str) -> list:
        """Extract paragraphs ending with 'Answer: Yes/No'."""
        if not text_block:
            return []
        paragraphs = re.findall(r'.*?Answer: (?:Yes|No)', text_block, re.DOTALL)
        return [p.strip() for p in paragraphs]

    def _get_answer(self, paragraph: str):
        m = re.search(r'Answer: (Yes|No)$', paragraph)
        return m.group(1) if m else None

    def check_all_answers_positive(self, paragraphs: list) -> float:
        answers = [self._get_answer(p) for p in paragraphs]
        valid = [1 if a and a.lower() == "yes" else 0 for a in answers if a is not None]
        return 1.0 if valid and all(v == 1 for v in valid) else 0.0

    def check_tuple_schema_ok(self, parsed_tuples: list) -> bool:
        if not parsed_tuples:
            return False
        return all(
            any(p.match(content) for p in _TUPLE_SCHEMA_PATTERNS)
            for _, content in parsed_tuples
        )

    def check_feedback_step_format(self, feedback_text: str) -> bool:
        """Valid if lines follow 'Step N:' (consecutive from 1) or text signals no correction needed."""
        if not feedback_text or not feedback_text.strip():
            return False
        normalized = feedback_text.strip().lower()
        if "no need" in normalized or "no correction" in normalized:
            return True
        lines = [l.strip() for l in feedback_text.split('\n') if l.strip()]
        step_re = re.compile(r'^Step\s*(\d+)\s*:', re.IGNORECASE)
        step_numbers = []
        for line in lines:
            m = step_re.match(line)
            if not m:
                return False
            step_numbers.append(int(m.group(1)))
        return step_numbers == list(range(1, len(step_numbers) + 1))

    def _normalize_content(self, nlp, text: str) -> str:
        if not text:
            return ""
        doc = nlp(text.lower())
        return "".join(token.lemma_.strip() for token in doc if not token.is_space)

    def _calculate_metrics(self, gt_tuples, pred_tuples, gt_verify_paragraphs, pred_verify_paragraphs):
        """Full metrics for offline evaluation."""
        metrics = {}
        gt_contents = {self._normalize_content(self.nlp, c) for _, c in gt_tuples}
        pred_contents = {self._normalize_content(self.nlp, c) for _, c in pred_tuples}
        correct_matches = len(gt_contents & pred_contents)
        total_gt = len(gt_tuples)

        metrics['part1_accuracy'] = correct_matches / total_gt if total_gt > 0 else 0.0

        correct_answers = 0
        pred_content_to_index = {c: i for i, (_, c) in enumerate(pred_tuples)}
        for gt_idx, (_, gt_content) in enumerate(gt_tuples):
            pred_idx = pred_content_to_index.get(gt_content)
            if pred_idx is not None and pred_idx < len(pred_verify_paragraphs) and gt_idx < len(gt_verify_paragraphs):
                gt_ans = self._get_answer(gt_verify_paragraphs[gt_idx])
                pred_ans = self._get_answer(pred_verify_paragraphs[pred_idx])
                if gt_ans and gt_ans == pred_ans:
                    correct_answers += 1

        metrics['part2_accuracy'] = correct_answers / total_gt if total_gt > 0 else 0.0
        metrics['part2_accuracy_only_matching'] = correct_answers / correct_matches if correct_matches > 0 else 0.0
        metrics['internal_consistency_ok'] = 1 if len(pred_tuples) == len(pred_verify_paragraphs) else 0
        metrics['part1_length_match_ok'] = 1 if len(gt_tuples) == len(pred_tuples) else 0

        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}

    def _calculate_metrics_for_reward(self, gt_tuples, pred_tuples, pred_verify_paragraphs):
        """F1 and consistency metrics for RL reward computation."""
        metrics = {}
        gt_contents = {self._normalize_content(self.nlp, c) for _, c in gt_tuples}
        pred_contents = {self._normalize_content(self.nlp, c) for _, c in pred_tuples}
        correct_matches = len(gt_contents & pred_contents)
        total_gt = len(gt_tuples)

        precision = correct_matches / len(pred_tuples) if pred_tuples else 0.0
        recall = correct_matches / total_gt if total_gt > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics['task2_part2_accuracy'] = f1
        metrics['task2_internal_consistency_ok'] = (
            1 if len(pred_tuples) == len(pred_verify_paragraphs) and len(pred_tuples) > 0 else 0
        )

        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}


def _find_subsequence(haystack: list, needle: list) -> int:
    """Return start index of first occurrence of needle in haystack, or -1 if not found."""
    n, m = len(haystack), len(needle)
    if m == 0:
        return 0
    for i in range(n - m + 1):
        if haystack[i:i + m] == needle:
            return i
    return -1


def build_segment_response_mask(
    response_ids: torch.Tensor,
    tokenizer,
    second_pattern: str = FormattingEvaluatorV3.SECOND_PATTERN,
    third_pattern: str = FormattingEvaluatorV3.THIRD_PATTERN,
) -> torch.Tensor:
    """
    Build a per-token segment mask for V3 format responses.

    Returns [B, T] tensor with values:
        0 = padding (pad_token_id positions)
        2 = Decompose segment (tuple lines)
        3 = Verify segment
        4 = Feedback segment

    Falls back to all-2 (Decompose) when boundary markers are absent.
    """
    pad_id = tokenizer.pad_token_id
    second_toks = tokenizer.encode(second_pattern, add_special_tokens=False)
    third_toks = tokenizer.encode(third_pattern, add_special_tokens=False)

    B, T = response_ids.shape
    seg_mask = torch.full((B, T), 2, dtype=torch.long, device=response_ids.device)

    for b in range(B):
        ids = response_ids[b].tolist()
        second_pos = _find_subsequence(ids, second_toks)
        third_pos = _find_subsequence(ids, third_toks)

        for t in range(T):
            if ids[t] == pad_id:
                seg_mask[b, t] = 0
            elif second_pos >= 0 and t >= second_pos:
                seg_mask[b, t] = 4 if (third_pos >= 0 and t >= third_pos) else 3
            # else remains 2 (Decompose)

    return seg_mask


def filter_entity_questions(feedback_tuple: str, vqa_question: str) -> str:
    """Remove vqa questions whose corresponding feedback_tuple entry is entity-type.

    feedback_tuple and vqa_question are 1-to-1 by index.
    Finds indices in feedback_tuple where the type starts with 'entity -',
    removes those same indices from vqa_question, and re-indexes from 1.

    Line format: "<idx> | <content>"
    """
    # Collect entity indices from feedback_tuple
    entity_indices = set()
    for line in feedback_tuple.strip().split('\n'):
        parts = line.split(' | ', 1)
        if len(parts) == 2:
            idx_str, content = parts
            if content.strip().startswith('entity -'):
                try:
                    entity_indices.add(int(idx_str.strip()))
                except ValueError:
                    pass

    # Filter vqa_question by those indices
    reindexed = []
    new_idx = 1
    for line in vqa_question.strip().split('\n'):
        parts = line.split(' | ', 1)
        if len(parts) == 2:
            idx_str, content = parts
            try:
                idx = int(idx_str.strip())
            except ValueError:
                idx = None
            if idx in entity_indices:
                continue
        reindexed.append(f"{new_idx} | {parts[1] if len(parts) == 2 else line}")
        new_idx += 1

    result = '\n'.join(reindexed)
    if not result.strip():
        return vqa_question
    return result

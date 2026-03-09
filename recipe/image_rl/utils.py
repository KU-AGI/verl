import re
import torch
import spacy

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
        metrics['part2_accuracy'] = f1_score
        
        metrics['part2_internal_consistency_ok'] = 1 if (len(pred_part2) == len(pred_paragraphs)) and (len(pred_part2) != 0) and (len(pred_paragraphs) != 0) else 0

        final_metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
        
        return final_metrics

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

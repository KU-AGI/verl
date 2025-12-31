import re
import torch

class FormattingEvaluator:
    def __init__(self):
        self.SECOND_PATTERN = "Second, Verify that the decomposed elements align with the image."
        self.THIRD_PATTERN = "Third, Generate corrective feedback."

    # --- 헬퍼 메서드들 (변경 없음) ---
    def _split_text_into_parts(self, text):
        match_second = re.search(re.escape(self.SECOND_PATTERN), text)
        match_third = re.search(re.escape(self.THIRD_PATTERN), text)
        if match_second and match_third:
            part1 = text[:match_second.start()].strip()
            part2 = text[match_second.start():match_third.start()].strip().replace(self.SECOND_PATTERN, "").strip()
            part3 = text[match_third.start():].strip().replace(self.THIRD_PATTERN, "").strip()
            return part1, part2, part3
        return None, None, None

    def _parse_part1(self, text_block):
        if not text_block: return []
        lines = [line for line in text_block.split('\n') if line.strip()]
        parsed_lines = []
        for line in lines:
            try:
                num, contents = line.split('|', 1)
                parsed_lines.append((int(num.strip()), contents.strip()))
            except ValueError:
                continue
        return parsed_lines

    def _extract_answer_paragraphs(self, text_block):
        if not text_block: return []
        pattern = r'.*?Answer: (?:Yes|No)'
        paragraphs = re.findall(pattern, text_block, re.DOTALL)
        return [p.strip() for p in paragraphs]

    def _get_answer_from_paragraph(self, paragraph):
        match = re.search(r'Answer: (Yes|No)$', paragraph)
        return match.group(1) if match else None

    def _calculate_metrics(self, gt_part1, pred_part1, gt_answers, pred_paragraphs):
        """
        정확도(Accuracy)와 형식 점수(0/1)를 함께 계산합니다.
        """
        metrics = {}

        # 1. [Accuracy] Part 1 내용 정확도
        gt_contents = {content for _, content in gt_part1}
        pred_contents = {content for _, content in pred_part1}
        correct_matches = len(gt_contents.intersection(pred_contents))
        total_gt = len(gt_part1)
        metrics['part1_accuracy'] = correct_matches / total_gt if total_gt > 0 else 0.0

        # 2. [Accuracy] Part 2 답변 정확도 (분모: 전체 GT 개수)
        correct_answers = 0
        pred_content_to_index = {content: i for i, (_, content) in enumerate(pred_part1)}
        for gt_index, (_, gt_content) in enumerate(gt_part1):
            if gt_content in pred_content_to_index:
                pred_para_index = pred_content_to_index[gt_content]
                if pred_para_index < len(pred_paragraphs) and gt_index < len(gt_answers):
                    gt_ans = self._get_answer_from_paragraph(gt_answers[gt_index])
                    pred_ans = self._get_answer_from_paragraph(pred_paragraphs[pred_para_index])
                    if gt_ans and gt_ans == pred_ans:
                        correct_answers += 1
        metrics['part2_accuracy'] = correct_answers / total_gt if total_gt > 0 else 0.0
        metrics['part2_accuracy_only_matching'] = correct_answers / correct_matches if correct_matches > 0 else 0.0
        
        # 3. [0/1 Score] Pred Part 1과 Pred Part 2의 항목 개수 일치 여부 (내부 일관성)
        metrics['internal_consistency_ok'] = 1 if len(pred_part1) == len(pred_paragraphs) else 0
        
        # 4. [0/1 Score] GT Part 1과 Pred Part 1의 항목 개수 일치 여부 (생성 완전성)
        metrics['part1_length_match_ok'] = 1 if len(gt_part1) == len(pred_part1) else 0

        final_metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
        
        return final_metrics

    def _calculate_metrics_for_reward(self, gt_part1, pred_part1, pred_paragraphs):
        """
        정확도(Accuracy)와 형식 점수(0/1)를 함께 계산합니다.
        """
        metrics = {}
        
        # Part 1: F1 score
        gt_contents = {content for _, content in gt_part1}
        pred_contents = {content for _, content in pred_part1}
        correct_matches = len(gt_contents.intersection(pred_contents))
        total_gt = len(gt_part1)

        precision = correct_matches / len(pred_part1) if len(pred_part1) > 0 else 0.0
        recall = correct_matches / total_gt if total_gt > 0 else 0.0
        metrics['part1_accuracy'] = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Part 2: Pred Part 1과 Pred Part 2의 항목 개수 일치 여부 (내부 일관성)
        metrics['internal_consistency_ok'] = 1 if len(pred_part1) == len(pred_paragraphs) else 0

        final_metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
        
        return final_metrics

class FormattingEvaluatorV2:
    def __init__(self):
        # 4단계 구조를 위한 새로운 패턴 정의
        self.SECOND_PATTERN = "Second, Decompose summarize"
        self.THIRD_PATTERN = "Third, Verify that the decomposed elements align with the image."
        self.FOURTH_PATTERN = "Fourth, Generate corrective feedback."

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

    def _parse_part1(self, text_block):
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

    def _calculate_metrics(self, gt_part2, pred_part2, gt_answers, pred_paragraphs):
        """
        핵심 메트릭 계산 로직 (변경 없음)
        gt_part2/pred_part2: 파싱된 튜플 리스트
        gt_answers/pred_paragraphs: 질문/답변 문단 리스트
        """
        metrics = {}

        # 1. Part 1(Decompose) 내용 정확도
        gt_contents = {content for _, content in gt_part2}
        pred_contents = {content for _, content in pred_part2}
        correct_matches = len(gt_contents.intersection(pred_contents))
        total_gt = len(gt_part2)
        metrics['part1accuracy'] = correct_matches / total_gt if total_gt > 0 else 0.0

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
        gt_contents = {content for _, content in gt_part2}
        pred_contents = {content for _, content in pred_part2}
        correct_matches = len(gt_contents.intersection(pred_contents))
        total_gt = len(gt_part2)

        precision = correct_matches / len(pred_part2) if len(pred_part2) > 0 else 0.0
        recall = correct_matches / total_gt if total_gt > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall)
        metrics['part1_accuracy'] = f1_score if (precision + recall) > 0 else 0.0
        
        # Part 2: Pred Part 2와 Pred Part 2의 항목 개수 일치 여부 (내부 일관성)
        metrics['internal_consistency_ok'] = 1 if len(pred_part2) == len(pred_paragraphs) else 0

        final_metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
        
        return final_metrics

from typing import List, Tuple, Optional, Union
import torch

def find_pattern_positions(
    seq: List[int],
    pattern: List[int],
) -> List[int]:
    """seq 안에서 pattern 이 시작하는 인덱스들을 모두 반환."""
    positions = []
    n, m = len(seq), len(pattern)
    if m == 0 or n < m:
        return positions
    for i in range(n - m + 1):
        if seq[i : i + m] == pattern:
            positions.append(i)
    return positions


def find_first_pattern_after(
    seq: List[int],
    pattern: Optional[List[int]],
    start_idx: int,
) -> Optional[int]:
    """seq[start_idx:] 구간에서 pattern 이 처음으로 등장하는 시작 인덱스 (seq 기준)를 반환."""
    if pattern is None:
        return None
    n, m = len(seq), len(pattern)
    if m == 0 or start_idx > n - m:
        return None
    for i in range(start_idx, n - m + 1):
        if seq[i : i + m] == pattern:
            return i
    return None


def find_after_step(
    input_ids: Union[List[int], torch.LongTensor, torch.Tensor],
    step_idx: int,
) -> Tuple[Optional[str], Optional[int]]:
    """
    특정 step (step_idx)의 '끝' 이후에 나오는 토큰 중:

    - step_idx가 1~6일 때:
        * <REFLECTION> 이 먼저 나오면: ("reflection", 그 시작 인덱스)
        * 다음 step (step_idx + 1)의 헤더가 먼저 나오면: ("next_step", 그 시작 인덱스)

    - step_idx가 7일 때:
        * <REFLECTION> 이 먼저 나오면: ("reflection", 그 시작 인덱스)
        * 아니면 <ANSWER> 가 나오면: ("answer", 그 시작 인덱스)

    둘 다 없으면: (None, None)

    step_idx 는 1~7 사이 정수라고 가정.
    """
    if isinstance(input_ids, torch.Tensor):
        seq = input_ids.tolist()
    else:
        seq = list(input_ids)

    # Step 헤더 및 <REFLECTION>, <ANSWER> 토큰 정의
    step_patterns = {
        1: [565, 14822, 220, 16],
        2: [565, 14822, 220, 17],
        3: [565, 14822, 220, 18],
        4: [565, 14822, 220, 19],
        5: [565, 14822, 220, 20],
        6: [565, 14822, 220, 21],
        7: [565, 14822, 220, 22],
    }
    reflection_pattern = [27, 5996, 28017]   # <REFLECTION>
    answer_pattern = [27, 11692, 39351]      # <ANSWER>

    # 0) 유효한 step_idx 체크
    if step_idx not in step_patterns:
        return None, None

    # 1) 기준이 되는 step 헤더의 위치 찾기 (첫 번째 등장 기준)
    current_step_pattern = step_patterns[step_idx]
    current_positions = find_pattern_positions(seq, current_step_pattern)
    if not current_positions:
        # 해당 step 헤더가 없으면 찾을 수 없음
        return None, None

    current_start = current_positions[0]
    search_start = current_start + len(current_step_pattern)

    # 2) 비교 대상 패턴 설정
    #    1~6: next_step 패턴
    #    7:   answer 패턴
    if step_idx < 7:
        other_pattern = step_patterns.get(step_idx + 1, None)
        other_kind = "next_step"
    else:  # step_idx == 7
        other_pattern = answer_pattern
        other_kind = "answer"

    # 3) search_start 이후에서 <REFLECTION> 과 other 패턴 둘 다의 위치를 찾고
    #    더 앞에 나오는 쪽을 선택
    reflection_pos = find_first_pattern_after(seq, reflection_pattern, search_start)
    other_pos = find_first_pattern_after(seq, other_pattern, search_start)

    # 둘 다 None이면 아무것도 없음
    if reflection_pos is None and other_pos is None:
        return None, None

    # 케이스별로 먼저 나오는 것 선택
    if reflection_pos is not None and other_pos is not None:
        if reflection_pos <= other_pos:
            return "reflection", reflection_pos
        else:
            return other_kind, other_pos
    elif reflection_pos is not None:
        return "reflection", reflection_pos
    else:
        return other_kind, other_pos

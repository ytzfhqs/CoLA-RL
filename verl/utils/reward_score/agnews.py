import re
from typing import List, Dict, Tuple, Optional


def extract_solution(solution_str: str) -> Tuple[Optional[str], Optional[str]]:
    content = None
    reasoning_content = None
    if m := re.match(r"<think>\n(.+)</think>\n\n", solution_str, flags=re.DOTALL):
        content = solution_str[len(m.group(0)):].strip()
        if thinking_content := m.group(1).strip():
            reasoning_content = thinking_content
    # if (content is None) or (reasoning_content is None):
        # print("[Error] 思维链与答案解析出错")
    return content, reasoning_content

def extract_content(answer_str: str) -> Tuple[Optional[str]]:
    """从答案中提取选项"""
    pass

def compute_score(solution_str: str, ground_truth: str, algorithm: str = 'grpo'):
    """计算总得分"""
    # print("\n" + "="*80)
    # print(" 开始新的采样 ".center(80, '='))
    # 从模型输出中分离答案和思考过程
    answer_text, processed_str = extract_solution(solution_str)
    # 成功解析
    if answer_text and processed_str:
        # print("\n[正确性验证]")
        # print(f"  真实标签: {ground_truth}")
        # print(f"  预测标签: {answer_text}")
        # print(f"\n[模型思考过程为]\n{processed_str}")
        # 检验答案是否正确
        if answer_text == ground_truth:
            total_score = 1
        else:
            total_score = -1
    else:
        total_score = -1
    # print(f" 最终得分{total_score} ".center(80, '-'))

    if algorithm == 'dapo':
        acc = 1 if total_score > 0 else 0
        return {
            "score": total_score,
            "acc": acc,
            "pred": answer_text,
        }
    else:
        return total_score
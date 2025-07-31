from typing import List, Dict, Any

def compute_accuracy(predicted_answers: List[str], ground_truth_answers: List[str]) -> Dict[str, float]:
    """
    Compute accuracy based on individual questions (4 questions per sample).
    
    Args:
        predicted_answers: List of predicted answer strings (comma-separated tags)
        ground_truth_answers: List of ground truth answer strings (comma-separated tags)
    
    Returns:
        Dictionary with overall accuracy and per-question accuracy
    """
    total_questions = 0
    correct_questions = 0
    per_question_stats = {f"question_{i+1}": {"correct": 0, "total": 0} for i in range(4)}
    
    assert len(predicted_answers) == len(ground_truth_answers), "Predicted and ground truth lists must have same length"
    
    for pred_str, gt_str in zip(predicted_answers, ground_truth_answers):
        # Parse predicted tags
        pred_tags = [tag.strip() for tag in pred_str.split(",")]
        
        # Parse ground truth tags
        gt_tags = [tag.strip() for tag in gt_str.split(",")]
        
        # Handle case where predicted has more than 4 tags - take first 4
        if len(pred_tags) > 4:
            pred_tags = pred_tags[:4]
        
        # Handle case where predicted has less than 4 tags - pad with empty string
        elif len(pred_tags) < 4:
            pred_tags.extend([""] * (4 - len(pred_tags)))
        
        # Ensure ground truth has exactly 4 tags (should always be true for XBRL data)
        if len(gt_tags) != 4:
            print(f"Warning: Ground truth has {len(gt_tags)} tags instead of 4: {gt_str}")
            # Pad or truncate ground truth if needed
            if len(gt_tags) > 4:
                gt_tags = gt_tags[:4]
            elif len(gt_tags) < 4:
                gt_tags.extend([""] * (4 - len(gt_tags)))
        
        # Compare each of the 4 questions
        for i in range(4):
            total_questions += 1
            per_question_stats[f"question_{i+1}"]["total"] += 1
            
            # Case-insensitive comparison
            if pred_tags[i].lower() == gt_tags[i].lower():
                correct_questions += 1
                per_question_stats[f"question_{i+1}"]["correct"] += 1
    
    # Calculate overall accuracy
    overall_accuracy = correct_questions / total_questions if total_questions > 0 else 0.0
    
    return {
        "overall_accuracy": overall_accuracy,
        "correct_questions": correct_questions,
        "total_questions": total_questions,
        "sample_count": len(predicted_answers)
    }


def compute_accuracy_single_sample(predicted: str, ground_truth: str) -> float:
    """
    Compute accuracy for a single sample (4 questions).
    
    Args:
        predicted: Comma-separated predicted tags
        ground_truth: Comma-separated ground truth tags
    
    Returns:
        Float accuracy (0.0 to 1.0) for this sample
    """
    # Parse predicted tags
    pred_tags = [tag.strip() for tag in predicted.split(",")]
    
    # Parse ground truth tags  
    gt_tags = [tag.strip() for tag in ground_truth.split(",")]
    
    # Handle case where predicted has more than 4 tags - take first 4
    if len(pred_tags) > 4:
        pred_tags = pred_tags[:4]
    
    # Handle case where predicted has less than 4 tags - pad with empty string
    elif len(pred_tags) < 4:
        pred_tags.extend([""] * (4 - len(pred_tags)))
    
    # Ensure ground truth has exactly 4 tags
    if len(gt_tags) != 4:
        # Pad or truncate ground truth if needed
        if len(gt_tags) > 4:
            gt_tags = gt_tags[:4]
        elif len(gt_tags) < 4:
            gt_tags.extend([""] * (4 - len(gt_tags)))
    
    # Count correct predictions
    correct = sum(1 for pred, gt in zip(pred_tags, gt_tags) 
                  if pred.lower() == gt.lower())
    
    return correct / 4.0


# Updated metric function for DSPy optimization
def individual_question_accuracy_metric(example, pred, trace=None):
    """
    Updated metric function for DSPy optimization that computes individual question accuracy.
    """
    if not hasattr(pred, 'answer'):
        return 0.0
    
    predicted = pred.answer
    ground_truth = example.target
    
    return compute_accuracy_single_sample(predicted, ground_truth)

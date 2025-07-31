import dspy
import json
import ast
import argparse
import random
import time
from typing import List, Dict, Any
import re
from accuracy_utils import *
from utils_gpt_v2 import *

api_key="4b2be69c-ead0-473f-9767-138702451d5d"
# Configure DSPy with your API
def configure_dspy():
    """Configure DSPy with SambaNova API - supporting multiple models"""
    # Generator model (lighter model for faster inference)
    generator_lm = dspy.LM(
        model="sambanova/Meta-Llama-3.1-8B-Instruct",
        api_key=api_key,  # Add your API key here
        api_base="https://api.sambanova.ai/v1"
    )
    
    # Reflector and Curator models (can use more powerful models)
    reflector_lm = dspy.LM(
        model="sambanova/Llama-4-Maverick-17B-128E-Instruct",
        api_key=api_key,  # Add your API key here
        api_base="https://api.sambanova.ai/v1"
    )
    
    curator_lm = dspy.LM(
        model="sambanova/Llama-4-Maverick-17B-128E-Instruct",
        api_key=api_key,  # Add your API key here
        api_base="https://api.sambanova.ai/v1"
    )
    
    # Set default LM for DSPy (used by optimizer)
    dspy.configure(lm=generator_lm)
    
    return {
        'generator': generator_lm,
        'reflector': reflector_lm,
        'curator': curator_lm
    }

# Define DSPy Signatures
class GeneratorSignature(dspy.Signature):
    """Generate US GAAP tags for financial entities based on cheatsheet guidance."""
    cheatsheet: str = dspy.InputField(desc="Curated cheatsheet with reasoning strategies and disambiguation rules")
    reflection: str = dspy.InputField(desc="Previous reflection on mistakes (if any)")
    question: str = dspy.InputField(desc="The XBRL tagging question with 4 sub-questions")
    context: str = dspy.InputField(desc="Available US GAAP tags and financial context")
    answer: str = dspy.OutputField(desc="Comma-separated US GAAP tags in order, exactly 4 tags")

class ReflectorSignature(dspy.Signature):
    """Analyze mistakes and provide insights for improving US GAAP tagging accuracy."""
    question: str = dspy.InputField(desc="The original XBRL tagging question")
    reasoning: str = dspy.InputField(desc="Model's reasoning process that led to the mistake")
    predicted: str = dspy.InputField(desc="Predicted answer that was incorrect")
    ground_truth: str = dspy.InputField(desc="Correct ground truth answer")
    cheatsheet: str = dspy.InputField(desc="Current cheatsheet available to the model")
    reflection: str = dspy.OutputField(desc="Diagnostic analysis of the error with improvement suggestions")

class CuratorSignature(dspy.Signature):
    """Update the cheatsheet based on reflection to improve future performance."""
    old_cheatsheet: str = dspy.InputField(desc="Previous version of the cheatsheet")
    reflection: str = dspy.InputField(desc="Analysis of recent mistakes and insights")
    question: str = dspy.InputField(desc="The question that triggered this update")
    correct_answer: str = dspy.InputField(desc="The correct answer for reference")
    new_cheatsheet: str = dspy.OutputField(desc="Updated cheatsheet with new rules and strategies")

# Define DSPy Modules
class Generator(dspy.Module):
    """Main generator module for US GAAP tagging"""
    def __init__(self, lm=None):
        super().__init__()
        if lm:
            # Use specific language model for this module
            with dspy.context(lm=lm):
                self.predict = dspy.ChainOfThought(GeneratorSignature)
        else:
            self.predict = dspy.ChainOfThought(GeneratorSignature)
    
    def forward(self, cheatsheet, reflection, question, context):
        result = self.predict(
            cheatsheet=cheatsheet, 
            reflection=reflection, 
            question=question, 
            context=context
        )
        return result

class Reflector(dspy.Module):
    """Reflection module for error analysis"""
    def __init__(self, lm=None):
        super().__init__()
        if lm:
            with dspy.context(lm=lm):
                self.predict = dspy.ChainOfThought(ReflectorSignature)
        else:
            self.predict = dspy.ChainOfThought(ReflectorSignature)
    
    def forward(self, question, reasoning, predicted, ground_truth, cheatsheet):
        result = self.predict(
            question=question,
            reasoning=reasoning,
            predicted=predicted,
            ground_truth=ground_truth,
            cheatsheet=cheatsheet
        )
        return result

class Curator(dspy.Module):
    """Curator module for cheatsheet updates"""
    def __init__(self, lm=None):
        super().__init__()
        if lm:
            with dspy.context(lm=lm):
                self.predict = dspy.ChainOfThought(CuratorSignature)
        else:
            self.predict = dspy.ChainOfThought(CuratorSignature)
    
    def forward(self, old_cheatsheet, reflection, question, correct_answer):
        result = self.predict(
            old_cheatsheet=old_cheatsheet,
            reflection=reflection,
            question=question,
            correct_answer=correct_answer
        )
        return result

# Main Pipeline
class XBRLAgenticPipeline(dspy.Module):
    """Complete agentic pipeline for XBRL classification with iterative improvement"""
    
    def __init__(self, max_rounds=3, models=None):
        super().__init__()
        if models:
            self.generator = Generator(models['generator'])
            self.reflector = Reflector(models['reflector'])
            self.curator = Curator(models['curator'])
        else:
            self.generator = Generator()
            self.reflector = Reflector()
            self.curator = Curator()
        self.max_rounds = max_rounds
        self.cheatsheet = "(empty)"
    
    def forward(self, question, context, ground_truth=None):
        """
        Forward pass that can handle both training and inference
        During training: uses ground_truth for reflection and improvement
        During inference: only generates answer
        """
        reflection = "(empty)"
        
        # Initial generation
        gen_result = self.generator(
            cheatsheet=self.cheatsheet,
            reflection=reflection,
            question=question,
            context=context
        )
        
        # Extract answer from generation
        predicted_answer = self.extract_answer(gen_result.answer)
        
        # If we have ground truth (training mode), potentially reflect and improve
        if ground_truth is not None:
            if not self.check_answer_quality(predicted_answer, ground_truth):
                # Iterate with reflection
                for round_num in range(self.max_rounds):
                    # Reflect on the mistake
                    reflection_result = self.reflector(
                        question=question,
                        reasoning=gen_result.answer,  # Full reasoning chain
                        predicted=predicted_answer,
                        ground_truth=ground_truth,
                        cheatsheet=self.cheatsheet
                    )
                    
                    reflection = reflection_result.reflection
                    
                    # Generate again with reflection
                    gen_result = self.generator(
                        cheatsheet=self.cheatsheet,
                        reflection=reflection,
                        question=question,
                        context=context
                    )
                    
                    predicted_answer = self.extract_answer(gen_result.answer)
                    
                    # Check if we've improved
                    if self.check_answer_quality(predicted_answer, ground_truth):
                        break
                
                # Update cheatsheet based on final reflection
                curator_result = self.curator(
                    old_cheatsheet=self.cheatsheet,
                    reflection=reflection,
                    question=question,
                    correct_answer=ground_truth
                )
                
                self.cheatsheet = self.extract_cheatsheet(curator_result.new_cheatsheet)
        
        return dspy.Prediction(answer=predicted_answer, cheatsheet=self.cheatsheet)
    
    def extract_answer(self, response: str) -> str:
        """Extract final answer from model response"""
        response = response.strip()
        # Look for Finish[...] pattern
        matches = re.findall(r"Finish\[(.*?)\]", response)
        if matches:
            return matches[-1].strip()
        
        # Fallback: look for comma-separated tags at the end
        lines = response.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if ',' in line and len(line.split(',')) == 4:
                return line
        
        return "No answer found"
    
    def extract_cheatsheet(self, response: str) -> str:
        """Extract cheatsheet from curator response"""
        response = response.strip()
        if "<cheatsheet>" in response:
            try:
                txt = response.split("<cheatsheet>")[1].strip()
                txt = txt.split("</cheatsheet>")[0].strip()
                return txt
            except:
                return self.cheatsheet
        return response if response else self.cheatsheet
    
    def check_answer_quality(self, predicted: str, ground_truth: str) -> bool:
        """Check if prediction meets quality threshold (relaxed check)"""
        pred_tags = [tag.lower().strip() for tag in predicted.split(",")]
        gt_tags = [tag.lower().strip() for tag in ground_truth.split(",")]
        
        correct_count = sum(1 for pred_tag in pred_tags if pred_tag in gt_tags)
        score = correct_count / len(pred_tags) if pred_tags else 0
        return score > 0.5

# Evaluation Metric
def xbrl_accuracy_metric(example, pred, trace=None):
    """Metric function for DSPy optimization"""
    if not hasattr(pred, 'answer'):
        return 0.0
    
    predicted = pred.answer
    ground_truth = example.target
    
    # Relaxed accuracy check
    pred_tags = [tag.lower().strip() for tag in predicted.split(",")]
    gt_tags = [tag.lower().strip() for tag in ground_truth.split(",")]
    
    if len(pred_tags) != 4 or len(gt_tags) != 4:
        return 0.0
    
    correct_count = sum(1 for pred_tag in pred_tags if pred_tag in gt_tags)
    score = correct_count / 4.0
    return score

# Data Processing Functions
def load_and_prepare_data(dataset_path: str, num_samples: int = -1):
    """Load and prepare training data for DSPy"""
    with open(dataset_path, 'r') as json_file:
        all_samples = list(json_file)
    
    if num_samples != -1:
        random.seed(42)
        random.shuffle(all_samples)
        all_samples = all_samples[:num_samples]
    
    examples = []
    for sample in all_samples:
        task_dict = ast.literal_eval(sample)
        context = task_dict["context"]
        
        # Parse context and question for XBRL format
        index = context.index("Answer the following 4 independent questions by providing only")
        available_tags, question = context[:index], context[index:]
        target = task_dict["target"]
        
        # Create DSPy Example
        example = dspy.Example(
            question=question,
            context=available_tags,
            target=target
        ).with_inputs("question", "context")
        
        examples.append(example)
    
    return examples

# Training and Optimization Functions
def train_pipeline(examples: List[dspy.Example], save_path: str, models: Dict[str, Any]):
    """Train and optimize the DSPy pipeline"""
    # Split data
    train_size = int(0.7 * len(examples))
    val_size = int(0.2 * len(examples))
    
    trainset = examples[:train_size]
    valset = examples[train_size:train_size + val_size]
    testset = examples[train_size + val_size:]

    print(f"Training on {len(trainset)} examples, validating on {len(valset)}")
    
    # Create pipeline with specific models
    pipeline = XBRLAgenticPipeline(max_rounds=3, models=models)
    
    predicted = []
    ground_truth = []
    for example in testset:
        result = pipeline(
            question=example.question,
            context=example.context,
            ground_truth=None  # Inference mode
        )
        predicted.append(result.answer)
        ground_truth.append(example.target)
        
    test_result = compute_accuracy(predicted, ground_truth)
    print(f"Initial test accuracy:\n{test_result}")
    breakpoint()

    # For training, we need to modify the pipeline to handle the training format
    class TrainingPipeline(dspy.Module):
        def __init__(self, base_pipeline):
            super().__init__()
            self.base_pipeline = base_pipeline
        
        def forward(self, question, context):
            # During optimization, only use generator module
            result = self.base_pipeline.generator(
                cheatsheet=self.base_pipeline.cheatsheet,
                reflection="(empty)",
                question=question,
                context=context
            )
            return dspy.Prediction(answer=self.base_pipeline.extract_answer(result.answer))
    
    training_pipeline = TrainingPipeline(pipeline)
    
    # Configure optimizer
    optimizer = dspy.MIPROv2(
        metric=xbrl_accuracy_metric,
        init_temperature=1.0,
        auto="medium"
    )
    
    print("Starting optimization...")
    
    # Optimize the pipeline
    optimized_pipeline = optimizer.compile(
        training_pipeline,
        trainset=trainset,
        valset=valset,
        requires_permission_to_run=False
    )
    
    # Evaluate optimized pipeline
    print("Evaluating optimized pipeline...")
    
    # Update the base pipeline with optimized modules
    pipeline.generator = optimized_pipeline.base_pipeline.generator
    
    # Now run the full agentic pipeline on training data to build cheatsheet
    print("Building cheatsheet through agentic training...")
    for i, example in enumerate(trainset):
        if i % 10 == 0:
            print(f"Processing example {i}/{len(trainset)}")
        
        # Run full pipeline with ground truth for cheatsheet building
        result = pipeline(
            question=example.question,
            context=example.context,
            ground_truth=example.target
        )
        
        # Save cheatsheet periodically
        if (i + 1) % 50 == 0:
            with open(f"{save_path}/cheatsheet_step_{i+1}.txt", "w") as f:
                f.write(pipeline.cheatsheet)
    
    # Final evaluation
    predicted = []
    ground_truth = []
    for example in testset:
        result = pipeline(
            question=example.question,
            context=example.context,
            ground_truth=None  # Inference mode
        )
        predicted.append(result.answer)
        ground_truth.append(example.target)
        
    test_result = compute_accuracy(predicted, ground_truth)
    print(f"Final test accuracy:\n{test_result}")
    
    # Save final pipeline and cheatsheet
    optimized_pipeline.save(f"{save_path}/optimized_pipeline.json")
    with open(f"{save_path}/final_cheatsheet.txt", "w") as f:
        f.write(pipeline.cheatsheet)
    
    return pipeline, accuracy

def main():
    parser = argparse.ArgumentParser(description='DSPy-based XBRL Classification System')
    parser.add_argument("--dataset_path", type=str, default="/import/ml-sc-scratch2/shubhangiu/jays_dc_repo/dynamic-cheatsheet/data/finlora/train/finer_train_batched.jsonl")
    parser.add_argument("--num_samples", default=100, type=int)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()
    
    # Configure DSPy and get model configurations
    models = configure_dspy()
    
    # Load data
    print("Loading and preparing data...")
    examples = load_and_prepare_data(args.dataset_path, args.num_samples)
    print(f"Loaded {len(examples)} examples")
    
    # Train pipeline with model configurations
    pipeline, accuracy = train_pipeline(examples, args.save_path, models)
    
    print(f"Training completed! Final accuracy: {accuracy:.3f}")
    print(f"Results saved to {args.save_path}")
    print("Model configuration:")
    print(f"  Generator: Meta-Llama-3.1-8B-Instruct")
    print(f"  Reflector: Llama-4-Maverick-17B-128E-Instruct") 
    print(f"  Curator: Llama-4-Maverick-17B-128E-Instruct")

if __name__ == "__main__":
    main()
import dspy
import json
import ast
import argparse
import random
import time
from typing import List, Dict, Any
import re
from accuracy_utils import *

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils_gpt_v2 import *

api_key="4b2be69c-ead0-473f-9767-138702451d5d"
api_key2="1db5b2f6-be77-46a5-8f04-5798c635126e"
api_key3="9a3a296e-2b71-4fad-bafa-9d19e445a92b"

# Configure DSPy with your API
def configure_dspy(model_name):
    """Configure DSPy with SambaNova API"""
    lm = dspy.LM(
        model=model_name,
        api_key=api_key,
        api_base="https://api.sambanova.ai/v1",
        max_tokens=100000
    )
    
    # Set default LM for DSPy (used by optimizer)
    dspy.configure(lm=lm)
    
    return lm

def create_generator_lm(model_name):
    """Create a separate LM instance for the generator"""
    generator_lm = dspy.LM(
        model=model_name,
        api_key=api_key2,
        api_base="https://api.sambanova.ai/v1",
        max_tokens=4096,
    )
    return generator_lm

# Define DSPy Signature for the baseline generator
class BaselineGeneratorSignature(dspy.Signature):
    """Generate US GAAP tags for financial entities."""
    question: str = dspy.InputField(desc="The XBRL tagging question with 4 sub-questions")
    context: str = dspy.InputField(desc="Available US GAAP tags and financial context")
    answer: str = dspy.OutputField(desc="Comma-separated US GAAP tags in order, exactly 4 tags")

# Define DSPy Module for baseline
class BaselineGenerator(dspy.Module):
    """Baseline generator module for US GAAP tagging"""
    def __init__(self, generator_lm=None):
        super().__init__()
        if generator_lm:
            # Use specific generator LM
            with dspy.context(lm=generator_lm):
                self.predict = dspy.ChainOfThought(BaselineGeneratorSignature)
            self.generator_lm = generator_lm
        else:
            # Use default configured LM
            self.predict = dspy.ChainOfThought(BaselineGeneratorSignature)
            self.generator_lm = None
    
    def forward(self, question, context):
        if self.generator_lm:
            # Use the specific generator LM
            with dspy.context(lm=self.generator_lm):
                result = self.predict(question=question, context=context, provide_traceback=True)
        else:
            # Use default LM
            result = self.predict(question=question, context=context, provide_traceback=True)
        return result

# Main Baseline Pipeline
class XBRLBaselinePipeline(dspy.Module):
    """Simple baseline pipeline for XBRL classification using only a generator"""
    
    def __init__(self, generator_lm=None):
        super().__init__()
        self.generator = BaselineGenerator(generator_lm=generator_lm)
        self.generator_lm = generator_lm
    
    def forward(self, question, context):
        """Forward pass that generates answer directly"""
        gen_result = self.generator(question=question, context=context)
        
        # Extract answer from generation
        predicted_answer = self.extract_answer(gen_result.answer)
        
        return dspy.Prediction(answer=predicted_answer)
    
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
        
        # Another fallback: look for any line with 4 comma-separated items
        for line in lines:
            line = line.strip()
            if line and ',' in line:
                tags = [tag.strip() for tag in line.split(',')]
                if len(tags) == 4:
                    return line
        
        return "No answer found"

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
def train_baseline_pipeline(examples: List[dspy.Example], testset: List[dspy.Example], generator_lm, args):
    """Train and optimize the baseline DSPy pipeline"""
    # Split training data into train and validation (no test since we have separate test data)
    # train_size = int(0.9 * len(examples))  # Use 90% for training, 10% for validation
    # trainset = examples[:train_size]
    # valset = examples[train_size:]
    assert args.train_size <= 900
    trainset = examples[:args.train_size]
    valset = examples[900:]

    print(f"Training on {len(trainset)} examples, validating on {len(valset)}, testing on {len(testset)}")
    
    # Create baseline pipeline with generator-specific LM
    pipeline = XBRLBaselinePipeline(generator_lm=generator_lm)
    
    print("Model configuration:")
    if generator_lm:
        print(f"  Default DSPy LM: {dspy.settings.lm.model}")
        print(f"  Generator LM: {generator_lm.model}")
        print("  Using separate models for optimization and generation")
    else:
        print(f"  Using single LM for both: {dspy.settings.lm.model}")
    
    # Initial evaluation on test set
    print("Evaluating initial pipeline...")
    predicted = []
    ground_truth = []
    for i, example in enumerate(testset):
        result = pipeline(question=example.question, context=example.context)
        predicted.append(result.answer)
        ground_truth.append(example.target)

    initial_accuracy = compute_accuracy(predicted, ground_truth)
    print(f"Initial test accuracy:\n{initial_accuracy}")

    # Configure optimizer
    optimizer = dspy.MIPROv2(
        metric=individual_question_accuracy_metric,
        init_temperature=1.0,
        auto=args.auto,
        max_bootstrapped_demos=args.max_bootstrapped_demos,
        metric_threshold=args.metric_threshold,
        log_dir=args.save_path
    )
    
    print("Starting optimization...")
    
    # Create a training wrapper that preserves the generator LM during optimization
    class TrainingPipeline(dspy.Module):
        def __init__(self, base_pipeline):
            super().__init__()
            self.base_pipeline = base_pipeline
        
        def forward(self, question, context):
            # During optimization, ensure we use the generator LM
            if self.base_pipeline.generator_lm:
                with dspy.context(lm=self.base_pipeline.generator_lm):
                    result = self.base_pipeline.generator.predict(question=question, context=context)
                    return dspy.Prediction(answer=self.base_pipeline.extract_answer(result.answer))
            else:
                result = self.base_pipeline.generator.predict(question=question, context=context)
                return dspy.Prediction(answer=self.base_pipeline.extract_answer(result.answer))
    
    training_pipeline = TrainingPipeline(pipeline)
    
    # Optimize the pipeline
    optimized_pipeline = optimizer.compile(
        training_pipeline,
        trainset=trainset,
        valset=valset,
        requires_permission_to_run=False
    )
    
    # Final evaluation on test set
    print("Evaluating optimized pipeline...")
    predicted = []
    ground_truth = []
    
    # Update the original pipeline with optimized components but preserve generator LM
    pipeline.generator = optimized_pipeline.base_pipeline.generator
    
    for example in testset:
        result = pipeline(question=example.question, context=example.context)
        predicted.append(result.answer)
        ground_truth.append(example.target)
        
    final_accuracy = compute_accuracy(predicted, ground_truth)
    print(f"Final test accuracy:\n{final_accuracy}")
    
    # Save optimized pipeline
    optimized_pipeline.save(f"{save_path}/baseline_optimized_pipeline.json")
    
    return pipeline, final_accuracy

def main():
    parser = argparse.ArgumentParser(description='DSPy Baseline XBRL Classification System')
    parser.add_argument("--dataset_path", type=str, default="/import/ml-sc-scratch2/shubhangiu/jays_dc_repo/dynamic-cheatsheet/data/finlora/train/finer_train_batched.jsonl")
    parser.add_argument("--test_dataset_path", type=str, default="/import/ml-sc-scratch5/fengluh/longicl/dynamic-cheatsheet/dspy/data/finer_test_subset_006_seed42.jsonl")
    parser.add_argument("--test_num_samples", default=-1, type=int, help="Number of test samples to use")
    parser.add_argument("--generator_model", type=str, default="sambanova/Meta-Llama-3.3-70B-Instruct", help="Model to use for the generator")
    parser.add_argument("--optimizer_model", type=str, default="sambanova/Llama-4-Maverick-17B-128E-Instruct", help="Model to use for DSPy optimization")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--train_size", default=500, type=int, help="Number of training samples to use")
    parser.add_argument("--max_bootstrapped_demos", type=int, default=4)
    parser.add_argument("--metric_threshold", type=float, default=1)
    parser.add_argument("--auto", type=str, default="medium")
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    
    # Configure DSPy with optimizer model
    print(f"Configuring DSPy with optimizer model: {args.optimizer_model}")
    default_lm = configure_dspy(args.optimizer_model)
    
    # Create generator LM if different from optimizer model
    generator_lm = None
    if args.generator_model != args.optimizer_model:
        print(f"Creating separate generator LM: {args.generator_model}")
        generator_lm = create_generator_lm(args.generator_model)
    else:
        print("Using same model for both optimization and generation")
    
    # Load training data
    print("Loading and preparing training data...")
    train_examples = load_and_prepare_data(args.dataset_path, 1000)
    print(f"Loaded {len(train_examples)} training examples")
    
    # Load test data
    print("Loading and preparing test data...")
    test_examples = load_and_prepare_data(args.test_dataset_path, args.test_num_samples)
    print(f"Loaded {len(test_examples)} test examples")
    
    # Train baseline pipeline
    pipeline, accuracy = train_baseline_pipeline(train_examples, test_examples, generator_lm, args)
    
    print(f"Training completed! Final accuracy: {accuracy}")
    print(f"Results saved to {args.save_path}")
    print("\nFinal configuration:")
    print(f"  Training samples: {len(train_examples)}")
    print(f"  Test samples: {len(test_examples)}")
    print(f"  Optimizer model: {args.optimizer_model}")
    print(f"  Generator model: {args.generator_model}")

if __name__ == "__main__":
    main()
import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple
import anthropic
from collections import defaultdict

class CheatsheetGenerator:
    def __init__(self, base_dir: str, anthropic_api_key: str):
        self.base_dir = Path(base_dir)
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.errors_data = []
        
    def find_all_files(self) -> Tuple[Dict[int, str], Dict[int, str]]:
        """Find all generator and reflector files"""
        generator_files = {}
        reflector_files = {}
        
        # Find generator files
        generator_pattern = str(self.base_dir / "detailed_llm_logs" / "generator_train_gen_initial_*.json")
        for filepath in glob.glob(generator_pattern):
            filename = os.path.basename(filepath)
            # Extract ID from filename: generator_train_gen_initial_199_20250725_171319_153.json
            parts = filename.split('_')
            if len(parts) >= 5:
                try:
                    question_id = int(parts[4])
                    generator_files[question_id] = filepath
                except ValueError:
                    continue
                    
        # Find reflector files (only round_0)
        reflector_pattern = str(self.base_dir / "detailed_llm_logs" / "reflector_train_reflect_*_round_0_*.json")
        for filepath in glob.glob(reflector_pattern):
            filename = os.path.basename(filepath)
            # Extract ID from filename: reflector_train_reflect_200_round_0_20250725_171427_294.json
            parts = filename.split('_')
            if len(parts) >= 6:
                try:
                    question_id = int(parts[3])
                    reflector_files[question_id] = filepath
                except ValueError:
                    continue
                    
        return generator_files, reflector_files
    
    def load_json_file(self, filepath: str) -> dict:
        """Load JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return {}
    
    def extract_error_info(self, generator_data: dict, reflector_data: dict) -> dict:
        """Extract key error information from generator and reflector data"""
        error_info = {
            'question': '',
            'predicted_answer': '',
            'ground_truth': '',
            'model_reasoning': '',
            'error_identification': '',
            'root_cause': '',
            'correct_approach': '',
            'key_insight': ''
        }
        
        # From generator data
        if 'prompt' in generator_data:
            # Extract question from prompt
            prompt = generator_data['prompt']
            if '**Question:**' in prompt:
                question_start = prompt.find('**Question:**') + len('**Question:**')
                question_end = prompt.find('**Context', question_start)
                if question_end == -1:
                    question_end = prompt.find('Output US GAAP tags:', question_start)
                error_info['question'] = prompt[question_start:question_end].strip()
        
        if 'response' in generator_data:
            response = generator_data['response']
            # Extract reasoning
            if '**Reasoning:**' in response:
                reasoning_start = response.find('**Reasoning:**') + len('**Reasoning:**')
                reasoning_end = response.find('**Final Answer:**', reasoning_start)
                if reasoning_end != -1:
                    error_info['model_reasoning'] = response[reasoning_start:reasoning_end].strip()
            
            # Extract predicted answer
            if 'Finish[' in response:
                answer_start = response.find('Finish[') + len('Finish[')
                answer_end = response.find(']', answer_start)
                if answer_end != -1:
                    error_info['predicted_answer'] = response[answer_start:answer_end].strip()
        
        # From reflector data
        if 'prompt' in reflector_data:
            prompt = reflector_data['prompt']
            # Extract ground truth
            if '**Ground Truth Answer:**' in prompt:
                gt_start = prompt.find('**Ground Truth Answer:**') + len('**Ground Truth Answer:**')
                gt_end = prompt.find('\n', gt_start)
                if gt_end != -1:
                    error_info['ground_truth'] = prompt[gt_start:gt_end].strip()
        
        if 'response' in reflector_data:
            response = reflector_data['response']
            # Extract error analysis
            if '**Error Identification:**' in response:
                ei_start = response.find('**Error Identification:**') + len('**Error Identification:**')
                ei_end = response.find('**Root Cause Analysis:**', ei_start)
                if ei_end != -1:
                    error_info['error_identification'] = response[ei_start:ei_end].strip()
            
            if '**Root Cause Analysis:**' in response:
                rc_start = response.find('**Root Cause Analysis:**') + len('**Root Cause Analysis:**')
                rc_end = response.find('**Correct Approach:**', rc_start)
                if rc_end != -1:
                    error_info['root_cause'] = response[rc_start:rc_end].strip()
            
            if '**Correct Approach:**' in response:
                ca_start = response.find('**Correct Approach:**') + len('**Correct Approach:**')
                ca_end = response.find('**Key Insight:**', ca_start)
                if ca_end != -1:
                    error_info['correct_approach'] = response[ca_start:ca_end].strip()
            
            if '**Key Insight:**' in response:
                ki_start = response.find('**Key Insight:**') + len('**Key Insight:**')
                error_info['key_insight'] = response[ki_start:].strip()
        
        return error_info
    
    def process_all_errors(self):
        """Process all error files and extract information"""
        generator_files, reflector_files = self.find_all_files()
        
        print(f"Found {len(generator_files)} generator files")
        print(f"Found {len(reflector_files)} reflector files (errors)")
        
        # Process each error
        for question_id in reflector_files:
            if question_id in generator_files:
                print(f"Processing error for question {question_id}")
                
                generator_data = self.load_json_file(generator_files[question_id])
                reflector_data = self.load_json_file(reflector_files[question_id])
                
                error_info = self.extract_error_info(generator_data, reflector_data)
                error_info['question_id'] = question_id
                
                self.errors_data.append(error_info)
        
        print(f"Processed {len(self.errors_data)} errors")
        
    def categorize_errors(self) -> Dict[str, List[dict]]:
        """Categorize errors by type"""
        categories = defaultdict(list)
        
        for error in self.errors_data:
            # Simple categorization based on root cause
            root_cause = error['root_cause'].lower()
            
            if 'tax' in root_cause and ('rate' in root_cause or 'reconciliation' in root_cause):
                categories['Tax Rate Confusion'].append(error)
            elif 'ownership' in root_cause or 'minority interest' in root_cause:
                categories['Ownership Percentage Issues'].append(error)
            elif 'credit' in root_cause or 'borrowing' in root_cause:
                categories['Credit Facility Classification'].append(error)
            elif 'revenue' in root_cause:
                categories['Revenue Recognition'].append(error)
            elif 'debt' in root_cause:
                categories['Debt Classification'].append(error)
            else:
                categories['Other'].append(error)
        
        return dict(categories)
    
    def generate_cheatsheet_with_claude(self) -> str:
        """Use Claude to generate an optimized cheatsheet"""
        
        # Prepare summary of errors
        error_categories = self.categorize_errors()
        
        summary = "# Error Analysis Summary\n\n"
        summary += f"Total errors analyzed: {len(self.errors_data)}\n\n"
        
        for category, errors in error_categories.items():
            summary += f"\n## {category} ({len(errors)} errors)\n"
            
            # Show up to 3 examples per category
            for i, error in enumerate(errors[:3]):
                summary += f"\n### Example {i+1}:\n"
                summary += f"- Question: {error['question'][:200]}...\n"
                summary += f"- Predicted: {error['predicted_answer']}\n"
                summary += f"- Correct: {error['ground_truth']}\n"
                summary += f"- Root Cause: {error['root_cause'][:200]}...\n"
                summary += f"- Key Insight: {error['key_insight'][:200]}...\n"
        
        # Add all key insights
        summary += "\n\n# All Key Insights:\n"
        unique_insights = set()
        for error in self.errors_data:
            if error['key_insight']:
                unique_insights.add(error['key_insight'])
        
        for insight in unique_insights:
            summary += f"- {insight}\n"
        
        # Create prompt for Claude
        prompt = f"""Based on the following error analysis from a financial GAAP tagging model, create a comprehensive cheatsheet that will help the model avoid these mistakes in the future.

{summary}

Please create a structured cheatsheet with:

1. **Core Thinking Framework**: A step-by-step methodology for approaching GAAP tagging questions
2. **Critical Distinctions**: Key differences between commonly confused tags (with clear decision criteria)
3. **Red Flags to Check**: A checklist of things to verify before finalizing an answer
4. **Pattern Recognition Guide**: How to identify context clues that indicate specific tags
5. **Common Pitfalls**: Specific mistakes to avoid with examples
6. **Quick Reference Decision Trees**: For the most problematic areas (tax rates, ownership, credit facilities, etc.)

Make the cheatsheet actionable, specific, and focused on preventing the exact types of errors seen in the analysis. Use bullet points and clear formatting for easy reference during problem-solving.
"""
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                temperature=0,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text
        except Exception as e:
            print(f"Error calling Claude API: {e}")
            return ""
    
    def save_results(self, output_dir: str):
        """Save all results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save error analysis
        with open(output_path / "error_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(self.errors_data, f, indent=2, ensure_ascii=False)
        
        # Save categorized errors
        categories = self.categorize_errors()
        with open(output_path / "error_categories.json", 'w', encoding='utf-8') as f:
            json.dump(categories, f, indent=2, ensure_ascii=False)
        
        # Generate and save cheatsheet
        print("Generating optimized cheatsheet with Claude...")
        cheatsheet = self.generate_cheatsheet_with_claude()
        
        if cheatsheet:
            with open(output_path / "optimized_cheatsheet.md", 'w', encoding='utf-8') as f:
                f.write(cheatsheet)
            print(f"Cheatsheet saved to {output_path / 'optimized_cheatsheet.md'}")
        
        # Save summary statistics
        stats = {
            'total_errors': len(self.errors_data),
            'error_categories': {cat: len(errors) for cat, errors in categories.items()},
            'total_questions_processed': len(self.errors_data)
        }
        
        with open(output_path / "statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nAll results saved to {output_path}")
        print(f"- Error analysis: error_analysis.json")
        print(f"- Error categories: error_categories.json")
        print(f"- Optimized cheatsheet: optimized_cheatsheet.md")
        print(f"- Statistics: statistics.json")


def main():
    # Configuration
    BASE_DIR = "/import/snvm-sc-scratch2/changranh/rl/dynamic-cheatsheet/V3_V3_V3_bulletpoint_multi_epoch_sambanova/run_20250725_121735_together_gen_DeepSeek-V3_epochs_1_samples_500_batched"
    OUTPUT_DIR = "./cheatsheet_analysis_output"
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

    # Initialize and run
    generator = CheatsheetGenerator(BASE_DIR, ANTHROPIC_API_KEY)

    # Process all errors
    generator.process_all_errors()

    # Save results
    generator.save_results(OUTPUT_DIR)

if __name__ == "__main__":
    main()
    
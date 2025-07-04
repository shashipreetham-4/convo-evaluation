# run_benchmark.py
import pandas as pd
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import sys
import os

# --- Configuration ---
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct" # Llama 3 8B
MAX_NEW_TOKENS_PER_FACET_CATEGORY = 200 # Adjust based on expected JSON output size per category
SCORE_SCALE_MIN = 1
SCORE_SCALE_MAX = 5
OUTPUT_SCORES_FILE = 'conversation_scores.json'
PROCESSED_FACETS_FILE = 'processed_facets.csv'
CONVERSATIONS_FILE = 'conversations.json'

# --- Utility Functions ---

def generate_facet_prompt(conversation_turn_text, facets_in_category_df):
    """
    Generates a prompt for the LLM to score facets within a specific category.
    """
    prompt = f"Given the following conversation turn: '{conversation_turn_text}'\n\n"
    prompt += "Please evaluate this turn based on the following facets. "
    prompt += f"Score each facet from {SCORE_SCALE_MIN} (very poor) to {SCORE_SCALE_MAX} (excellent). "
    prompt += "Provide your response as a JSON object with facet names as keys and integer scores as values.\n\n"

    for _, row in facets_in_category_df.iterrows():
        prompt += f"- {row['facet_name']}:\n"
        if pd.notna(row['description']):
            prompt += f"  Definition: {row['description']}\n"
        if pd.notna(row['example_good']):
            prompt += f"  Example Good: '{row['example_good']}'\n"
        if pd.notna(row['example_bad']):
            prompt += f"  Example Bad: '{row['example_bad']}'\n"
        prompt += f"  Score: [{SCORE_SCALE_MIN}-{SCORE_SCALE_MAX}]\n" # Placeholder for LLM to fill

    prompt += "\nOutput JSON:\n"
    return prompt

def parse_llm_output(output_text, expected_facets):
    """
    Parses the JSON output from the LLM and validates scores.
    Returns a dictionary of scores or None if parsing fails.
    """
    try:
        # Find the JSON part in the LLM's output
        json_start = output_text.find('{')
        json_end = output_text.rfind('}') + 1
        
        if json_start == -1 or json_end == -1:
            print(f"Warning: JSON object not found in LLM output. Raw output: {output_text[:500]}...")
            return None

        json_str = output_text[json_start:json_end]
        scores = json.loads(json_str)

        # Basic validation: Check if scores are integers and within scale
        validated_scores = {}
        for facet_name, score in scores.items():
            if facet_name in expected_facets and isinstance(score, int) and SCORE_SCALE_MIN <= score <= SCORE_SCALE_MAX:
                validated_scores[facet_name] = score
            else:
                print(f"Warning: Invalid score for facet '{facet_name}'. Expected integer between {SCORE_SCALE_MIN}-{SCORE_SCALE_MAX}. Got: {score}")
                validated_scores[facet_name] = -1 # Indicate invalid score

        return validated_scores

    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error parsing JSON output: {e}")
        print(f"Problematic LLM output snippet: {output_text[max(0, json_start-100):min(len(output_text), json_end+100)]}...")
        return None

def initialize_llm():
    """
    Initializes and returns the LLM pipeline.
    Uses bfloat16 and device_map="auto" for efficient GPU utilization.
    """
    print(f"Initializing LLM: {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16, # Recommended for T4/A100 GPUs for memory efficiency
            device_map="auto"           # Automatically distributes model across available GPUs/CPU
        )
        llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print("LLM initialized successfully.")
        return llm_pipeline
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        print("Ensure you have sufficient GPU memory and installed necessary libraries (e.g., bitsandbytes).")
        sys.exit(1) # Exit if LLM cannot be initialized

def main():
    # 1. Load Data
    if not os.path.exists(PROCESSED_FACETS_FILE):
        print(f"Error: '{PROCESSED_FACETS_FILE}' not found. Please run the data preparation step.")
        sys.exit(1)
    df_facets = pd.read_csv(PROCESSED_FACETS_FILE)
    print(f"Loaded {len(df_facets)} facets from {PROCESSED_FACETS_FILE}.")

    if not os.path.exists(CONVERSATIONS_FILE):
        print(f"Error: '{CONVERSATIONS_FILE}' not found. Please ensure your conversation data is present.")
        sys.exit(1)
    with open(CONVERSATIONS_FILE, 'r') as f:
        conversations = json.load(f)
    print(f"Loaded {len(conversations)} conversations from {CONVERSATIONS_FILE}.")

    # Pre-process facets by category for efficient lookup
    unique_categories = df_facets['category'].unique()
    facets_by_category = {
        category: df_facets[df_facets['category'] == category].copy()
        for category in unique_categories
    }

    # 2. Initialize LLM
    llm_pipeline = initialize_llm()

    # 3. Score Conversations
    all_conversation_turn_scores = []

    for conv in conversations:
        conv_id = conv['id']
        print(f"\n--- Processing Conversation: {conv_id} ---")
        for i, turn in enumerate(conv['turns']):
            turn_text = turn['text']
            turn_number = i + 1
            speaker = turn['speaker']
            
            print(f"  Scoring Turn {turn_number} ({speaker}): '{turn_text[:100]}...'")

            turn_facet_scores = {}
            
            for category in unique_categories:
                current_category_facets_df = facets_by_category[category]
                if current_category_facets_df.empty:
                    continue # Skip empty categories

                prompt = generate_facet_prompt(turn_text, current_category_facets_df)
                
                # LLM inference
                try:
                    outputs = llm_pipeline(
                        prompt,
                        max_new_tokens=MAX_NEW_TOKENS_PER_FACET_CATEGORY,
                        num_return_sequences=1,
                        do_sample=False, # For more deterministic scoring
                        pad_token_id=tokenizer.eos_token_id # Important for Llama 3 with pipeline
                    )
                    llm_response = outputs[0]['generated_text']
                    
                    # Llama 3 instruct models prepend the prompt to the output.
                    # We need to extract only the generated part.
                    if llm_response.startswith(prompt):
                        llm_response = llm_response[len(prompt):].strip()

                    expected_facets_in_category = current_category_facets_df['facet_name'].tolist()
                    category_scores = parse_llm_output(llm_response, expected_facets_in_category)
                    
                    if category_scores:
                        turn_facet_scores.update(category_scores)
                    else:
                        print(f"    Warning: Could not parse scores for category '{category}' for turn {turn_number}.")
                        # Assign -1 to all facets in this category if parsing failed for it
                        for facet_name in expected_facets_in_category:
                            if facet_name not in turn_facet_scores: # Avoid overwriting if it was already scored
                                turn_facet_scores[facet_name] = -1

                except Exception as e:
                    print(f"    Error during LLM inference for category '{category}': {e}")
                    # Assign -1 to all facets in this category if LLM call failed
                    for facet_name in current_category_facets_df['facet_name'].tolist():
                        if facet_name not in turn_facet_scores:
                            turn_facet_scores[facet_name] = -1

            # Ensure all facets have a score, even if some categories failed
            for _, facet_row in df_facets.iterrows():
                if facet_row['facet_name'] not in turn_facet_scores:
                    turn_facet_scores[facet_row['facet_name']] = -1 # Mark as unscored

            # Store results for this turn
            turn_result = {
                "conversation_id": conv_id,
                "turn_number": turn_number,
                "speaker": speaker,
                "turn_text": turn_text,
                "scores": turn_facet_scores
            }
            all_conversation_turn_scores.append(turn_result)

    # 4. Save Results
    with open(OUTPUT_SCORES_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_conversation_turn_scores, f, indent=4, ensure_ascii=False)
    
    print(f"\nBenchmark completed! Results saved to '{OUTPUT_SCORES_FILE}'.")
    print(f"Total turns scored: {len(all_conversation_turn_scores)}")

if __name__ == "__main__":
    main()

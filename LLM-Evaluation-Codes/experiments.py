import pandas as pd
import llm_evaluation

def load_texts_from_csv(csv_path, label_required=False):
    df = pd.read_csv(csv_path)
    if label_required:
        assert {'text', 'label'}.issubset(df.columns)
        return list(zip(df['text'].tolist(), df['label'].tolist()))
    else:
        assert 'text' in df.columns
        return df['text'].tolist()
    
def run_llm_alignment_from_csv(sample_csv, example_pool_csv=None, prompt_type="ZS"):
    # Load sample texts
    samples = load_texts_from_csv(sample_csv)

    # Load FS examples if required
    example_pool = None
    if prompt_type.upper() == "FS":
        example_pool = load_texts_from_csv(example_pool_csv, label_required=True)

    # Run classification and evaluation
    outputs = classify_with_all_llms(samples, prompt_type=prompt_type, example_pool=example_pool)
    
    from your_previous_module import llm_consensus_evaluation
    D, D_tilde, best_llm_index = llm_consensus_evaluation(outputs)

    print("Raw Deviation Scores:", D)
    print("Normalized Scores:", D_tilde)
    print("Most aligned LLM:", LLM_NAMES[best_llm_index])
    return D, D_tilde, best_llm_index

if __name__ == "__main__":
    sample_csv = "samples.csv"
    example_pool_csv = "example_pool.csv"

    run_llm_alignment_from_csv(sample_csv, example_pool_csv, prompt_type="FS")

    # samples.csv — Classification Inputs
    """
    column: text
    The food was amazing!
    I really disliked the service.
    It was okay, just not memorable.
    Totally loved the ambiance and staff.
    Worst experience I've ever had.
    """

    # example_pool.csv — FS Demonstration Pool
    """
    columns: text,label
    Absolutely fantastic meal! , 5
    I hated every minute. , 1
    Mediocre and bland. , 3
    Friendly staff and good food. , 4
    Horrible service ruined the night. , 2
    """

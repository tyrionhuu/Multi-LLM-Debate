if __name__ == "__main__":
    from datasets import load_dataset

    LLMBar = load_dataset("princeton-nlp/LLMBar", "LLMBar")
    CaseStudy = load_dataset("princeton-nlp/LLMBar", "CaseStudy")
    print(LLMBar)
    print(CaseStudy)

from transformers import pipeline

model_path = "fine_tuned_model"


def main():
    seq2seq_pipeline = pipeline(
        "text2text-generation", model=model_path, tokenizer=model_path)
    input_text = "你好"
    result = seq2seq_pipeline(input_text, max_length=160)
    print("Predicted Output:", result[0]['generated_text'])


if __name__ == "__main__":
    main()

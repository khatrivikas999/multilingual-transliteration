import json
import ctranslate2
from transformers import AutoTokenizer
from evaluate import load as load_metric
from tqdm import tqdm


def evaluate_model():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained('models/transliteration')

    # Use CPU to avoid cublas DLL error
    translator = ctranslate2.Translator(
        'models/transliteration_ct2',
        device='cpu',
        compute_type='int8',
        inter_threads=4
    )

    print("Loading test data...")
    test_data = []
    with open('data/test.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))

    # Limit to 3000 samples for fast evaluation
    test_data = test_data[:3000]

    print(f"Evaluating on {len(test_data)} samples...")
    predictions = []
    references = []

    for item in tqdm(test_data):
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(item['source']))
        result = translator.translate_batch([tokens], beam_size=4)
        output_tokens = result[0].hypotheses[0]
        pred = tokenizer.decode(
            tokenizer.convert_tokens_to_ids(output_tokens),
            skip_special_tokens=True
        )
        predictions.append(pred)
        references.append(item['target'])

    exact_matches = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
    accuracy = exact_matches / len(predictions)

    cer_metric = load_metric("cer")
    wer_metric = load_metric("wer")

    cer = cer_metric.compute(predictions=predictions, references=references)
    wer = wer_metric.compute(predictions=predictions, references=references)

    results = {
        'accuracy': accuracy,
        'cer': cer,
        'wer': wer,
        'total_samples': len(predictions)
    }

    print(f"\nResults:")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"CER:      {cer:.4f}")
    print(f"WER:      {wer:.4f}")

    print("\nSample predictions:")
    for i in range(min(10, len(predictions))):
        match = "✓" if predictions[i].strip() == references[i].strip() else "✗"
        print(f"{match} {test_data[i]['source']} -> {predictions[i]} (ref: {references[i]})")

    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nSaved to evaluation_results.json")


if __name__ == '__main__':
    evaluate_model()
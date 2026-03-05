import os
import time
import json
import torch
import ctranslate2
from transformers import AutoTokenizer, MT5ForConditionalGeneration

def convert_model():
    print("Converting to CTranslate2...")
    converter = ctranslate2.converters.TransformersConverter('models/transliteration')
    converter.convert(output_dir='models/transliteration_ct2', quantization='int8', force=True)
    print("Conversion complete")

def get_size(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total / (1024 * 1024)

def benchmark():
    test_inputs = [
        '<hi> namaste',
        '<hi> dhanyavaad',
        '<bn> nomoskar',
        '<bn> dhonnobad',
        '<ta> vanakkam',
        '<ta> nandri'
    ]
    
    tokenizer = AutoTokenizer.from_pretrained('models/transliteration')
    
    # PyTorch benchmark
    print("\nBenchmarking PyTorch model...")
    model = MT5ForConditionalGeneration.from_pretrained('models/transliteration')
    model.eval()
    
    with torch.no_grad():
        for text in test_inputs[:3]:
            inputs = tokenizer(text, return_tensors='pt')
            _ = model.generate(**inputs, max_length=128)
    
    start = time.time()
    num_runs = 100
    with torch.no_grad():
        for _ in range(num_runs):
            for text in test_inputs:
                inputs = tokenizer(text, return_tensors='pt')
                _ = model.generate(**inputs, max_length=128)
    pytorch_time = time.time() - start
    pytorch_latency = (pytorch_time / (num_runs * len(test_inputs))) * 1000
    
    # CTranslate2 benchmark
    print("Benchmarking CTranslate2 model...")
    translator = ctranslate2.Translator('models/transliteration_ct2', device='cpu', compute_type='int8')
    
    for text in test_inputs[:3]:
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
        _ = translator.translate_batch([tokens])
    
    start = time.time()
    for _ in range(num_runs):
        for text in test_inputs:
            tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
            _ = translator.translate_batch([tokens])
    ct2_time = time.time() - start
    ct2_latency = (ct2_time / (num_runs * len(test_inputs))) * 1000
    
    # Results
    pytorch_size = get_size('models/transliteration')
    ct2_size = get_size('models/transliteration_ct2')
    
    results = {
        'pytorch': {
            'size_mb': pytorch_size,
            'latency_ms': pytorch_latency,
            'throughput': (num_runs * len(test_inputs)) / pytorch_time
        },
        'ctranslate2': {
            'size_mb': ct2_size,
            'latency_ms': ct2_latency,
            'throughput': (num_runs * len(test_inputs)) / ct2_time
        },
        'improvement': {
            'size_reduction': ((pytorch_size - ct2_size) / pytorch_size) * 100,
            'latency_reduction': ((pytorch_latency - ct2_latency) / pytorch_latency) * 100,
            'speedup': pytorch_latency / ct2_latency
        }
    }
    
    print("\nResults:")
    print(f"PyTorch: {pytorch_size:.0f} MB, {pytorch_latency:.1f} ms")
    print(f"CTranslate2: {ct2_size:.0f} MB, {ct2_latency:.1f} ms")
    print(f"Size reduction: {results['improvement']['size_reduction']:.1f}%")
    print(f"Speed improvement: {results['improvement']['speedup']:.1f}x")
    
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    convert_model()
    benchmark()
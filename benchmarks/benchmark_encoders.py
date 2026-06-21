"""
Benchmark script comparing custom encoders with sklearn's LabelEncoder
"""
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import time
from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder
from label_encoder_random.transformers.label_encoder_random import LabelEncoderRandom
from label_encoder_random.transformers.label_encoder_manual import LabelEncoderManual


def benchmark_transform(encoder_name, encoder, y_train, y_test, num_runs=5):
    """Benchmark transform operation"""
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        encoder.transform(y_test)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    return avg_time, std_time


def benchmark_inverse_transform(encoder_name, encoder, y_train, y_test, num_runs=5):
    """Benchmark inverse_transform operation"""
    # First encode the data
    y_encoded = encoder.transform(y_test)
    
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        encoder.inverse_transform(y_encoded)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    return avg_time, std_time


def run_benchmark_suite(n_samples_list, n_classes_list, dtypes):
    """Run comprehensive benchmark"""
    print("=" * 80)
    print("LABEL ENCODER BENCHMARK")
    print("=" * 80)
    
    for dtype in dtypes:
        print(f"\n{'='*80}")
        print(f"Data Type: {dtype}")
        print(f"{'='*80}")
        
        for n_classes in n_classes_list:
            for n_samples in n_samples_list:
                print(f"\nSamples: {n_samples:,}, Classes: {n_classes}")
                print("-" * 80)
                
                # Generate test data
                y_indices = np.random.choice(n_classes, size=n_samples)
                if dtype == "int":
                    y = y_indices
                elif dtype == "str":
                    y = np.array([f"class_{i}" for i in y_indices], dtype=object)
                elif dtype == "uint32":
                    y = y_indices.astype(np.uint32)
                else:
                    y = y_indices.astype(dtype)
                
                # Split into train/test
                split_idx = int(0.8 * n_samples)
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                results = {}
                
                # Benchmark sklearn
                try:
                    sk_encoder = SklearnLabelEncoder()
                    sk_encoder.fit(y_train)
                    
                    transform_time, transform_std = benchmark_transform(
                        "sklearn", sk_encoder, y_train, y_test
                    )
                    inverse_time, inverse_std = benchmark_inverse_transform(
                        "sklearn", sk_encoder, y_train, y_test
                    )
                    
                    results["sklearn"] = {
                        "transform": (transform_time, transform_std),
                        "inverse": (inverse_time, inverse_std)
                    }
                except Exception as e:
                    print(f"  sklearn: FAILED ({e})")
                
                # Benchmark LabelEncoderRandom
                try:
                    random_encoder = LabelEncoderRandom(randomize=False)
                    random_encoder.fit(y_train)
                    
                    transform_time, transform_std = benchmark_transform(
                        "LabelEncoderRandom", random_encoder, y_train, y_test
                    )
                    inverse_time, inverse_std = benchmark_inverse_transform(
                        "LabelEncoderRandom", random_encoder, y_train, y_test
                    )
                    
                    results["LabelEncoderRandom"] = {
                        "transform": (transform_time, transform_std),
                        "inverse": (inverse_time, inverse_std)
                    }
                except Exception as e:
                    print(f"  LabelEncoderRandom: FAILED ({e})")
                
                # Benchmark LabelEncoderManual
                try:
                    # Create a manual mapping
                    classes = np.unique(y_train)
                    mapping = {c: i for i, c in enumerate(classes)}
                    manual_encoder = LabelEncoderManual(mapping)
                    manual_encoder.fit(y_train)
                    
                    transform_time, transform_std = benchmark_transform(
                        "LabelEncoderManual", manual_encoder, y_train, y_test
                    )
                    inverse_time, inverse_std = benchmark_inverse_transform(
                        "LabelEncoderManual", manual_encoder, y_train, y_test
                    )
                    
                    results["LabelEncoderManual"] = {
                        "transform": (transform_time, transform_std),
                        "inverse": (inverse_time, inverse_std)
                    }
                except Exception as e:
                    print(f"  LabelEncoderManual: FAILED ({e})")
                
                # Print results
                print(f"\n  Transform (ms):")
                for encoder_name, metrics in results.items():
                    t_avg, t_std = metrics["transform"]
                    print(f"    {encoder_name:25s}: {t_avg:8.4f} ± {t_std:6.4f}")
                
                print(f"\n  Inverse Transform (ms):")
                for encoder_name, metrics in results.items():
                    i_avg, i_std = metrics["inverse"]
                    print(f"    {encoder_name:25s}: {i_avg:8.4f} ± {i_std:6.4f}")
                
                # Show speedup relative to sklearn
                if "sklearn" in results and "LabelEncoderRandom" in results:
                    sk_transform = results["sklearn"]["transform"][0]
                    random_transform = results["LabelEncoderRandom"]["transform"][0]
                    speedup_transform = sk_transform / random_transform
                    print(f"\n  Speedup vs sklearn (transform): {speedup_transform:.2f}x")
                    
                    sk_inverse = results["sklearn"]["inverse"][0]
                    random_inverse = results["LabelEncoderRandom"]["inverse"][0]
                    speedup_inverse = sk_inverse / random_inverse
                    print(f"  Speedup vs sklearn (inverse): {speedup_inverse:.2f}x")


if __name__ == "__main__":
    # Configuration
    n_samples_list = [1000, 10000, 100000]
    n_classes_list = [10, 100]
    dtypes = ["int", "str", "uint32"]
    
    run_benchmark_suite(n_samples_list, n_classes_list, dtypes)
    
    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)

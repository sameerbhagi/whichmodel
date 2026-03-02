import sys
import os

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openrouter

def check_coverage():
    print("Checking model data coverage (Global Scan)...")
    
    # fetch_models is synchronous and caches results
    models = openrouter.fetch_models()
    
    benchmarks = openrouter.CURATED_BENCHMARKS
    meta = openrouter.CURATED_META
    
    print(f"\nTotal Active Models (After Filtering): {len(models)}")
    
    missing_bench = []
    missing_meta = []
    incomplete_meta = []
    
    for m in models:
        # m.openrouter_id is the "provider/slug" version used in our dicts
        real_id = m.openrouter_id
        
        if not real_id:
            continue

        if real_id not in benchmarks:
            missing_bench.append(real_id)
        if real_id not in meta:
            missing_meta.append(real_id)
        else:
            data = meta[real_id]
            missing_fields = []
            if not data.get("release_date"): missing_fields.append("release_date")
            if not data.get("use_cases"): missing_fields.append("use_cases")
            if missing_fields:
                incomplete_meta.append(f"{real_id}: {', '.join(missing_fields)}")

    print(f"\nMissing Benchmarks ({len(missing_bench)}):")
    for m in sorted(missing_bench): print(f" - {m}")
        
    print(f"\nMissing Metadata ({len(missing_meta)}):")
    for m in sorted(missing_meta): print(f" - {m}")
        
    print(f"\nIncomplete Metadata ({len(incomplete_meta)}):")
    for m in sorted(incomplete_meta): print(f" - {m}")

if __name__ == "__main__":
    check_coverage()

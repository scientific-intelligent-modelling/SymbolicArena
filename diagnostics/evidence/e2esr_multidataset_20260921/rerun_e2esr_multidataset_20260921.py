"""Post-fix spot checks, predetermined rather than selected by held-out scores."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import run_e2esr_multidataset_20260921 as suite

suite.OUT = Path('/tmp/e2esr_multidataset_fixed_20260921')
suite.DATASETS = ['korns/Korns-1', 'llm-srbench/bio_pop_growth/BPG3']
suite.OUT.mkdir(exist_ok=False)
with ThreadPoolExecutor(max_workers=2) as executor:
    results = list(executor.map(suite.run_one, suite.DATASETS))
(suite.OUT / 'execution_summary.json').write_text(json.dumps(results, indent=2))

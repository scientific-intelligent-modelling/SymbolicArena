import os
import resource


limit = 2 * 1024**3
resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.semantic_validation_worker import main


if __name__ == '__main__':
    raise SystemExit(main())

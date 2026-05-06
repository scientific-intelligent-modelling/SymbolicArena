# srkit/subprocess_runner.py
import argparse
import json
import importlib
import sys
import traceback
import os
try:
    from .config_manager import config_manager
except ImportError:
    from config_manager import config_manager

try:
    from scientific_intelligent_modelling.srkit.exceptions import NoValidOutputError
except ImportError:
    try:
        from .exceptions import NoValidOutputError
    except ImportError:
        from exceptions import NoValidOutputError
    from exceptions import NoValidOutputError

# print(f"current note: {os.getcwd()}")
# print(f"note: {os.path.dirname(os.path.abspath(__file__))}")

def main():
    """note"""
    # note/note note
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    parser = argparse.ArgumentParser(description='symbolic regressionnote')
    parser.add_argument('--input', required=True, help='note')
    parser.add_argument('--output', required=True, help='note')
    args = parser.parse_args()
    
    try:
        # note
        with open(args.input, 'r') as f:
            command = json.load(f)
        
        # note
        tool_name = command.get('tool_name')
        if not tool_name:
            raise ValueError("note")
        
        # note
        wrapper_module = importlib.import_module(f"scientific_intelligent_modelling.algorithms.{tool_name}_wrapper.wrapper")
        
        # note
        result = execute_command(wrapper_module, command)
        
        # note
        with open(args.output, 'w') as f:
            json.dump(result, f)
            
    except NoValidOutputError as e:
        no_valid_result = {
            'success': False,
            'no_valid_output': True,
            'message': str(e),
            'traceback': traceback.format_exc(),
        }
        with open(args.output, 'w') as f:
            json.dump(no_valid_result, f)
    except Exception as e:
        # note
        error_result = {
            'error': True,
            'message': str(e),
            'traceback': traceback.format_exc()
        }
        with open(args.output, 'w') as f:
            json.dump(error_result, f)
        sys.exit(1)

def execute_command(module, command):
    """note"""
    action = command['action']
    tool_name = command['tool_name']
    
    # note
    regressor_class = get_regressor_class(module, tool_name)
    
    if action == 'fit':
        return handle_fit(regressor_class, command)
    elif action == 'recover_from_timeout':
        return handle_recover_from_timeout(regressor_class, command)
    elif action == 'predict':
        return handle_predict(regressor_class, command)
    elif action == 'get_optimal_equation':
        return handle_get_optimal_equation(regressor_class, command)
    elif action == 'get_total_equations':
        return handle_get_total_equations(regressor_class, command)
    elif action == 'get_fitted_params':
        return handle_get_fitted_params(regressor_class, command)
    elif action == 'get_total_equations_with_params':
        return handle_get_total_equations_with_params(regressor_class, command)
    elif action == 'export_canonical_symbolic_program':
        return handle_export_canonical_symbolic_program(regressor_class, command)
    else:
        raise ValueError(f"note: {action}")

def handle_get_optimal_equation(regressor_class, command):
    """noteget_optimal_equationnote note"""
    # note
    serialized_model = command['serialized_model']
    
    # note
    regressor = regressor_class()
    
    # notedeserializenote note
    if hasattr(regressor, 'deserialize'):
        regressor = regressor.deserialize(serialized_model)
    
    # note
    equation = None
    
    # note note
    if hasattr(regressor, 'get_optimal_equation'):
        equation = regressor.get_optimal_equation()
    elif hasattr(regressor, 'get_equation'):
        equation = regressor.get_equation()
    elif hasattr(regressor, 'get_model_string'):
        equation = regressor.get_model_string()
    elif hasattr(regressor, 'symbolic_model'):
        equation = str(regressor.symbolic_model)
    elif hasattr(regressor, 'best_'):
        equation = str(regressor.best_)
    elif hasattr(regressor, 'program_'):
        equation = str(regressor.program_)
    else:
        raise ValueError("note")
    
    return {
        'success': True,
        'equation': equation
    }

def handle_get_total_equations(regressor_class, command):
    """noteget_total_equationsnote note note n note"""
    # note
    serialized_model = command['serialized_model']
    
    # note
    regressor = regressor_class()
    
    # notedeserializenote note
    if hasattr(regressor, 'deserialize'):
        regressor = regressor.deserialize(serialized_model)
    
    # note
    equations = None
    
    # note n
    n = command.get('n', None)

    # note
    if hasattr(regressor, 'get_total_equations'):
        try:
            equations = regressor.get_total_equations(n) if n is not None else regressor.get_total_equations()
        except TypeError:
            equations = regressor.get_total_equations()
    elif hasattr(regressor, 'get_all_equations'):
        equations = regressor.get_all_equations()
    elif hasattr(regressor, 'hall_of_fame_'):
        equations = [str(program) for program in regressor.hall_of_fame_]
    elif hasattr(regressor, 'models_'):
        equations = [str(model) for model in regressor.models_]
    else:
        raise ValueError("note")
    
    return {
        'success': True,
        'equations': equations
    }

def handle_get_fitted_params(regressor_class, command):
    """note note  """
    serialized_model = command['serialized_model']
    regressor = regressor_class()
    if hasattr(regressor, 'deserialize'):
        regressor = regressor.deserialize(serialized_model)
    params = None
    if hasattr(regressor, 'get_fitted_params'):
        params = regressor.get_fitted_params()
    elif hasattr(regressor, '_best_params'):
        p = getattr(regressor, '_best_params')
        try:
            import numpy as _np
            params = _np.asarray(p).tolist() if p is not None else None
        except Exception:
            params = None
    else:
        raise ValueError("note get_fitted_params")
    return {
        'success': True,
        'params': params
    }

def handle_get_total_equations_with_params(regressor_class, command):
    """note noteTop-N note note  """
    serialized_model = command['serialized_model']
    n = command.get('n')
    regressor = regressor_class()
    if hasattr(regressor, 'deserialize'):
        regressor = regressor.deserialize(serialized_model)
    if hasattr(regressor, 'get_total_equations_with_params'):
        try:
            items = regressor.get_total_equations_with_params(n) if n is not None else regressor.get_total_equations_with_params()
        except TypeError:
            items = regressor.get_total_equations_with_params()
    else:
        raise ValueError("note get_total_equations_with_params")
    return {
        'success': True,
        'items': items
    }


def handle_export_canonical_symbolic_program(regressor_class, command):
    """note """
    from scientific_intelligent_modelling.benchmarks.artifact_schema import (
        validate_canonical_symbolic_program,
    )

    serialized_model = command['serialized_model']
    regressor = regressor_class()
    if hasattr(regressor, 'deserialize'):
        regressor = regressor.deserialize(serialized_model)

    if not hasattr(regressor, 'export_canonical_symbolic_program'):
        raise ValueError("note export_canonical_symbolic_program")

    artifact = regressor.export_canonical_symbolic_program()
    artifact = validate_canonical_symbolic_program(artifact)
    return {
        'success': True,
        'artifact': artifact,
    }

# note
def get_class_mapping_from_config():
    # notetoolboxnote
    toolbox_config = config_manager.get_config("toolbox_config")
    tool_mapping = toolbox_config.get("tool_mapping", {})
    
    # note
    class_mapping = {}
    for tool_name, tool_info in tool_mapping.items():
        class_mapping[tool_name] = tool_info.get("regressor")
    
    # note note
    if "srbench" not in class_mapping:
        class_mapping["srbench"] = "SRBenchRegressor"
        
    return class_mapping

def get_regressor_class(module, tool_name):
    """note"""
    # note
    class_mapping = get_class_mapping_from_config()
    
    class_name = class_mapping.get(tool_name)
    if class_name and hasattr(module, class_name):
        return getattr(module, class_name)
    
    # # note noteRegressornote
    # for attr_name in dir(module):
    #     if attr_name.endswith('Regressor'):
    #         return getattr(module, attr_name)
    
    raise ValueError(f"note {module.__name__} note")

def handle_fit(regressor_class, command):
    """notefitnote"""
    import numpy as np
    
    # note
    data = command['data']
    params = dict(command.get('params', {}) or {})
    # note note wrapper 
    params.pop('timeout_in_seconds', None)
    
    # notenumpynote
    X = np.array(data['X'])
    y = np.array(data['y'])
    
    # note
    regressor = None
    serialized_model_from_command = command.get('serialized_model')

    if serialized_model_from_command and hasattr(regressor_class, 'deserialize'):
        # note
        regressor = regressor_class.deserialize(serialized_model_from_command)
        # note note `params` note fit note note 
        # note note  
        # note `fit` note 
        # note note `fit` note 
    else:
        # note note note
        regressor = regressor_class(**params)
    regressor.fit(X, y)
    
    # note
    serialized_model = None
    if hasattr(regressor, 'serialize'):
        serialized_model = regressor.serialize()
    else:
        raise ValueError(f"note {regressor_class.__name__} noteserializenote")
    
    return {
        'success': True,
        'serialized_model': serialized_model
    }


def handle_recover_from_timeout(regressor_class, command):
    """note """
    import numpy as np

    data = command.get('data') or {}
    params = dict(command.get('params', {}) or {})
    X = np.array(data.get('X', []))
    y = np.array(data.get('y', []))
    if X.ndim == 1 and X.size > 0:
        X = X.reshape(-1, 1)
    y = y.reshape(-1)

    experiment_dir = command.get('experiment_dir')
    if isinstance(experiment_dir, str) and experiment_dir.strip():
        params.setdefault('existing_exp_dir', experiment_dir.strip())
        params.setdefault('exp_dir', experiment_dir.strip())

    regressor = regressor_class(**params)

    # note wrapper note fit note note note 
    if command.get('tool_name') == 'drsr':
        regressor.fit(X, y)

    equation = ""
    if hasattr(regressor, 'get_optimal_equation'):
        equation = str(regressor.get_optimal_equation() or "")
    if not equation.strip():
        raise ValueError("note note")

    if X.size > 0:
        sample_rows = min(8, X.shape[0])
        sample_X = X[:sample_rows]
        predictions = np.asarray(regressor.predict(sample_X)).reshape(-1)
        if predictions.shape != (sample_rows,):
            raise ValueError(
                f"note note note {(sample_rows,)}, note {predictions.shape}"
            )
        if not np.all(np.isfinite(predictions)):
            raise ValueError("note note")

    if hasattr(regressor, 'serialize'):
        serialized_model = regressor.serialize()
    else:
        raise ValueError(f"note {regressor_class.__name__} note serialize note")

    return {
        'success': True,
        'serialized_model': serialized_model,
        'equation': equation,
        'recovered_from_timeout': True,
    }

def handle_predict(regressor_class, command):
    """notepredictnote"""
    import numpy as np
    
    # note
    data = command['data']
    serialized_model = command['serialized_model']
    
    # notenumpynote
    X = np.array(data['X'])
    
    # note
    regressor = regressor_class()  # note
    
    # notedeserializenote note
    if hasattr(regressor, 'deserialize'):
        regressor = regressor.deserialize(serialized_model)
    
    # note
    predictions = regressor.predict(X)
    
    # note
    if isinstance(predictions, np.ndarray):
        predictions = predictions.tolist()
    
    return {
        'success': True,
        'predictions': predictions
    }

if __name__ == "__main__":
    main()

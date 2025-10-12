import os
import subprocess
import time 
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
import pandas as pd 

app = Flask(__name__)

# --- Configuration ---
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
VENV_PYTHON_EXEC = PROJECT_ROOT / '.venv' / 'Scripts' / 'python.exe'
UPLOAD_FOLDER = PROJECT_ROOT / 'uploads'
OUTPUT_FOLDER = PROJECT_ROOT / 'static' / 'results' 

# Ensure directories exist
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
# ---------------------

def _check_venv():
    """Helper to ensure VENV executable path exists."""
    if not VENV_PYTHON_EXEC.exists():
        return {'status': 'FAILURE', 'error': f"VENV Error: Cannot find Python executable at {VENV_PYTHON_EXEC}. Check VENV name/path."}
    return None

def run_comparison_script(filepath_a: str, filepath_b: str, output_id: str) -> dict:
    """Executes the symph-compare.py script."""
    
    venv_check = _check_venv()
    if venv_check: return venv_check
    
    run_output_dir = OUTPUT_FOLDER / output_id
    run_output_dir.mkdir(exist_ok=True)
    
    command = [
        str(VENV_PYTHON_EXEC), '-m', 'cli.symph-compare', 
        '--a', filepath_a,
        '--b', filepath_b,
        '--out', str(run_output_dir), 
        '--nodata', '0,-9999',
        '--sample', '500000'
    ]
    
    try:
        result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
        metrics_path = run_output_dir / 'metrics.csv'
        
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            stats = metrics_df.iloc[0].to_dict()
        else:
            stats = {'error': 'Metrics file not generated. Check stdout/stderr.'}

        return {
            'status': 'SUCCESS', 'output_id': output_id, 'metrics': stats,
            'stdout': result.stdout, 'stderr': result.stderr
        }

    except subprocess.CalledProcessError as e:
        return {
            'status': 'FAILURE', 'output_id': output_id,
            'error': f"Compare Script execution failed. Error: {e.stderr}",
            'stderr': e.stderr
        }
    except Exception as e:
        return {'status': 'FAILURE', 'error': f"General execution error: {e}"}


def run_prediction_script(target_file: str, predictor_files: list, model: str, output_id: str) -> dict:
    """
    Executes the symph-predict.py script.
    """
    
    venv_check = _check_venv()
    if venv_check: return venv_check
    
    run_output_dir = OUTPUT_FOLDER / output_id
    run_output_dir.mkdir(exist_ok=True)
    
    command = [
        str(VENV_PYTHON_EXEC), '-m', 'cli.symph-predict', 
        '--target', target_file,
        '--predictors'
    ]
    command.extend(predictor_files) 
    command.extend([
        '--out', str(run_output_dir), 
        '--model', model,
        '--transform_y', 'log1p', 
        '--sample', '200000'
    ])
    
    try:
        result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
        metrics_path = run_output_dir / 'metrics.csv'
        
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            stats = metrics_df.iloc[0].to_dict()
        else:
            stats = {'error': 'Metrics file not generated.'}

        return {
            'status': 'SUCCESS', 'output_id': output_id, 'metrics': stats,
            'stdout': result.stdout, 'stderr': result.stderr
        }

    except subprocess.CalledProcessError as e:
        return {
            'status': 'FAILURE', 'output_id': output_id,
            'error': f"Predict Script execution failed. Error: {e.stderr}",
            'stderr': e.stderr
        }
    except Exception as e:
        return {'status': 'FAILURE', 'error': f"General execution error: {e}"}


# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare')
def compare_page():
    return render_template('compare.html')

@app.route('/predict')
def predict_page():
    model_choices = ['rf', 'gbm', 'xgb', 'nn', 'auto']
    return render_template('predict.html', model_choices=model_choices)

@app.route('/run_compare', methods=['POST'])
def run_compare():
    file1 = request.files.get('user_file1')
    file2 = request.files.get('user_file2')

    if not file1 or not file2: return "Error: Both files must be selected.", 400
    if not (secure_filename(file1.filename) and secure_filename(file2.filename)): return "Error: Invalid file name detected.", 400

    unique_id = str(int(time.time()))
    temp_filename1 = f"{unique_id}_A_{secure_filename(file1.filename)}"
    temp_filename2 = f"{unique_id}_B_{secure_filename(file2.filename)}"
    
    filepath1 = str(UPLOAD_FOLDER / temp_filename1)
    filepath2 = str(UPLOAD_FOLDER / temp_filename2)

    try:
        file1.save(filepath1)
        file2.save(filepath2)
        
        result_data = run_comparison_script(filepath1, filepath2, unique_id)
        
        if result_data['status'] == 'SUCCESS':
            return redirect(url_for('results_page', run_id=unique_id))
        else:
            return render_template('error.html', data=result_data), 500

    finally:
        try: os.remove(filepath1); os.remove(filepath2)
        except Exception: pass 

@app.route('/run_predict', methods=['POST'])
def run_predict():
    target_file = request.files.get('target_file')
    predictor_files = request.files.getlist('predictor_files')
    model_choice = request.form.get('model_choice')

    if not target_file or not any(f.filename for f in predictor_files):
        return "Error: Must select a target file and at least one valid predictor file.", 400

    unique_id = str(int(time.time()))
    
    target_filename = secure_filename(target_file.filename)
    target_filepath = str(UPLOAD_FOLDER / f"{unique_id}_T_{target_filename}")
    target_file.save(target_filepath)
    
    
    predictor_filepaths = []
    
    all_temp_files = [target_filepath] 
    
    for i, file in enumerate(predictor_files):
        if file.filename:
            pred_filename = secure_filename(file.filename)
            pred_filepath = str(UPLOAD_FOLDER / f"{unique_id}_P{i}_{pred_filename}")
            file.save(pred_filepath)
            predictor_filepaths.append(pred_filepath)
            all_temp_files.append(pred_filepath)

    try:
        # 3. RUN THE PREDICTION SCRIPT
        result_data = run_prediction_script(target_filepath, predictor_filepaths, model_choice, unique_id)
        
        # 4. Handle result and redirect
        if result_data['status'] == 'SUCCESS':
            return redirect(url_for('results_page', run_id=unique_id))
        else:
            return render_template('error.html', data=result_data), 500

    finally:
        for p in all_temp_files:
            try: os.remove(p)
            except Exception: pass

@app.route('/results/<run_id>')
def results_page(run_id):
    metrics_path = OUTPUT_FOLDER / run_id / 'metrics.csv'
    
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        metrics = metrics_df.iloc[0].to_dict()
        
        is_prediction = 'best_model' in metrics
        best_model_name = metrics.get('best_model', '')
        # Determine if the best model is one that produces feature importance
        is_tree_model = best_model_name in ['RF', 'GBM', 'XGB'] 
        
        # Paths to static assets
        scatter_url = url_for('static', filename=f"results/{run_id}/parity.png") 
        residuals_url = url_for('static', filename=f"results/{run_id}/residuals.png") 
        importance_url = url_for('static', filename=f"results/{run_id}/feature_importance.png")
        
        return render_template('results.html', 
                               run_id=run_id, 
                               metrics=metrics, 
                               scatter_url=scatter_url, 
                               residuals_url=residuals_url,
                               importance_url=importance_url,
                               is_prediction=is_prediction,
                               is_tree_model=is_tree_model) 
    
    return render_template('error.html', data={'error': f"Results not found for run ID {run_id}."}), 404

if __name__ == '__main__':
    app.run(debug=True)
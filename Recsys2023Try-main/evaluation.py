import sys
import sklearn.metrics as metrics
import glob
import os
import math
import pandas as pd

ROW_ID_AKA     = "RowId"
ROW_ID_COLUMN  = "row_id"
CLICK_COLUMN   = 'is_clicked'
INSTALL_COLUMN = 'is_installed'

def validate_submitted_predictions(actual_records, predicted_records, actual_record_ids):    
    valid = True
    error = ""
    # Check for schema
    try:
        predicted_records.rename(columns={ROW_ID_AKA: ROW_ID_COLUMN}, inplace=True)
        if ROW_ID_AKA not in predicted_records.columns and ROW_ID_COLUMN not in predicted_records.columns:
            raise ValueError("RowId/row_id Header Missing")
        if INSTALL_COLUMN not in predicted_records.columns:
            raise ValueError("is_installed Header Missing")
        if CLICK_COLUMN not in predicted_records.columns:
            raise ValueError("is_clicked Header Missing")
        
        # Check for unique and all row_id present
        if predicted_records.shape[0] != actual_records.shape[0]:
            raise ValueError(f"Total entries should be {actual_records.shape[0]}, found:{predicted_records.shape[0]}")

        missing_row_id = len(actual_record_ids.difference(set(predicted_records.row_id)))
        if missing_row_id != 0:
            raise ValueError(f"Total unique entries should be {actual_records.shape[0]}, found:{actual_records.shape[0]-missing_row_id}")

        # Check for values
        if predicted_records.is_installed.min() < 0.0:
            raise ValueError(f"Min predicted value should be 0, found:{predicted_records.is_installed.min()}")
        
        if predicted_records.is_installed.max() > 1.0:
            raise ValueError(f"Max predicted value should be 1, found:{predicted_records.is_installed.max()}")
        
        if predicted_records.is_installed.isnull().values.sum() > 0:
            raise ValueError("One or more predictions are NaNs.")
    except Exception as e:
        valid = False
        error = str(e)
    return valid, error

def calculate_log_loss(merged_data):
    loss = math.inf
    try:
        loss = metrics.log_loss(merged_data.true_installed, merged_data.is_installed, labels=[0,1], eps=1e-7, normalize=True)
    except Exception as e:
        pass
    return loss

def evaluate_submission(fn, ground_truth, all_record_ids):
    valid = True
    error = ""
    score = math.inf
    if not os.path.exists(fn):
        valid = False
        error = "Internal error !"
        return valid, error, score
    try:
        submission = pd.read_csv(fn, header=0, sep="\t")
    except Exception as e:
        valid = False
        error = str(e.__cause__)
        return valid, error, score
    
    valid, error = validate_submitted_predictions(ground_truth, submission, all_record_ids)
    if not valid:
        return valid, error, score
        
    # Join on row_id
    try: 
        merged_data = pd.merge(left=ground_truth, right=submission, on=ROW_ID_COLUMN,how="left")
        score = calculate_log_loss(merged_data)
    except Exception as e:
        valid = False
        error = str(e.__cause__)
    return valid, error, score

def process_ground_truth(fn):
    valid          = True
    error          = ""
    ground_truth   = None
    all_record_ids = set()
    if not os.path.exists(fn):
        valid = False
        error = "Ground truth filename not found!"
        return valid, error, ground_truth, all_record_ids

    try:
        ground_truth = pd.read_csv(fn, header=0, sep="\t")
    except Exception as e:
        valid = False
        error = str(e.__cause__)
        return valid, error, ground_truth, all_record_ids
    
    all_record_ids = set(ground_truth.row_id)
    return valid, error, ground_truth, all_record_ids

if __name__ == '__main__':
    ground_truth_filename = None
    submitted_filenames = ['./Result/result9.txt']
    baseline_filename   = None
    if len(sys.argv) > 2:
        ground_truth_filename = sys.argv[1]
        submitted_filenames   = sys.argv[2]
    else:
        print(f"USAGE: {sys.argv[0]} ground-truth-filename submitted-filename(s)")
        sys.exit(-1)

    # process ground truth
    ground_truth = None
    all_record_ids = set([])
    baseline_score = None
    valid, error, ground_truth, all_record_ids = process_ground_truth(ground_truth_filename)
    if not valid:
        print("ERROR: loading ground truth data failed.")
        sys.exit(-1)

    # process baseline
    # baseline_score -- the actual score is hidden currently
    baseline_score = 1.0

    # evaluate the files
    for fn in glob.glob(submitted_filenames):
        submission_score = math.inf
        valid, error, submission_score = evaluate_submission(fn, ground_truth, all_record_ids)
        if valid:
            print(f"Filename: {fn}, Score: {submission_score/baseline_score}")
        else:
            print(f"Filename: {fn}, Error: {error}")

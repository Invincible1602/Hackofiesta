import warnings
warnings.filterwarnings("ignore", message="Glyph 9")

import re
import os
import json
from io import BytesIO
from typing import List, Dict

import requests
from textblob import TextBlob  # For spelling correction

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from dotenv import load_dotenv  # <-- Load environment variables
load_dotenv()  # <-- Call to load .env file values

# ----- Utility Function -----
def try_float(s):
    try:
        return float(s.replace(',', ''))
    except ValueError:
        return s

# ----- Text Extraction -----
def extract_text(image_bytes, filename, api_key):
    url = "https://api.ocr.space/parse/image"
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".png":
        mime_type = "image/png"
    elif ext in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    else:
        mime_type = "application/octet-stream"

    files = {"image": (filename, image_bytes, mime_type)}
    response = requests.post(
        url,
        files=files,
        data={
            "apikey": api_key,
            "language": "eng",
            "detectOrientation": "true",
            "isTable": "true",
            "scale": "true",
            "OCREngine": "2"
        }
    )
    response_json = response.json()
    if response_json.get("IsErroredOnProcessing"):
        error_msg = " ".join(response_json.get("ErrorMessage", []))
        raise HTTPException(status_code=400, detail=f"OCR processing error: {error_msg}")
    if "ParsedResults" in response_json and len(response_json["ParsedResults"]) > 0:
        return response_json["ParsedResults"][0]["ParsedText"], response_json
    else:
        raise HTTPException(status_code=400, detail="No parsed results found.")

# ----- Parsing Functions -----
def parse_freeform_report(text):
    tests = []
    total_lines = 0
    pattern = re.compile(r"""
        ^\s*
        (?P<test>.+?)(?:,\s*(?P<unit>[\w\/%]+))?\s+
        (?P<value>\d+(?:\.\d+)?)\s+
        (?P<ref>.+?)\s*$
        """, re.VERBOSE)
    
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        total_lines += 1
        match = pattern.match(line)
        if match:
            test_name = match.group('test').strip().lower()
            unit = match.group('unit').strip() if match.group('unit') else ""
            value_str = match.group('value').strip()
            ref = match.group('ref').strip()

            value_candidate = try_float(value_str)
            is_numeric = isinstance(value_candidate, float)

            ref_lower, ref_upper, ref_operator = None, None, None
            if is_numeric:
                range_match = re.search(r'([\d\.]+)\s*[-–]\s*([\d\.]+)', ref)
                if range_match:
                    ref_lower = try_float(range_match.group(1))
                    ref_upper = try_float(range_match.group(2))
                    ref_operator = 'range'
                else:
                    single_match = re.search(r'([<≤>≥])\s*([\d\.]+)', ref)
                    if single_match:
                        operator = single_match.group(1)
                        value = try_float(single_match.group(2))
                        if operator in ['<', '≤']:
                            ref_upper = value
                        elif operator in ['>', '≥']:
                            ref_lower = value
                        ref_operator = operator

            tests.append({
                'test': test_name,
                'value': value_candidate,
                'unit': unit,
                'ref_text': ref,
                'ref_lower': ref_lower,
                'ref_upper': ref_upper,
                'ref_operator': ref_operator,
                'is_numeric': is_numeric
            })
    return tests, total_lines

def parse_table_report(text):
    tests = []
    total_lines = 0
    for line in text.splitlines():
        line = line.strip()
        if not line or '|' not in line:
            continue
        total_lines += 1
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 4:
            continue
        test_name, unit, value_str, ref = parts[0].lower(), parts[1], parts[2], parts[3]
        value_candidate = try_float(value_str)
        is_numeric = isinstance(value_candidate, float)
        ref_lower, ref_upper, ref_operator = None, None, None
        if is_numeric:
            range_match = re.search(r'([\d\.]+)\s*[-–]\s*([\d\.]+)', ref)
            if range_match:
                ref_lower = try_float(range_match.group(1))
                ref_upper = try_float(range_match.group(2))
                ref_operator = 'range'
            else:
                single_match = re.search(r'([<≤>≥])\s*([\d\.]+)', ref)
                if single_match:
                    operator = single_match.group(1)
                    value = try_float(single_match.group(2))
                    if operator in ['<', '≤']:
                        ref_upper = value
                    elif operator in ['>', '≥']:
                        ref_lower = value
                    ref_operator = operator
        tests.append({
            'test': test_name,
            'value': value_candidate,
            'unit': unit,
            'ref_text': ref,
            'ref_lower': ref_lower,
            'ref_upper': ref_upper,
            'ref_operator': ref_operator,
            'is_numeric': is_numeric
        })
    return tests, total_lines

def parse_medical_report(text):
    tests, total_lines = parse_freeform_report(text)
    if not tests:
        tests_table, total_lines_table = parse_table_report(text)
        tests += tests_table
        total_lines += total_lines_table
    if not tests:
        raise HTTPException(status_code=400, detail="No tests could be parsed with available strategies.")
    return tests, total_lines

# ----- Evaluation Function -----
def evaluate_tests(tests):
    evaluated = []
    for test in tests:
        if test['is_numeric'] and test['ref_operator'] is not None:
            if test['ref_operator'] == 'range':
                if test['ref_lower'] <= test['value'] <= test['ref_upper']:
                    status = "Within normal range"
                elif test['value'] < test['ref_lower']:
                    status = "Lower than usual"
                else:
                    status = "Higher than usual"
            elif test['ref_operator'] in ['<', '≤']:
                if (test['ref_operator'] == '<' and test['value'] < test['ref_upper']) or \
                   (test['ref_operator'] == '≤' and test['value'] <= test['ref_upper']):
                    status = "Within normal range"
                else:
                    status = "Higher than usual"
            elif test['ref_operator'] in ['>', '≥']:
                if (test['ref_operator'] == '>' and test['value'] > test['ref_lower']) or \
                   (test['ref_operator'] == '≥' and test['value'] >= test['ref_lower']):
                    status = "Within normal range"
                else:
                    status = "Lower than usual"
        elif test['is_numeric']:
            status = "No reference range"
        else:
            status = "Non-numeric"
        evaluated.append({**test, 'status': status})
    return evaluated

# ----- Heuristic Accuracy Function -----
def calculate_heuristic_accuracy(tests, total_lines, ocr_response):
    if total_lines == 0:
        return 0.0
    parsing_success_rate = (len(tests) / total_lines) * 100 if tests else 0.0
    consistency_score = 0.0
    if tests:
        valid_tests = 0
        for test in tests:
            is_valid = True
            if test['is_numeric']:
                if not (0 <= test['value'] <= 10000):
                    is_valid = False
                if not test['unit']:
                    is_valid = False
                if test['ref_operator'] == 'range' and (test['ref_lower'] is None or test['ref_upper'] is None or test['ref_lower'] > test['ref_upper']):
                    is_valid = False
            if is_valid:
                valid_tests += 1
        consistency_score = (valid_tests / len(tests)) * 100
    accuracy = (parsing_success_rate * 0.5) + (consistency_score * 0.5)
    return min(accuracy, 100.0)

# ----- FastAPI Service Setup -----
app = FastAPI(title="Medical Report OCR & Analysis API")

# Default OCR API key loaded from environment variables.
DEFAULT_OCR_API_KEY = os.getenv("OCR_API")

def process_pipeline(image_file: UploadFile, api_key: str):
    image_bytes = image_file.file.read()
    filename = image_file.filename

    text, ocr_response = extract_text(image_bytes, filename, api_key)
    tests, total_lines = parse_medical_report(text)
    for test in tests:
        corrected_name = str(TextBlob(test['test']).correct())
        test['test'] = corrected_name
    evaluated_tests = evaluate_tests(tests)
    accuracy = calculate_heuristic_accuracy(evaluated_tests, total_lines, ocr_response)
    return {
        "filename": filename,
        "extracted_text": text,
        "evaluated_tests": evaluated_tests,
        "heuristic_accuracy": accuracy
    }

# ----- Modified Endpoint for Multiple Files -----
@app.post("/reports-json")
async def get_report_json(
    api_key: str = Form(default=DEFAULT_OCR_API_KEY),
    files: List[UploadFile] = File(...)
):
    if not api_key:
        raise HTTPException(status_code=400, detail="OCR API key is required (provide via form or set OCR_API_KEY environment variable).")
    reports = {}
    for file in files:
        try:
            report = process_pipeline(file, api_key)
            reports[file.filename] = report
        except HTTPException as e:
            reports[file.filename] = {"error": e.detail}
    return JSONResponse(content=reports)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

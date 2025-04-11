import warnings
# Filter out warnings mentioning missing glyph 9.
warnings.filterwarnings("ignore", message="Glyph 9")

import streamlit as st
import re
import matplotlib.pyplot as plt
import pandas as pd
import requests
import json
import os
import datetime
from io import BytesIO
from textblob import TextBlob  # Added for spelling correction

# Change default font to Arial to avoid glyph issues
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

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
        st.error("Error: " + " ".join(response_json.get("ErrorMessage", [])))
        return "", response_json
    if "ParsedResults" in response_json and len(response_json["ParsedResults"]) > 0:
        return response_json["ParsedResults"][0]["ParsedText"], response_json
    else:
        st.error("No parsed results found.")
        return "", response_json

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
            # Convert test name to lowercase here
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
        else:
            st.info("Line not parsed (freeform): " + line)
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
            st.info("Line not parsed (table): " + line)
            continue
        # Lowercase test name here
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
        st.info("Freeform parsing failed, attempting table parsing...")
        tests_table, total_lines_table = parse_table_report(text)
        tests += tests_table
        total_lines += total_lines_table
    if not tests:
        st.error("No tests could be parsed with available strategies.")
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

# ----- Visualization Functions -----
def visualize_numeric_test(test):
    fig, ax = plt.subplots(figsize=(8, 2))
    if test['ref_operator'] == 'range':
        ax.hlines(y=0, xmin=test['ref_lower'], xmax=test['ref_upper'], color='grey',
                  linewidth=8, alpha=0.5, label=f"Ref Range: {test['ref_lower']} - {test['ref_upper']}")
    elif test['ref_operator'] in ['<', '≤']:
        ax.axvline(x=test['ref_upper'], color='grey', linestyle='--', label=f"Ref: {test['ref_operator']} {test['ref_upper']}")
    elif test['ref_operator'] in ['>', '≥']:
        ax.axvline(x=test['ref_lower'], color='grey', linestyle='--', label=f"Ref: {test['ref_operator']} {test['ref_lower']}")

    color = 'blue' if test['status'] == "Within normal range" else 'red'
    ax.scatter(test['value'], 0, marker='o', color=color, s=100, zorder=3,
               label=f"Value: {test['value']} {test['unit']}")
    ax.text(test['value'], 0, f' {test["value"]}', va='center', ha='left', fontsize=10, color=color)
    ax.set_xlabel("Value")
    ax.set_title(f"{test['test']} ({test['status']})")
    ax.set_yticks([])
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)  # Close the figure to free memory

def visualize_categorical_test(test):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(test['test'], 1, color='green')
    label = f"{test['value']} {test['unit']}".strip()
    ax.text(0, 0.5, label, ha='center', va='center', color='white',
            fontsize=12, fontweight='bold')
    ax.set_title(f"{test['test']} ({test['status']})")
    ax.set_xticks([])
    ax.set_yticks([])
    st.pyplot(fig)
    plt.close(fig)  # Close the figure

def visualize_all_tests(df):
    for index, row in df.iterrows():
        test = row.to_dict()
        if test['is_numeric']:
            visualize_numeric_test(test)
        else:
            visualize_categorical_test(test)

# ----- Main Pipeline -----
def run_pipeline(image_file, api_key):
    image_bytes = image_file.read()
    filename = image_file.name

    st.info("Processing the image...")
    text, ocr_response = extract_text(image_bytes, filename, api_key)
    st.subheader("Extracted Text")
    st.text(text)

    tests, total_lines = parse_medical_report(text)
    if not tests:
        st.error("No tests were parsed. Please check OCR accuracy or report format.")
        return pd.DataFrame()

    # ----- Correcting test name spelling using TextBlob -----
    for test in tests:
        # Correct the spelling of the test name
        corrected_name = str(TextBlob(test['test']).correct())
        test['test'] = corrected_name

    evaluated_tests = evaluate_tests(tests)

    st.subheader("Parsed and Evaluated Test Results")
    for result in evaluated_tests:
        st.write(f"**Test:** {result['test']}")
        st.write(f"- **Value:** {result['value']} {result['unit']}")
        st.write(f"- **Reference:** {result['ref_text']}")
        st.write(f"- **Status:** {result['status']}")
        st.write("---")

    accuracy = calculate_heuristic_accuracy(evaluated_tests, total_lines, ocr_response)
    st.write(f"**Estimated Extraction Accuracy:** {accuracy:.2f}%")

    df = pd.DataFrame(evaluated_tests)
    st.subheader("Extracted Tests DataFrame")
    st.dataframe(df)

    st.subheader("Visualizations")
    visualize_all_tests(df)

    return df

# ----- Streamlit App Layout -----
st.title("Medical Report OCR & Analysis App")

st.markdown("""
This app uses OCR to extract text from medical report images, parses lab tests, evaluates them against reference ranges, and provides visualizations.  
Now supports multiple reports for comparison.
""")

st.sidebar.header("Upload & Settings")
uploaded_files = st.sidebar.file_uploader(
    "Upload Medical Report Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
api_key_input = st.sidebar.text_input("OCR API Key", type="password")

def extract_date_from_filename(name):
    try:
        return datetime.datetime.strptime(name.split('.')[0], "%Y-%m-%d")
    except:
        return name

if st.sidebar.button("Process Reports"):
    if uploaded_files and api_key_input:
        reports_data = {}
        for file in uploaded_files:
            st.write(f"Processing: {file.name}")
            df = run_pipeline(file, api_key_input)
            if not df.empty:
                reports_data[file.name] = df

        if reports_data:
            # Combine all reports and return JSON instead of an analysis graph.
            st.subheader("Reports JSON Output")
            json_data = {src: df.to_dict(orient="records") for src, df in reports_data.items()}
            st.json(json_data)
    else:
        st.error("Please upload at least one image and enter your API key.")

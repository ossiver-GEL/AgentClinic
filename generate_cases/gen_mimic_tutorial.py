import os, csv
import json, re, time, argparse, os, sys, pathlib, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Iterable
try:
  from tqdm import tqdm
except ImportError:  # fallback if tqdm not installed
  def tqdm(x, **kwargs):
    return x
# Ensure project root on sys.path for 'llm' module import
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))
from llm import load_llm_config, query_model

# First install the MIMIC-IV dataset from 
# https://physionet.org/content/mimiciv/2.2/
# And place it into this folder (generated_cases)

#############################
# Streaming dataset loading #
#############################

# Change this according to need
base_str = "./datasets/mimic-iv-3.1/"

def stream_csv(path: str) -> Iterable[list]:
  with open(path, 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
      yield row, header

patient_info: Dict[str, Dict[str, Any]] = {}

# Build diagnosis code reverse map (d_icd_diagnoses.csv)
rev_diag_code: Dict[str, str] = {}
for row, header in tqdm(stream_csv(os.path.join(base_str, 'hosp/d_icd_diagnoses.csv')), desc='d_icd_diagnoses'):  # columns: icd_code, icd_version, long_title
  try:
    icd_code = row[0]
    long_title = row[2]
    rev_diag_code[icd_code] = long_title
  except Exception:
    continue

# Admissions (initialize patient records)
for row, header in tqdm(stream_csv(os.path.join(base_str, 'hosp/admissions.csv')), desc='admissions'):
  try:
    pat_id = row[0]
    if pat_id not in patient_info:
      patient_info[pat_id] = {
        'tests': {},
        'history': [],
        'diagnosis': -1,
        'diag_imp': 9999,
        'demographics': {'race': row[12] if len(row) > 12 else ''},
      }
  except Exception:
    continue

# Process diagnoses_icd to collect history / diagnosis counts
num_diagnoses: Dict[str, int] = {}
for row, header in tqdm(stream_csv(os.path.join(base_str, 'hosp/diagnoses_icd.csv')), desc='diagnoses_icd'):
  try:
    pat_id = row[0]
    icd_code = row[3]
    imp_raw = row[2]
    descr = rev_diag_code.get(icd_code, '')
    if pat_id not in patient_info:
      continue  # skip those not in admissions
    if pat_id not in num_diagnoses:
      num_diagnoses[pat_id] = 0
    if 'history' in descr.lower():
      patient_info[pat_id]['history'].append(descr)
    else:
      num_diagnoses[pat_id] += 1
      try:
        patient_info[pat_id]['diagnosis'] = descr
        patient_info[pat_id]['diag_imp'] = int(imp_raw)
      except Exception:
        pass
  except Exception:
    continue

# Select patients with <2 diagnoses up to 300
selected_ids = []
for pid, count in num_diagnoses.items():
  if count < 2:
    selected_ids.append(pid)
    if len(selected_ids) >= 300:
      break
selected_set = set(selected_ids)

# Patients.csv enrich demographics only for selected
for row, header in tqdm(stream_csv(os.path.join(base_str, 'hosp/patients.csv')), desc='patients'):
  try:
    pid = row[0]
    if pid in selected_set and pid in patient_info:
      patient_info[pid]['demographics']['gender'] = row[1]
      patient_info[pid]['demographics']['anchor_age'] = row[2]
  except Exception:
    continue

# Lab items mapping (d_labitems.csv)
rev_item_code: Dict[str, str] = {}
for row, header in tqdm(stream_csv(os.path.join(base_str, 'hosp/d_labitems.csv')), desc='d_labitems'):
  try:
    rev_item_code[row[0]] = (row[1] + ' ' + row[2]).strip()
  except Exception:
    continue

# OMR
for row, header in tqdm(stream_csv(os.path.join(base_str, 'hosp/omr.csv')), desc='omr'):
  try:
    pid = row[0]
    if pid in selected_set and row[3] and row[3] not in patient_info[pid]['tests']:
      patient_info[pid]['tests'][row[3]] = row[4]
  except Exception:
    continue

# Microbiology events
micro_header_index = None
for row, header in tqdm(stream_csv(os.path.join(base_str, 'hosp/microbiologyevents.csv')), desc='microbio'):
  try:
    if micro_header_index is None:
      # header captured in generator already; we rely on known names
      micro_header_index = {name: idx for idx, name in enumerate(header)}
    pid = row[1]
    if pid in selected_set:
      test_name = row[micro_header_index.get('test_name', 4)].lower() if len(row) > 4 else ''
      comments = row[micro_header_index.get('comments', 10)].lower() if len(row) > 10 else ''
      if test_name and test_name not in patient_info[pid]['tests']:
        patient_info[pid]['tests'][test_name] = comments
  except Exception:
    continue

# Lab events
lab_header_index = None
for row, header in tqdm(stream_csv(os.path.join(base_str, 'hosp/labevents.csv')), desc='labevents'):
  try:
    if lab_header_index is None:
      lab_header_index = {name: idx for idx, name in enumerate(header)}
    pid = row[1]
    if pid not in selected_set:
      continue
    value = row[lab_header_index.get('value', 5)] if len(row) > lab_header_index.get('value', 5) else ''
    itemid = row[lab_header_index.get('itemid', 4)] if len(row) > lab_header_index.get('itemid', 4) else ''
    if not value or '_' in value or itemid not in rev_item_code:
      continue
    test_name = rev_item_code[itemid]
    if test_name not in patient_info[pid]['tests']:
      patient_info[pid]['tests'][test_name] = value
  except Exception:
    continue

case_studies = {pid: patient_info[pid] for pid in selected_ids if pid in patient_info}

# Provide an example of the OSCE template
examples = """
Here is an example of the structure:
{
  "OSCE_Examination": {
    "Objective_for_Doctor": "Assess and diagnose the patient presenting with acute abdominal pain.",
    "Patient_Actor": {
      "Demographics": "30-year-old female",
      "History": "The patient complains of sudden onset of sharp, right lower quadrant abdominal pain since last night. The pain has progressively worsened over the last 12 hours. She mentions that she felt nauseous this morning but has not vomited. No recent changes in bowel habits or urinary symptoms have been reported.",
      "Symptoms": {
        "Primary_Symptom": "Sharp, right lower quadrant abdominal pain",
        "Secondary_Symptoms": ["Nausea", "No vomiting", "No change in bowel habits", "No urinary symptoms"]
      },
      "Past_Medical_History": "No significant past medical history. No previous surgeries.",
      "Social_History": "Non-smoker, occasional alcohol use. Works as a software developer.",
      "Review_of_Systems": "Denies fever, vomiting, diarrhea, dysuria, or flank pain."
    },
    "Physical_Examination_Findings": {
      "Vital_Signs": {
        "Temperature": "37.2°C (99°F)",
        "Blood_Pressure": "120/75 mmHg",
        "Heart_Rate": "78 bpm",
        "Respiratory_Rate": "16 breaths/min"
      },
      "Abdominal_Examination": {
        "Inspection": "No distension or visible masses.",
        "Auscultation": "Normal bowel sounds.",
        "Percussion": "Tympanic throughout, no shifting dullness.",
        "Palpation": "Tenderness in the right lower quadrant. No guarding or rebound tenderness. Rovsing's sign positive, suggesting peritoneal irritation."
      }
    },
    "Test_Results": {
      "Complete_Blood_Count": {
        "WBC": "12,000 /μL (elevated)",
        "Hemoglobin": "13.5 g/dL",
        "Platelets": "250,000 /μL"
      },
      "Urinalysis": {
        "Appearance": "Clear",
        "WBC": "2-5 /HPF",
        "RBC": "0-2 /HPF",
        "Nitrites": "Negative",
        "Leukocyte_Esterase": "Negative"
      },
      "Imaging": {
        "Ultrasound_Abdomen": {
          "Findings": "Enlarged appendix with wall thickening and fluid collection. No evidence of ovarian cyst or ectopic pregnancy."
        }
      }
    },
    "Correct_Diagnosis": "Acute Appendicitis",
  }
}
"""

def main():
  parser = argparse.ArgumentParser(description="Generate MIMIC-derived OSCE cases using unified LLM & generation config")
  parser.add_argument('--gen-config', type=str, default='generate_cases\gen.config.json', help='Path to generation config JSON')
  parser.add_argument('--llm-config', type=str, default=None, help='Override llm.config.json path')
  parser.add_argument('--model', type=str, default=None, help='Override model name')
  parser.add_argument('--output', type=str, default=None, help='Override output file')
  parser.add_argument('--limit', type=int, default=None, help='Override generation limit')
  parser.add_argument('--workers', type=int, default=1, help='Number of parallel LLM workers')
  args = parser.parse_args()

  if not os.path.exists(args.gen_config):
    raise FileNotFoundError(f"Generation config not found: {args.gen_config}")
  with open(args.gen_config, 'r', encoding='utf-8') as f:
    gen_cfg = json.load(f)
  common = gen_cfg.get('common', {})
  mimic_cfg = gen_cfg.get('mimic', {})
  model = args.model or common.get('model', 'gpt-4o-mini')
  output = args.output or mimic_cfg.get('output', 'grounded.jsonl')
  limit = args.limit or mimic_cfg.get('limit', 100)
  sleep_seconds = float(common.get('sleep_seconds', 1.0))
  # Dataset base dir override
  base_dir_override = mimic_cfg.get('base_dataset_dir')
  if base_dir_override:
    global base_str
    base_str = base_dir_override

  load_llm_config()

  system_prompt = "Generate OSCE cases"

  targets = list(case_studies.items())[:limit]
  lock = threading.Lock()
  # Truncate output file first
  open(output, 'w', encoding='utf-8').close()

  def worker(pat_pair):
    pat_id, data = pat_pair
    user_prompt = (
      f"Generate an OSCE JSON for this patient data: {data}\n"
      "Follow the example format strictly. Example format:\n" + examples + "\nReturn ONLY JSON."
    )
    try:
      answer = query_model(model, user_prompt, system_prompt)
      answer = re.sub(r"\s+", " " , answer)
      answer = answer.replace("```json ", "").replace("```json", "").replace("```", "")
      json.loads(answer)  # validate
      with lock:
        with open(output, 'a', encoding='utf-8') as f:
          f.write(answer + "\n")
      return True, pat_id, None
    except Exception as e:
      return False, pat_id, str(e)

  workers = max(1, args.workers)
  if workers == 1:
    for item in targets:
      ok, pid, err = worker(item)
      if not ok:
        print(f"Failed to parse/generate for patient {pid}, skipping... Error: {err}")
      time.sleep(sleep_seconds)
  else:
    with ThreadPoolExecutor(max_workers=workers) as executor:
      futures = {executor.submit(worker, item): item[0] for item in targets}
      for fut in tqdm(as_completed(futures), total=len(futures), desc='generating'):
        ok, pid, err = fut.result()
        if not ok:
          print(f"Failed to parse/generate for patient {pid}, skipping... Error: {err}")
        if sleep_seconds > 0:
          time.sleep(sleep_seconds / workers)

if __name__ == "__main__":
  main()

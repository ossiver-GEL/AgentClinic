import json, re, time, argparse, os, sys, pathlib
# Ensure project root on sys.path for 'llm' module import when running from subdirectory
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))
from llm import load_llm_config, query_model
from datasets import load_dataset

def main():
  parser = argparse.ArgumentParser(description="Generate MedQA OSCE cases using unified LLM config")
  parser.add_argument('--gen-config', type=str, default='gen.config.json', help='Path to generation config JSON')
  parser.add_argument('--llm-config', type=str, default=None, help='Override llm.config.json path')
  parser.add_argument('--model', type=str, default=None, help='Override model name')
  parser.add_argument('--output', type=str, default=None, help='Override output JSONL file')
  parser.add_argument('--limit', type=int, default=None, help='Override max cases')
  args = parser.parse_args()
  # Load generation config
  if not os.path.exists(args.gen_config):
    raise FileNotFoundError(f"Generation config not found: {args.gen_config}")
  with open(args.gen_config, 'r', encoding='utf-8') as f:
    gen_cfg = json.load(f)
  common = gen_cfg.get('common', {})
  medqa_cfg = gen_cfg.get('medqa', {})
  llm_conf_path = args.llm_config or common.get('llm_config', 'llm.config.json')
  model = args.model or common.get('model', 'gpt-4o-mini')
  output = args.output or medqa_cfg.get('output', 'grounded.jsonl')
  limit = args.limit or medqa_cfg.get('limit')
  sleep_seconds = float(common.get('sleep_seconds', 1.0))

  load_llm_config(llm_conf_path)

  # Extract the testing set for the MedQA dataset
  medqa_test_set = load_dataset("bigbio/med_qa")["test"]

  # Extract all case studies from MedQA
  case_studies = [case for case in medqa_test_set if "likely diagnosis?" in case["question"]]

  # Randomize cases (optional)
  import random
  random.shuffle(case_studies)

  # How many cases studies to generate
  cases_to_gen = limit or (108 - 78)  # config override > legacy fallback

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

  outp_str = ""
  cases_generated = 0
  target_total = cases_to_gen
  system_prompt = "Generate OSCE cases"
  for _case in case_studies:
    user_prompt = (
      f"Generate an OSCE for the following MedQA case study: {_case}\n"
      "Use the provided answer field as ground truth.\n\nExample format:\n" + examples + "\n\nPlease create a new one now:"
    )
    try:
      answer = query_model(model, user_prompt, system_prompt)
      answer = re.sub(r"\s+", " ", answer)
      answer = answer.replace("```json ", "").replace("```json", "").replace("```", "")
      parsed = json.loads(answer)
      gt = _case.get("answer", "").lower()
      cd = str(parsed.get("OSCE_Examination", {}).get("Correct_Diagnosis", "")).lower()
      if not gt or gt != cd:
        continue
      outp_str += json.dumps(parsed, ensure_ascii=False) + "\n"
      with open(output, "w", encoding="utf-8") as f:
        f.write(outp_str)
      cases_generated += 1
    except Exception:
      pass
    if cases_generated >= target_total:
      break
    time.sleep(sleep_seconds)

if __name__ == "__main__":
  main()

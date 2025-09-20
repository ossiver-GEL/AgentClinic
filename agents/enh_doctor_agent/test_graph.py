import re
from typing import Dict, List, Tuple


def canonicalize_test_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    return cleaned.strip("_")


_TEST_GRAPH: Dict[str, Dict[str, List[str] or str]] = {
    "vital_signs_panel": {"prerequisites": [], "cost": "low"},
    "basic_metabolic_panel": {"prerequisites": [], "cost": "low"},
    "comprehensive_metabolic_panel": {"prerequisites": ["basic_metabolic_panel"], "cost": "moderate"},
    "complete_blood_count": {"prerequisites": [], "cost": "low"},
    "serum_electrolytes": {"prerequisites": ["basic_metabolic_panel"], "cost": "low"},
    "liver_function_tests": {"prerequisites": ["basic_metabolic_panel"], "cost": "moderate"},
    "renal_function_panel": {"prerequisites": ["basic_metabolic_panel"], "cost": "moderate"},
    "blood_culture": {"prerequisites": ["complete_blood_count"], "cost": "moderate"},
    "urinalysis": {"prerequisites": [], "cost": "low"},
    "pregnancy_test": {"prerequisites": [], "cost": "low"},
    "d_dimer": {"prerequisites": ["complete_blood_count"], "cost": "low"},
    "troponin": {"prerequisites": ["electrocardiogram"], "cost": "moderate"},
    "electrocardiogram": {"prerequisites": ["vital_signs_panel"], "cost": "low"},
    "echocardiogram": {"prerequisites": ["electrocardiogram"], "cost": "moderate"},
    "cardiac_stress_test": {"prerequisites": ["electrocardiogram"], "cost": "high"},
    "holter_monitor": {"prerequisites": ["electrocardiogram"], "cost": "moderate"},
    "chest_x_ray": {"prerequisites": [], "cost": "low"},
    "ct_chest": {"prerequisites": ["chest_x_ray"], "cost": "high"},
    "ct_pulmonary_angiography": {"prerequisites": ["chest_x_ray", "d_dimer"], "cost": "high"},
    "ultrasound_abdomen": {"prerequisites": [], "cost": "moderate"},
    "ct_abdomen": {"prerequisites": ["ultrasound_abdomen"], "cost": "high"},
    "mri_abdomen": {"prerequisites": ["ct_abdomen"], "cost": "high"},
    "ct_head": {"prerequisites": [], "cost": "high"},
    "mri_brain": {"prerequisites": ["ct_head"], "cost": "high"},
    "mri_spine": {"prerequisites": ["ct_spine"], "cost": "high"},
    "ct_spine": {"prerequisites": ["plain_spine_x_ray"], "cost": "high"},
    "plain_spine_x_ray": {"prerequisites": [], "cost": "moderate"},
    "lower_extremity_ultrasound": {"prerequisites": ["d_dimer"], "cost": "moderate"},
    "arterial_blood_gas": {"prerequisites": ["vital_signs_panel"], "cost": "moderate"},
    "pulmonary_function_tests": {"prerequisites": ["chest_x_ray"], "cost": "moderate"},
    "spirometry": {"prerequisites": ["chest_x_ray"], "cost": "low"}
}


_TEST_SYNONYMS: Dict[str, str] = {
    "cbc": "complete_blood_count",
    "complete blood count": "complete_blood_count",
    "cmp": "comprehensive_metabolic_panel",
    "bmp": "basic_metabolic_panel",
    "electrocardiogram": "electrocardiogram",
    "ecg": "electrocardiogram",
    "ekg": "electrocardiogram",
    "stress_test": "cardiac_stress_test",
    "cardiac stress test": "cardiac_stress_test",
    "treadmill stress test": "cardiac_stress_test",
    "echo": "echocardiogram",
    "echocardiography": "echocardiogram",
    "xray": "chest_x_ray",
    "chest x-ray": "chest_x_ray",
    "ct pulmonary angiography": "ct_pulmonary_angiography",
    "cta": "ct_pulmonary_angiography",
    "ct angiography": "ct_pulmonary_angiography",
    "ct chest": "ct_chest",
    "ct abdomen": "ct_abdomen",
    "ultrasound": "ultrasound_abdomen",
    "abdominal ultrasound": "ultrasound_abdomen",
    "ct head": "ct_head",
    "head ct": "ct_head",
    "brain mri": "mri_brain",
    "ct brain": "ct_head",
    "mri": "mri_brain",
    "plain xray": "chest_x_ray",
    "spirometry": "spirometry",
    "pfts": "pulmonary_function_tests"
}

_DISPLAY_OVERRIDES: Dict[str, str] = {
    "chest_x_ray": "Chest_X-Ray",
    "ct_chest": "CT_Chest",
    "ct_pulmonary_angiography": "CT_Pulmonary_Angiography",
    "ultrasound_abdomen": "Ultrasound_Abdomen",
    "ct_abdomen": "CT_Abdomen",
    "mri_abdomen": "MRI_Abdomen",
    "ct_head": "CT_Head",
    "mri_brain": "MRI_Brain",
    "mri_spine": "MRI_Spine",
    "ct_spine": "CT_Spine",
    "plain_spine_x_ray": "Plain_Spine_X-Ray",
    "lower_extremity_ultrasound": "Lower_Extremity_Ultrasound",
    "arterial_blood_gas": "Arterial_Blood_Gas",
    "pulmonary_function_tests": "Pulmonary_Function_Tests",
    "spirometry": "Spirometry",
    "vital_signs_panel": "Vital_Signs_Panel",
    "basic_metabolic_panel": "Basic_Metabolic_Panel",
    "comprehensive_metabolic_panel": "Comprehensive_Metabolic_Panel",
    "complete_blood_count": "Complete_Blood_Count",
    "serum_electrolytes": "Serum_Electrolytes",
    "liver_function_tests": "Liver_Function_Tests",
    "renal_function_panel": "Renal_Function_Panel",
    "blood_culture": "Blood_Culture",
    "urinalysis": "Urinalysis",
    "pregnancy_test": "Pregnancy_Test",
    "d_dimer": "D-Dimer",
    "troponin": "Troponin",
    "electrocardiogram": "Electrocardiogram",
    "echocardiogram": "Echocardiogram",
    "cardiac_stress_test": "Cardiac_Stress_Test",
    "holter_monitor": "Holter_Monitor"
}


def resolve_test_name(name: str) -> Tuple[str, Dict[str, List[str] or str]]:
    key = canonicalize_test_name(name)
    canonical = _TEST_SYNONYMS.get(key, key)
    if canonical not in _TEST_GRAPH:
        return canonical, {"prerequisites": [], "cost": "moderate"}
    return canonical, _TEST_GRAPH[canonical]


def missing_prerequisites(test_name: str, completed: List[str], pending: List[str]) -> List[str]:
    canonical, meta = resolve_test_name(test_name)
    completed_keys = {canonicalize_test_name(t) for t in completed}
    pending_keys = {canonicalize_test_name(t) for t in pending}
    missing = []
    for prereq in meta.get("prerequisites", []):
        if prereq not in completed_keys and prereq not in pending_keys:
            missing.append(prereq)
    return missing


def cost_level(test_name: str) -> str:
    canonical, meta = resolve_test_name(test_name)
    return meta.get("cost", "moderate")


def display_name(test_name: str) -> str:
    canonical, _ = resolve_test_name(test_name)
    if canonical in _DISPLAY_OVERRIDES:
        return _DISPLAY_OVERRIDES[canonical]
    pretty = canonical.replace("_", " ").title()
    pretty = pretty.replace("Ct", "CT").replace("Mri", "MRI").replace("X Ray", "X-Ray")
    return pretty.replace(" ", "_")

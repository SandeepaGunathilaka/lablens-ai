"""Keyword and phrase lists used by the Safety Agent.

All matching is case-insensitive and whole-word, and multi-word phrases tolerate any
whitespace between words. Except for MEDICATION_ACTIONS, a trailing "s"/"es" plural
is accepted automatically, so list singular forms only.
"""

# --- Diagnosis -------------------------------------------------------------------

# Always a diagnosis, wherever they appear.
DIAGNOSIS_PHRASES = [
    "diagnosed with",
    "diagnosis of",
    "you have been diagnosed",
    "you are anemic",
    "you are anaemic",
    "you are diabetic",
    "you are prediabetic",
    "you suffer from",
    "you are suffering from",
]

# Personal statements that only count as a diagnosis when a DISEASE_TERM appears in
# the same sentence ("you have anemia" is flagged, "if you have questions" is not).
DIAGNOSIS_ASSERTIONS = [
    "you have",
    "you've got",
    "you may have",
    "you might have",
    "you likely have",
    "you probably have",
    "you could have",
    "you are at risk of",
    "this means you",
    "this confirms",
    "this indicates you",
    "this suggests you",
]

# Disease names. On their own they are NOT flagged as a diagnosis, because explaining
# a test often needs them ("HbA1c is used to monitor diabetes"). They are also used
# by the unsupported-claims check.
DISEASE_TERMS = [
    "anemia",
    "anaemia",
    "iron deficiency",
    "diabetes",
    "prediabetes",
    "hypothyroidism",
    "hyperthyroidism",
    "kidney disease",
    "chronic kidney disease",
    "liver disease",
    "fatty liver",
    "hepatitis",
    "cirrhosis",
    "leukemia",
    "leukaemia",
    "lymphoma",
    "cancer",
    "tumor",
    "tumour",
    "infection",
    "sepsis",
    "thalassemia",
    "sickle cell disease",
    "heart disease",
    "heart attack",
    "stroke",
    "hypertension",
    "high blood pressure",
    "hyperlipidemia",
    "gout",
    "lupus",
    "rheumatoid arthritis",
    "celiac disease",
    "vitamin deficiency",
]

# --- Medication -------------------------------------------------------------------

# Always medication advice, wherever they appear.
MEDICATION_PHRASES = [
    "prescribe",
    "prescribed",
    "prescription",
    "dosage",
    "dose",
    "mg per day",
    "mg daily",
    "twice daily",
    "once daily",
    "over the counter",
    "over-the-counter",
]

# Specific drug names: always flagged, since explaining lab results never requires them.
DRUG_NAMES = [
    "metformin",
    "insulin",
    "statin",
    "atorvastatin",
    "rosuvastatin",
    "simvastatin",
    "levothyroxine",
    "aspirin",
    "ibuprofen",
    "paracetamol",
    "acetaminophen",
    "warfarin",
    "heparin",
    "amoxicillin",
    "antibiotic",
    "prednisone",
    "omeprazole",
    "ferrous sulfate",
    "ferrous sulphate",
    "iron supplement",
    "folic acid supplement",
    "vitamin supplement",
]

# Action verbs that only count as medication advice when a MEDICATION_TERM appears in
# the same sentence ("take iron supplements" is flagged, "take this report to your
# doctor" is not). Matched exactly, without plurals, so descriptive phrasing like
# "your body takes in vitamin B12 from food" is not flagged.
MEDICATION_ACTIONS = [
    "take",
    "taking",
    "start",
    "starting",
    "stop",
    "stopping",
    "try",
    "increase",
    "decrease",
    "reduce",
]

MEDICATION_TERMS = [
    "medication",
    "medicine",
    "drug",
    "supplement",
    "pill",
    "tablet",
    "capsule",
    "injection",
    "vitamin",
]

# --- Unsupported claims -----------------------------------------------------------

# Medical terms that must trace back to the patient's results or the retrieved
# sources. Disease names are always included (see UNSUPPORTED_CLAIM_TERMS below).
MEDICAL_TERMS = [
    "bone marrow",
    "kidney",
    "liver",
    "thyroid",
    "pancreas",
    "spleen",
    "heart",
    "blood loss",
    "bleeding",
    "internal bleeding",
    "dehydration",
    "inflammation",
    "malnutrition",
    "pregnancy",
    "vitamin b12",
    "folate",
    "ferritin",
    "red blood cell",
    "white blood cell",
    "platelet",
    "hormone",
    "immune system",
]

UNSUPPORTED_CLAIM_TERMS = MEDICAL_TERMS + DISEASE_TERMS

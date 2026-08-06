"""Built-in syllabus catalog exposed by the MAID user interface."""

from chinese_medical_syllabus import CHINESE_MEDICAL_LICENSING_SYLLABUS
from usmle_step1_syllabus import USMLE_STEP1_SYLLABUS


DEFAULT_SYLLABUS_NAME = "USMLE Step 1 (English)"

BUILTIN_SYLLABUSES = {
    DEFAULT_SYLLABUS_NAME: USMLE_STEP1_SYLLABUS,
    "Chinese Medical Licensing Examination (Chinese)": CHINESE_MEDICAL_LICENSING_SYLLABUS,
}

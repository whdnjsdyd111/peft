from tasks.glue.dataset import task_to_keys as glue_tasks
from tasks.superglue.dataset import task_to_keys as superglue_tasks
from tasks.qa.dataset import task_to_keys as qa_tasks

GLUE_DATASETS = list(glue_tasks.keys())
SUPERGLUE_DATASETS = list(superglue_tasks.keys())
QA_DATASETS = list(qa_tasks.keys())

TASKS = [
    "glue", "super_glue", "qa"
]

DATASETS = GLUE_DATASETS + SUPERGLUE_DATASETS + QA_DATASETS
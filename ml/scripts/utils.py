from datetime import datetime, timedelta
from pathlib import Path

import sys

file = Path(__file__).resolve()
parent , root = file.parent , file.parents[1]
# print(f"parent--{parent}")
# print(f"root--{root}")
sys.path.append(str(root))


def utc_to_ist(utc_dt):
    return utc_dt + timedelta(hours=5, minutes=30)

def get_current_ist():
    current_utc_time = datetime.utcnow()
    current_ist_time = utc_to_ist(current_utc_time)
    return current_ist_time


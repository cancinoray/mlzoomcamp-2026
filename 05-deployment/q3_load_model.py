"""Question 3: load `pipeline.bin` directly with pickle and score one lead.

This is the "no server" path: the same artifact the API serves is unpickled in
process. Going through `normalize_record` keeps the training-time
missing-value policy identical to the API, even though this lead has every
field populated.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from model import normalize_record


MODEL_PATH = Path(__file__).with_name("pipeline.bin")
LEAD = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}


def main() -> None:
    with MODEL_PATH.open("rb") as model_file:
        pipeline = pickle.load(model_file)

    probability = float(pipeline.predict_proba([normalize_record(LEAD)])[0, 1])
    print(
        json.dumps(
            {
                "conversion_probability": round(probability, 3),
                "raw_probability": probability,
                "conversion": probability >= 0.5,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

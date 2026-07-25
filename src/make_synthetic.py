"""Generate a synthetic review corpus so the pipeline can be run on sample input.

Each review is drawn from one of eight latent themes whose shares mirror the topic
proportions the published study reports. The text is then assembled from that theme's
phrase pools, sampled and ordered at random, so the corpus carries the same shape a real
review set has: recurring vocabulary per theme, but few identical reviews.

The output is machine-written text. No row corresponds to a real review, a real person or
a real employer, and the company labels are placeholders. It exists to demonstrate that
the code path runs end to end, not to reproduce the published findings, which depend on
the real corpus.

Usage:
    python src/make_synthetic.py --n 6000 --out data/synthetic_reviews.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMPANIES = [f"Company {chr(65 + i)}" for i in range(11)]
POSITIONS = [
    "Customer Service Representative", "Software Engineer", "Project Manager",
    "Systems Administrator", "Business Analyst", "Technical Support Specialist",
    "Senior Consultant", "Network Engineer", "Team Lead", "Data Analyst",
]

# share, rating pool, and phrase pools per theme. Shares follow the topic proportions
# reported in the study.
THEMES = [
    {
        "name": "Features of Great Companies", "share": 0.23, "ratings": [4, 5, 5],
        "pros": ["great place to work", "good people and a strong work environment",
                 "great opportunity to learn", "good benefits and a good environment",
                 "lots of opportunity to grow", "great culture and great people",
                 "good work environment", "great benefits and a good place"],
        "cons": ["little to complain about", "the busy period can stretch",
                 "large place so things move slowly", "nothing serious to note"],
        "body": ["a great company with a good culture", "good place with real opportunity",
                 "great environment and good people to learn from",
                 "the work environment is good and the people are great"],
    },
    {
        "name": "Economic Value and Layoffs", "share": 0.17, "ratings": [1, 2, 2, 3],
        "pros": ["the pay arrived on time", "good benefits while they lasted",
                 "decent pay at the start"],
        "cons": ["constant layoffs every year", "low pay and no raise for years",
                 "job security is poor", "no pay raise while costs climbed",
                 "layoffs come round every year", "low pay for the hours asked",
                 "job security disappeared after the change"],
        "body": ["the company cut people every year and the pay never changed",
                 "layoffs and low pay defined my time there",
                 "no job security, they let people go without warning"],
    },
    {
        "name": "Social and Development Value", "share": 0.14, "ratings": [4, 5],
        "pros": ["strong culture and real career opportunities",
                 "good technology and interesting clients",
                 "career development is taken seriously", "a supportive organisation"],
        "cons": ["career progress can be slow", "a large organisation moves slowly"],
        "body": ["the culture and the technology gave me a career path",
                 "good career opportunities across a strong organisation",
                 "the client work developed my career"],
    },
    {
        "name": "Joy at Work", "share": 0.12, "ratings": [4, 5],
        "pros": ["enjoyable typical day", "learned a lot from the team",
                 "the people made the day enjoyable", "enjoyed the daily work"],
        "cons": ["some days were long", "the hardest part was the pace"],
        "body": ["a typical day was enjoyable and i learned something new",
                 "the most enjoyable part was the people, the hardest part was leaving",
                 "enjoyed the work and learned every day"],
    },
    {
        "name": "Management Value", "share": 0.11, "ratings": [1, 1, 2],
        "pros": ["a few good supervisors", "one decent team lead"],
        "cons": ["poor management and no leadership", "upper management expects too much",
                 "bad management and high stress", "no support from management",
                 "poor management with little direction", "management lacks the skill"],
        "body": ["management expects a lot and gives little support",
                 "poor management, high stress and no team support",
                 "the supervisor and upper management were the problem"],
    },
    {
        "name": "Job Duties", "share": 0.10, "ratings": [3, 4],
        "pros": ["clear process and a supportive project team",
                 "good project structure", "clear role and steady service work"],
        "cons": ["the process is slow", "too much process for a small project"],
        "body": ["provided customer support on the service product",
                 "supported the project team and delivered the service",
                 "managed the process and the product for the customer team",
                 "project work covering support and service delivery"],
    },
    {
        "name": "Application Value", "share": 0.08, "ratings": [2, 3],
        "pros": ["good training when you start", "the health cover was fine",
                 "you learn the systems quickly"],
        "cons": ["long hours on call with a short break", "short break between calls",
                 "low pay for a long day of calls", "long hour shifts with a short lunch"],
        "body": ["handled calls all day with a short break",
                 "the training was good but the day is call after call",
                 "long hours answering calls for low pay"],
    },
    {
        "name": "Location and Working Conditions", "share": 0.05, "ratings": [3, 4, 5],
        "pros": ["comfortable office and a short commute", "good computer equipment",
                 "a helpful office and a clean center"],
        "cons": ["the call center office felt dated", "the office equipment is old"],
        "body": ["worked in the call center office helping people on the phone",
                 "the office and the computer setup shaped the day"],
    },
]

TITLES = {
    "high": ["Great place to work", "Good company", "Enjoyed my time", "Would recommend",
             "Solid employer", "Good culture"],
    "mid": ["Mixed experience", "Depends on the team", "Average employer", "Okay overall",
            "Good and bad"],
    "low": ["Poor management", "Not what was promised", "Avoid", "Disappointing",
            "Stressful and underpaid"],
}


def _title_pool(rating: int) -> str:
    return "high" if rating >= 4 else "low" if rating <= 2 else "mid"


def _sample_join(rng: random.Random, pool: list[str], low: int, high: int) -> str:
    """Pick a few fragments and join them, so few reviews come out identical."""
    k = min(len(pool), rng.randint(low, high))
    return ". ".join(rng.sample(pool, k=k))


def generate(n: int, seed: int = 11) -> list[dict]:
    rng = random.Random(seed)
    shares = [t["share"] for t in THEMES]
    rows: list[dict] = []
    for i in range(n):
        theme = rng.choices(THEMES, weights=shares, k=1)[0]
        rating = rng.choice(theme["ratings"])
        # Later years carry more weight, so the covariate profiling has a trend to find.
        year = rng.choices(range(2012, 2021),
                           weights=[1, 1, 1.2, 1.3, 1.5, 1.7, 2, 2.2, 2.4], k=1)[0]
        status = rng.choices(["Current Employee", "Former Employee"], weights=[45, 55], k=1)[0]
        rows.append(
            {
                "id": i + 1,
                "review_title": rng.choice(TITLES[_title_pool(rating)]),
                "review": _sample_join(rng, theme["body"], 1, 2),
                "pros": _sample_join(rng, theme["pros"], 1, 3),
                "cons": _sample_join(rng, theme["cons"], 1, 2),
                "rating": rating,
                "job_status": status,
                "position": rng.choice(POSITIONS),
                "company": rng.choice(COMPANIES),
                "year": year,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=6000, help="number of synthetic reviews")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "synthetic_reviews.csv")
    args = parser.parse_args()

    rows = generate(args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} synthetic reviews to {args.out}")


if __name__ == "__main__":
    main()

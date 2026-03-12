"""Build movie_ood dataset from TMDB API.

Fetches English-language films with primary_release_date >= 2025-08-01
(post MedGemma training cutoff July 2025) and generates 4-choice MCQ questions
from structured metadata: director, cast, genre, and production country.

Usage:
    export TMDB_API_KEY=your_key_here
    python scripts/build_movie_ood_dataset.py
    python scripts/build_movie_ood_dataset.py --output data/movie_ood.jsonl --min-date 2025-08-01 --pages 20

Output JSONL format (compatible with MedQADataset loader):
    {"question": "...", "options": {"A": "...", "B": "...", "C": "...", "D": "..."}, "answer": "A",
     "metadata": {"movie_id": ..., "title": ..., "release_date": ..., "question_type": ...}}

Requirements: requests (already in project deps)
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

TMDB_BASE = "https://api.themoviedb.org/3"
GENRES = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Science Fiction",
    10770: "TV Movie",
    53: "Thriller",
    10752: "War",
    37: "Western",
}


# ---------------------------------------------------------------------------
# TMDB API helpers
# ---------------------------------------------------------------------------


def tmdb_get(path: str, api_key: str, params: Dict = None, retries: int = 3) -> Dict:
    url = f"{TMDB_BASE}{path}"
    p = {"api_key": api_key, "language": "en-US"}
    if params:
        p.update(params)
    for attempt in range(retries):
        try:
            r = requests.get(url, params=p, timeout=10)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    return {}


def fetch_movies(api_key: str, min_date: str, pages: int = 20) -> List[Dict]:
    """Fetch popular English-language movies released on or after min_date."""
    movies = []
    for page in range(1, pages + 1):
        data = tmdb_get(
            "/discover/movie",
            api_key,
            {
                "primary_release_date.gte": min_date,
                "sort_by": "popularity.desc",
                "with_original_language": "en",
                "page": page,
            },
        )
        results = data.get("results", [])
        if not results:
            break
        movies.extend(results)
        total_pages = data.get("total_pages", 1)
        if page >= total_pages:
            break
        time.sleep(0.25)  # rate limit courtesy
    print(f"  Fetched {len(movies)} movies across {page} pages")
    return movies


def fetch_credits(api_key: str, movie_id: int) -> Tuple[Optional[str], List[str]]:
    """Return (director_name, [top_cast_names])."""
    try:
        data = tmdb_get(f"/movie/{movie_id}/credits", api_key)
    except Exception:
        return None, []
    director = None
    for crew in data.get("crew", []):
        if crew.get("job") == "Director":
            director = crew.get("name")
            break
    cast = [m["name"] for m in data.get("cast", [])[:5]]
    return director, cast


def fetch_details(api_key: str, movie_id: int) -> Dict:
    try:
        return tmdb_get(f"/movie/{movie_id}", api_key)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# MCQ generation
# ---------------------------------------------------------------------------


def _shuffle_choices(
    correct: str, distractors: List[str], rng: random.Random
) -> Tuple[Dict[str, str], str]:
    """Shuffle correct + 3 distractors into A/B/C/D, return (options_dict, answer_letter)."""
    pool = [correct] + distractors[:3]
    while len(pool) < 4:
        pool.append("Unknown")
    rng.shuffle(pool)
    letters = "ABCD"
    options = {letters[i]: pool[i] for i in range(4)}
    answer = letters[pool.index(correct)]
    return options, answer


def make_director_question(
    movie: Dict, director: str, all_directors: List[str], rng: random.Random
) -> Optional[Dict]:
    title = movie.get("title", "")
    year = (movie.get("release_date") or "")[:4]
    distractors = [d for d in all_directors if d != director]
    if len(distractors) < 3:
        return None
    distractors = rng.sample(distractors, 3)
    options, answer = _shuffle_choices(director, distractors, rng)
    return {
        "question": f"Who directed the film '{title}' ({year})?",
        "options": options,
        "answer": answer,
        "metadata": {
            "movie_id": movie["id"],
            "title": title,
            "release_date": movie.get("release_date"),
            "question_type": "director",
        },
    }


def make_cast_question(
    movie: Dict, cast: List[str], all_cast: List[str], rng: random.Random
) -> Optional[Dict]:
    if not cast:
        return None
    title = movie.get("title", "")
    year = (movie.get("release_date") or "")[:4]
    star = cast[0]
    distractors = [c for c in all_cast if c not in cast]
    if len(distractors) < 3:
        return None
    distractors = rng.sample(distractors, 3)
    options, answer = _shuffle_choices(star, distractors, rng)
    return {
        "question": f"Which of the following actors starred in '{title}' ({year})?",
        "options": options,
        "answer": answer,
        "metadata": {
            "movie_id": movie["id"],
            "title": title,
            "release_date": movie.get("release_date"),
            "question_type": "cast",
        },
    }


def make_genre_question(movie: Dict, rng: random.Random) -> Optional[Dict]:
    genre_ids = movie.get("genre_ids", [])
    if not genre_ids:
        return None
    correct_genre = GENRES.get(genre_ids[0])
    if not correct_genre:
        return None
    title = movie.get("title", "")
    year = (movie.get("release_date") or "")[:4]
    other_genres = [g for gid, g in GENRES.items() if gid != genre_ids[0]]
    distractors = rng.sample(other_genres, 3)
    options, answer = _shuffle_choices(correct_genre, distractors, rng)
    return {
        "question": f"What is the primary genre of '{title}' ({year})?",
        "options": options,
        "answer": answer,
        "metadata": {
            "movie_id": movie["id"],
            "title": title,
            "release_date": movie.get("release_date"),
            "question_type": "genre",
        },
    }


def make_country_question(
    movie: Dict, details: Dict, all_countries: List[str], rng: random.Random
) -> Optional[Dict]:
    countries = details.get("production_countries", [])
    if not countries:
        return None
    correct = countries[0].get("name")
    if not correct:
        return None
    title = movie.get("title", "")
    year = (movie.get("release_date") or "")[:4]
    distractors = [c for c in all_countries if c != correct]
    if len(distractors) < 3:
        return None
    distractors = rng.sample(distractors, 3)
    options, answer = _shuffle_choices(correct, distractors, rng)
    return {
        "question": f"In which country was '{title}' ({year}) primarily produced?",
        "options": options,
        "answer": answer,
        "metadata": {
            "movie_id": movie["id"],
            "title": title,
            "release_date": movie.get("release_date"),
            "question_type": "country",
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build(api_key: str, min_date: str, pages: int, output: Path, seed: int):
    rng = random.Random(seed)

    print(f"[1/4] Fetching movies released >= {min_date}...")
    movies = fetch_movies(api_key, min_date, pages)

    print(f"[2/4] Fetching credits and details for {len(movies)} movies...")
    enriched = []
    all_directors: List[str] = []
    all_cast: List[str] = []
    all_countries: List[str] = []

    for i, movie in enumerate(movies):
        mid = movie["id"]
        director, cast = fetch_credits(api_key, mid)
        details = fetch_details(api_key, mid)

        if director:
            all_directors.append(director)
        all_cast.extend(cast)
        for c in details.get("production_countries", []):
            name = c.get("name")
            if name:
                all_countries.append(name)

        enriched.append({"movie": movie, "director": director, "cast": cast, "details": details})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(movies)} done...")
        time.sleep(0.15)

    # Deduplicate distractor pools
    all_directors = list(set(all_directors))
    all_cast = list(set(all_cast))
    all_countries = list(set(all_countries))

    print(
        f"  Directors: {len(all_directors)}, Cast: {len(all_cast)}, Countries: {len(all_countries)}"
    )

    print("[3/4] Generating MCQ questions...")
    questions: List[Dict[str, Any]] = []
    skipped = 0

    for item in enriched:
        movie = item["movie"]
        director = item["director"]
        cast = item["cast"]
        details = item["details"]

        if director:
            q = make_director_question(movie, director, all_directors, rng)
            if q:
                questions.append(q)
            else:
                skipped += 1

        q = make_cast_question(movie, cast, all_cast, rng)
        if q:
            questions.append(q)
        else:
            skipped += 1

        q = make_genre_question(movie, rng)
        if q:
            questions.append(q)
        else:
            skipped += 1

        q = make_country_question(movie, details, all_countries, rng)
        if q:
            questions.append(q)
        else:
            skipped += 1

    rng.shuffle(questions)
    print(f"  Generated {len(questions)} questions (skipped {skipped} with missing data)")

    print(f"[4/4] Writing to {output}...")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # Summary by question type
    type_counts: Dict[str, int] = defaultdict(int)
    for q in questions:
        type_counts[q["metadata"]["question_type"]] += 1
    print("\nQuestion type breakdown:")
    for qtype, count in sorted(type_counts.items()):
        print(f"  {qtype}: {count}")
    print(f"\nDone. {len(questions)} questions saved to {output}")


def main():
    parser = argparse.ArgumentParser(description="Build movie_ood MCQ dataset from TMDB API")
    parser.add_argument("--output", default="data/movie_ood.jsonl", help="Output JSONL path")
    parser.add_argument(
        "--min-date", default="2025-08-01", help="Minimum release date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--pages", type=int, default=20, help="TMDB discover pages to fetch (20 results/page)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for distractor shuffling")
    args = parser.parse_args()

    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        raise SystemExit(
            "TMDB_API_KEY environment variable not set.\n"
            "Get a free key at: https://developer.themoviedb.org/docs/getting-started"
        )

    build(
        api_key=api_key,
        min_date=args.min_date,
        pages=args.pages,
        output=Path(args.output),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

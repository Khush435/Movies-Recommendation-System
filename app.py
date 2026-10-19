import os
import pickle
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import joblib
import pandas as pd
import streamlit as st

# ----------------------------- Page setup -----------------------------
st.set_page_config(
    page_title="CineMatch | Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MOVIES_PATH = BASE_DIR / "movies.pickle"
SIMILARITY_PATH = BASE_DIR / "similarity.joblib"

# To enable TMDB posters, create .streamlit/secrets.toml in this project folder:
# TMDB_API_KEY = "your_tmdb_api_key"
# Do not commit secrets.toml or publish your API key.

# ----------------------------- Styling -----------------------------
# TMDB is used only for poster lookup; NLP recommendations remain local.
st.markdown(
    """
    <style>
    :root {
        --bg: #0b0d14;
        --panel: #151824;
        --panel-soft: #1b1e2c;
        --text: #f6f6fb;
        --muted: #b6bacb;
        --pink: #ff5c82;
        --orange: #ff9668;
        --border: rgba(255,255,255,.10);
    }

    .stApp {
        background:
            radial-gradient(ellipse at 8% 0%, rgba(255,92,130,.13), transparent 34%),
            radial-gradient(ellipse at 94% 7%, rgba(103,90,200,.15), transparent 30%),
            var(--bg);
        color: var(--text);
    }
    [data-testid="stHeader"] { background: rgba(11,13,20,.88); }
    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 1.7rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 { letter-spacing: -.035em; }
    .brand {
        display: block;
        width: 100%;
        color: #ffffff;
        font-size: 1.08rem;
        font-weight: 900;
        letter-spacing: .14em;
        line-height: 1.5;
        margin: .2rem 0 1rem;
        padding: .15rem 0;
        opacity: 1;
    }
    .brand span { color: var(--pink); }
    .hero {
        border: 1px solid var(--border);
        border-radius: 26px;
        padding: clamp(1.4rem, 4vw, 2.6rem);
        margin-bottom: 1.5rem;
        background: linear-gradient(115deg, rgba(255,92,130,.16), rgba(98,81,175,.13) 58%, rgba(255,255,255,.025));
        box-shadow: 0 18px 50px rgba(0,0,0,.18);
    }
    .eyebrow {
        color: #ff9eb4;
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .16em;
        text-transform: uppercase;
    }
    .hero-title {
        color: #fff;
        font-size: clamp(2.25rem, 5.2vw, 3.8rem);
        font-weight: 850;
        line-height: 1.06;
        letter-spacing: -.055em;
        margin: .65rem 0 .8rem;
    }
    .hero-title .accent { color: var(--pink); }
    .hero-copy {
        color: #c6c8d5;
        max-width: 690px;
        font-size: 1.02rem;
        line-height: 1.7;
        margin-bottom: 0;
    }
    .section-kicker {
        color: #c5c8d7;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .13em;
        text-transform: uppercase;
        margin: .3rem 0 .65rem;
    }
    .helper { color: var(--muted); font-size: .9rem; }
    div[data-testid="stSelectbox"] label {
        color: #f3f3fa;
        font-weight: 700;
    }
    div[data-testid="stSelectbox"] > div > div {
        background: #171a27;
        border: 1px solid #373a4d;
        border-radius: 13px;
    }
    .stButton > button {
        width: 100%;
        min-height: 3.1rem;
        border: 0;
        border-radius: 13px;
        color: #fff;
        font-weight: 800;
        background: linear-gradient(100deg, var(--pink), var(--orange));
        transition: filter .15s ease, transform .15s ease;
    }
    .stButton > button:hover {
        color: #fff;
        filter: brightness(1.07);
        transform: translateY(-1px);
    }
    .result-heading {
        font-size: 1.65rem;
        font-weight: 800;
        color: #fff;
        letter-spacing: -.04em;
        margin: .1rem 0 .2rem;
    }
    .result-subtitle { color: var(--muted); margin-bottom: 1.1rem; }
    .movie-card {
        min-height: 0;
        padding: 1.1rem;
        margin-bottom: .8rem;
        border: 1px solid var(--border);
        border-radius: 18px;
        background: linear-gradient(150deg, rgba(29,32,47,.98), rgba(18,20,30,.98));
        box-shadow: 0 10px 25px rgba(0,0,0,.12);
        overflow-wrap: anywhere;
    }
    .movie-rank {
        color: #ff8ea8;
        font-size: .72rem;
        font-weight: 850;
        letter-spacing: .13em;
        text-transform: uppercase;
        margin-bottom: .55rem;
    }
    .movie-title {
        color: #fff;
        font-size: 1.08rem;
        line-height: 1.4;
        font-weight: 800;
        margin-bottom: .65rem;
    }
    .movie-detail {
        color: #c4c7d5;
        font-size: .86rem;
        line-height: 1.55;
    }
    .movie-title {
        text-transform: uppercase;
    }
    .movie-tag {
        display: inline-block;
        border: 1px solid rgba(255,117,151,.24);
        background: rgba(255,92,130,.10);
        color: #ffb5c5;
        padding: .22rem .55rem;
        border-radius: 999px;
        font-size: .72rem;
        margin: .15rem .2rem .1rem 0;
    }
    .info-strip {
        border: 1px solid var(--border);
        border-radius: 15px;
        background: rgba(255,255,255,.035);
        padding: .9rem 1rem;
        color: #c6c8d6;
        font-size: .9rem;
    }
    .footer {
        color: #777d91;
        text-align: center;
        font-size: .8rem;
        padding-top: 1.6rem;
    }
    #MainMenu, footer { visibility: hidden; }

    @media (max-width: 640px) {
        [data-testid="stMainBlockContainer"] {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .hero { border-radius: 20px; }
        .movie-card { min-height: 0; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------- Load NLP recommendation data -----------------------------
@st.cache_resource(show_spinner="Loading your movie library...")
def load_data():
    if not MOVIES_PATH.exists():
        raise FileNotFoundError("movies.pickle is missing. Put it in the same folder as app.py.")
    if not SIMILARITY_PATH.exists():
        raise FileNotFoundError("similarity.joblib is missing. Put it in the same folder as app.py.")

    # Load only pickle/joblib files that you created or trust.
    with MOVIES_PATH.open("rb") as file:
        movie_data = pickle.load(file)
    similarity_data = joblib.load(SIMILARITY_PATH)

    if not isinstance(movie_data, pd.DataFrame):
        raise TypeError("movies.pickle must contain a pandas DataFrame.")
    if "title" not in movie_data.columns:
        raise ValueError("The movie DataFrame must contain a 'title' column.")
    if len(movie_data) != len(similarity_data):
        raise ValueError(
            "The movie list and similarity matrix have different sizes. "
            "Check that both files were generated from the same dataset and row order."
        )

    movie_data = movie_data.reset_index(drop=True).copy()
    movie_data["title"] = movie_data["title"].fillna("").astype(str)

    # Keep every row in its original order so movie indexes still align with
    # the precomputed similarity matrix. Blank titles are excluded from search below.
    # Do not change the NLP model or recalculate its similarity matrix here.
    return movie_data, similarity_data


def optional_text(row, column_names, default=""):
    """Get optional metadata only when that column exists in the user's dataset."""
    for column in column_names:
        if column in row.index:
            value = row[column]
            if pd.notna(value) and str(value).strip() and str(value).lower() != "nan":
                return str(value).strip()
    return default


def recommend(movie_name, movie_data, similarity_data, count=5):
    """Use the existing precomputed NLP similarity matrix to find similar titles."""
    matches = movie_data.index[movie_data["title"] == movie_name].tolist()
    if not matches:
        return []

    movie_index = matches[0]
    scores = similarity_data[movie_index]
    ranked_movies = sorted(
        (
            (index, float(score))
            for index, score in enumerate(scores)
            if index != movie_index
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:count]

    return [movie_data.iloc[index] for index, _score in ranked_movies]


# ----------------------------- TMDB poster lookup -----------------------------
def get_tmdb_api_key():
    """Read the TMDB key from Streamlit secrets or an environment variable."""
    try:
        secret_key = st.secrets.get("TMDB_API_KEY", "")
    except Exception:
        # Streamlit raises an exception if no secrets file is configured.
        secret_key = ""

    return str(secret_key or os.environ.get("TMDB_API_KEY", "")).strip()


@st.cache_data(ttl=86400, show_spinner=False)
def get_tmdb_poster(movie_title, api_key):
    """
    Search TMDB by title and return a poster image URL.
    Caches results for one day to reduce repeated API calls.
    """
    if not api_key or not movie_title:
        return None

    search_url = "https://api.themoviedb.org/3/search/movie"
    params = urlencode({
        "api_key": api_key,
        "query": movie_title,
        "include_adult": "false",
    })
    request = Request(
        f"{search_url}?{params}",
        headers={"User-Agent": "CineMatchMovieRecommender/1.0"},
    )

    try:
        with urlopen(request, timeout=8) as response:
            import json
            payload = json.loads(response.read().decode("utf-8"))

        results = payload.get("results", [])
        if not results:
            return None

        # Prefer an exact title match when possible to avoid choosing a similarly
        # named film. Fall back to TMDB's first search result otherwise.
        def normalize_title(value):
            return " ".join(str(value or "").casefold().strip().split())

        wanted = normalize_title(movie_title)
        exact_match = next(
            (
                item for item in results
                if normalize_title(item.get("title")) == wanted
                or normalize_title(item.get("original_title")) == wanted
            ),
            None,
        )
        match = exact_match or results[0]
        poster_path = match.get("poster_path")

        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"

    except (HTTPError, URLError, TimeoutError, ValueError, OSError):
        # Poster lookup should never prevent the recommendation system from working.
        return None

    return None


# ----------------------------- Website header -----------------------------
st.markdown(
    '<div class="brand">🎬 CINE<span>MATCH</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">A little inspiration for your next movie night</div>
        <div class="hero-title">Love a movie.<br>Find your <span class="accent">next favourite.</span></div>
        <p class="hero-copy">Pick a movie you like. Discover five similar movies.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

try:
    movies, similarities = load_data()
except Exception as error:
    st.error(f"Could not load the recommendation files: {error}")
    st.info(
        "Keep app.py, movies.pickle, and similarity.joblib together in the same project folder. "
        "If the error mentions pandas or a data type, run the app with the same compatible "
        "Python and package versions used when the pickle file was created."
    )
    st.stop()

movie_titles = sorted(
    movies.loc[movies["title"].str.strip() != "", "title"].drop_duplicates().tolist(),
    key=str.casefold,
)
if not movie_titles:
    st.warning("No movie titles were found in your dataset.")
    st.stop()

tmdb_api_key = get_tmdb_api_key()
if not tmdb_api_key:
    st.info(
        "Movie recommendations work without a TMDB key. To show posters, add "
        "TMDB_API_KEY to .streamlit/secrets.toml as explained below."
    )

st.markdown('<div class="section-kicker">01 / Choose a movie</div>', unsafe_allow_html=True)
search_col, button_col = st.columns([3.2, 1.2], vertical_alignment="bottom")

with search_col:
    selected_movie = st.selectbox(
        "Search your movie library",
        options=movie_titles,
        index=None,
        placeholder="Start typing a movie title...",
        help="Type part of a title to find it quickly.",
    )

with button_col:
    recommend_clicked = st.button("✨ Recommend movies", use_container_width=True)

st.markdown(
    f'<div class="helper">🍿 Your library contains <b>{len(movie_titles):,}</b> movie titles.</div>',
    unsafe_allow_html=True,
)

if recommend_clicked:
    if not selected_movie:
        st.warning("Please choose a movie before asking for recommendations.")
    else:
        st.session_state["cine_match_selected_movie"] = selected_movie

if "cine_match_selected_movie" in st.session_state:
    selected_title = st.session_state["cine_match_selected_movie"]
    st.markdown("---")
    st.markdown('<div class="section-kicker">02 / Made for your taste</div>', unsafe_allow_html=True)
    display_selected_title = (
        str(selected_title).upper()
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    st.markdown(
        f'<div class="result-heading">Because you liked {display_selected_title}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="result-subtitle">Here are five suggestions from your existing NLP similarity model.</div>',
        unsafe_allow_html=True,
    )

    recommendations = recommend(selected_title, movies, similarities, count=5)

    if not recommendations:
        st.warning("No recommendations were found for that title. Try choosing another movie.")
    else:
        # Use a two-column card layout for readability on desktop and mobile.
        for start in range(0, len(recommendations), 2):
            row_cols = st.columns(2, gap="medium")
            for offset, col in enumerate(row_cols):
                index = start + offset
                if index >= len(recommendations):
                    break

                movie_row = recommendations[index]
                title = optional_text(movie_row, ["title"], "Untitled movie").upper()
                year = optional_text(movie_row, ["release_year", "year", "release_date"])
                if len(year) >= 4 and year[:4].isdigit():
                    year = year[:4]
                # Keep the results minimal; no long descriptions are shown.
                # NLP datasets often use a combined text/tags column. Keep it optional
                # and avoid exposing long raw tags by default.
                with col:
                    poster_url = get_tmdb_poster(title, tmdb_api_key) if tmdb_api_key else None

                    # Center each poster and explicitly limit its width.
                    # This prevents Streamlit from stretching portrait posters
                    # to the full width of the recommendation column.
                    poster_left, poster_center, poster_right = st.columns(
                        [1, 1, 1], vertical_alignment="top"
                    )
                    with poster_center:
                        if poster_url:
                            st.image(poster_url, width=160)
                        else:
                            st.markdown(
                                """
                                <div style="
                                    width:160px; height:240px; box-sizing:border-box;
                                    display:flex; flex-direction:column;
                                    align-items:center; justify-content:center; gap:.55rem;
                                    text-align:center; border:1px solid rgba(255,255,255,.10);
                                    border-radius:12px; margin:0 auto .65rem;
                                    background:linear-gradient(145deg,#202335,#12141d);
                                    color:#969bad; padding:.8rem;">
                                    <span style="font-size:1.5rem">🎬</span>
                                    <span style="font-size:.82rem">Poster unavailable</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    year = optional_text(movie_row, ["release_year", "year", "release_date"])
                    if len(year) >= 4 and year[:4].isdigit():
                        year = year[:4]

                    safe_title = (
                        title.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                    )
                    safe_year = (
                        year.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                    )

                    year_html = (
                        f'<div class="movie-detail">{safe_year}</div>'
                        if safe_year else ""
                    )
                    st.markdown(
                        f"""
                        <div class="movie-card" style="min-height:0; margin-top:.2rem;">
                            <div class="movie-rank">Recommendation {index + 1:02d}</div>
                            <div class="movie-title">{safe_title}</div>
                            {year_html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        with st.expander("How does CineMatch choose these movies?"):
            st.write(
                "Your NLP model uses the similarity matrix saved in similarity.joblib to rank "
                "the top five similar titles. TMDB is used only to look up poster images; it "
                "does not generate or change your recommendations."
            )

st.markdown(
    """
    <div class="info-strip">
        <b>Your NLP model makes the recommendations.</b>
        TMDB is used only to fetch movie posters. If a poster cannot be found or TMDB is
        unavailable, the recommendation system will still work.
    </div>
    <div class="footer">
        CineMatch · Built with Streamlit and your NLP recommendation model<br>
        This product uses the TMDB API but is not endorsed or certified by TMDB.
        <br>Movie poster images are provided by TMDB.
    </div>
    """,
    unsafe_allow_html=True,
)

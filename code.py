import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import os

# ─── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main { background-color: #0d0d0d; }
    .block-container { padding: 2rem 3rem; }

    h1 { font-family: 'Bebas Neue', sans-serif; font-size: 4rem !important;
         background: linear-gradient(90deg, #e50914, #ff6b35);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

    .movie-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #2a2a4a;
        transition: transform 0.2s;
        color: #fff;
    }
    .movie-card:hover { transform: translateY(-4px); border-color: #e50914; }
    .movie-title { font-size: 1.1rem; font-weight: 600; color: #ffffff; margin-bottom: 6px; }
    .movie-meta { font-size: 0.82rem; color: #aaaacc; }
    .score-badge {
        background: linear-gradient(90deg, #e50914, #ff6b35);
        color: white; border-radius: 20px;
        padding: 3px 12px; font-size: 0.78rem; font-weight: 600;
        display: inline-block; margin-top: 8px;
    }
    .genre-tag {
        background: #2a2a4a; color: #9090cc;
        border-radius: 12px; padding: 2px 10px;
        font-size: 0.75rem; display: inline-block;
        margin: 2px 3px 0 0;
    }
    .stSelectbox > div > div { background-color: #1a1a2e; color: #fff; }
    .stButton > button {
        background: linear-gradient(90deg, #e50914, #ff6b35);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; padding: 10px 28px; font-size: 1rem;
        width: 100%;
    }
    .stButton > button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load and preprocess movie data."""
    # Use bundled sample data if TMDB files aren't present
    credits_path = "data/tmdb_5000_credits.csv"
    movies_path  = "data/tmdb_5000_movies.csv"

    if os.path.exists(credits_path) and os.path.exists(movies_path):
        credits = pd.read_csv(credits_path)
        movies  = pd.read_csv(movies_path)
        movies  = movies.merge(credits, on="title")
        movies  = movies[["movie_id", "title", "overview", "genres",
                           "keywords", "cast", "crew", "vote_average",
                           "vote_count", "release_date"]].dropna()
    else:
        movies = make_sample_data()

    return movies


def make_sample_data():
    """Built-in sample dataset so the app works without downloads."""
    data = {
        "movie_id": list(range(1, 31)),
        "title": [
            "The Dark Knight", "Inception", "Interstellar", "The Matrix",
            "Avengers: Endgame", "Pulp Fiction", "The Shawshank Redemption",
            "The Godfather", "Forrest Gump", "The Silence of the Lambs",
            "Goodfellas", "Fight Club", "Schindler's List", "The Lord of the Rings",
            "Star Wars: A New Hope", "Jurassic Park", "Titanic", "The Lion King",
            "Toy Story", "Finding Nemo", "Up", "WALL-E", "Coco",
            "Spider-Man: Into the Spider-Verse", "Black Panther",
            "Iron Man", "Thor: Ragnarok", "Doctor Strange",
            "Guardians of the Galaxy", "Captain America: Civil War"
        ],
        "overview": [
            "Batman raises the stakes in his war on crime with the Joker",
            "A thief who enters the dreams of others to steal secrets",
            "A team of explorers travel through a wormhole in space",
            "A computer hacker learns reality is a simulation",
            "The Avengers assemble to undo Thanos' devastating snap",
            "Several stories of Los Angeles criminals intertwine",
            "Two imprisoned men bond over years finding solace",
            "The aging patriarch of an organized crime dynasty",
            "The life story of Forrest Gump an Alabama man",
            "A young FBI cadet must receive help from Dr. Hannibal Lecter",
            "Henry Hill rises through the mob ranks in New York",
            "An insomniac office worker forms an underground fight club",
            "A German businessman saves Jewish refugees during the Holocaust",
            "A fellowship sets out on a quest to destroy a powerful ring",
            "Luke Skywalker joins rebels to battle the Galactic Empire",
            "A theme park with cloned dinosaurs descends into chaos",
            "A love story unfolds aboard the ill-fated RMS Titanic",
            "A young lion must overcome tragedy and reclaim his kingdom",
            "A cowboy doll is threatened by the arrival of Buzz Lightyear",
            "A father clownfish searches the ocean for his missing son",
            "An old widower and a boy scout travel to South America",
            "A robot falls in love while cleaning Earth alone",
            "A young musician must choose between his family and his passion",
            "Miles Morales becomes the Spider-Man of his dimension",
            "T'Challa returns to his African nation to defend it",
            "Tony Stark builds an armored suit to fight evil",
            "Thor must recruit Bruce Banner to stop his sister Hela",
            "A surgeon becomes the Master of the Mystic Arts",
            "A ragtag crew of misfit heroes save the galaxy",
            "Disagreements among the Avengers split the team apart"
        ],
        "genres": [
            "Action,Crime,Drama", "Action,Adventure,Sci-Fi",
            "Adventure,Drama,Sci-Fi", "Action,Sci-Fi",
            "Action,Adventure,Drama", "Crime,Drama", "Drama",
            "Crime,Drama", "Drama,Romance", "Crime,Drama,Thriller",
            "Biography,Crime,Drama", "Drama,Thriller", "Biography,Drama,History",
            "Adventure,Drama,Fantasy", "Action,Adventure,Fantasy",
            "Action,Adventure,Sci-Fi", "Drama,Romance", "Animation,Adventure,Drama",
            "Animation,Adventure,Comedy", "Animation,Adventure,Comedy",
            "Animation,Adventure,Comedy", "Animation,Family,Romance",
            "Animation,Adventure,Family", "Animation,Action,Adventure",
            "Action,Adventure,Sci-Fi", "Action,Adventure,Sci-Fi",
            "Action,Adventure,Sci-Fi", "Action,Adventure,Fantasy",
            "Action,Adventure,Comedy", "Action,Adventure,Sci-Fi"
        ],
        "keywords": [
            "dc,joker,batman,gotham", "dreams,heist,subconscious",
            "space,wormhole,time,love", "simulation,hacker,reality",
            "avengers,marvel,time travel", "nonlinear,hitman,drug",
            "prison,hope,friendship", "mafia,family,power",
            "disability,run,love,life", "fbi,serial killer,cannibal",
            "mafia,crime,drugs,new york", "insomnia,soap,underground",
            "holocaust,jew,germany", "ring,hobbit,magic,elf",
            "jedi,force,empire,rebel", "dinosaur,cloning,island",
            "ship,ocean,romance,class", "lion,africa,kingdom,pride",
            "toy,cowboy,space ranger", "ocean,clownfish,reef,shark",
            "balloon,adventure,old man", "robot,earth,pollution,love",
            "music,family,mexico,memory", "multiverse,spider,teen",
            "africa,king,wakanda,vibranium", "iron,suit,billionaire",
            "thunder,asgard,hammer,hela", "sorcerer,magic,dimension",
            "galaxy,groot,raccoon,star lord", "avengers,split,sokovia"
        ],
        "cast": [
            "Christian Bale,Heath Ledger,Aaron Eckhart",
            "Leonardo DiCaprio,Joseph Gordon-Levitt,Ellen Page",
            "Matthew McConaughey,Anne Hathaway,Jessica Chastain",
            "Keanu Reeves,Laurence Fishburne,Carrie-Anne Moss",
            "Robert Downey Jr.,Chris Evans,Chris Hemsworth",
            "John Travolta,Uma Thurman,Samuel L. Jackson",
            "Tim Robbins,Morgan Freeman",
            "Marlon Brando,Al Pacino,James Caan",
            "Tom Hanks,Robin Wright,Gary Sinise",
            "Jodie Foster,Anthony Hopkins",
            "Ray Liotta,Robert De Niro,Joe Pesci",
            "Brad Pitt,Edward Norton,Helena Bonham Carter",
            "Liam Neeson,Ben Kingsley,Ralph Fiennes",
            "Elijah Wood,Ian McKellen,Viggo Mortensen",
            "Mark Hamill,Harrison Ford,Carrie Fisher",
            "Sam Neill,Laura Dern,Jeff Goldblum",
            "Leonardo DiCaprio,Kate Winslet",
            "Matthew Broderick,Jeremy Irons,James Earl Jones",
            "Tom Hanks,Tim Allen",
            "Albert Brooks,Ellen DeGeneres,Alexander Gould",
            "Edward Asner,Christopher Plummer,Jordan Nagai",
            "Ben Burtt,Elissa Knight",
            "Anthony Gonzalez,Gael Garcia Bernal,Benjamin Bratt",
            "Shameik Moore,Jake Johnson,Hailee Steinfeld",
            "Chadwick Boseman,Michael B. Jordan,Lupita Nyong'o",
            "Robert Downey Jr.,Gwyneth Paltrow,Jeff Bridges",
            "Chris Hemsworth,Tom Hiddleston,Cate Blanchett",
            "Benedict Cumberbatch,Chiwetel Ejiofor,Rachel McAdams",
            "Chris Pratt,Zoe Saldana,Vin Diesel",
            "Chris Evans,Robert Downey Jr.,Scarlett Johansson"
        ],
        "crew": [
            "Christopher Nolan", "Christopher Nolan", "Christopher Nolan",
            "Lana Wachowski", "Anthony Russo", "Quentin Tarantino",
            "Frank Darabont", "Francis Ford Coppola", "Robert Zemeckis",
            "Jonathan Demme", "Martin Scorsese", "David Fincher",
            "Steven Spielberg", "Peter Jackson", "George Lucas",
            "Steven Spielberg", "James Cameron", "Roger Allers",
            "John Lasseter", "Andrew Stanton", "Pete Docter",
            "Andrew Stanton", "Lee Unkrich", "Peter Ramsey",
            "Ryan Coogler", "Jon Favreau", "Taika Waititi",
            "Scott Derrickson", "James Gunn", "Anthony Russo"
        ],
        "vote_average": [
            9.0, 8.8, 8.6, 8.7, 8.4, 8.9, 9.3, 9.2, 8.8, 8.6,
            8.7, 8.8, 9.0, 8.9, 8.6, 8.1, 7.8, 8.5, 8.3, 8.2,
            8.3, 8.4, 8.4, 8.4, 7.3, 7.9, 7.9, 7.5, 8.0, 7.8
        ],
        "vote_count": [
            27000, 22000, 18000, 21000, 19000, 20000, 23000, 18000,
            22000, 14000, 12000, 21000, 14000, 18000, 17000, 14000,
            20000, 16000, 15000, 15000, 14000, 13000, 12000, 11000,
            16000, 19000, 15000, 12000, 17000, 16000
        ],
        "release_date": [
            "2008", "2010", "2014", "1999", "2019", "1994", "1994",
            "1972", "1994", "1991", "1990", "1999", "1993", "2001",
            "1977", "1993", "1997", "1994", "1995", "2003", "2009",
            "2008", "2017", "2018", "2018", "2008", "2017", "2016",
            "2014", "2016"
        ]
    }
    return pd.DataFrame(data)


# ─── Feature Engineering ─────────────────────────────────────────
@st.cache_data
def build_features(movies: pd.DataFrame):
    df = movies.copy()

    def safe_parse(val):
        if isinstance(val, list): return val
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return [i.get("name", "") for i in parsed if isinstance(i, dict)]
        except Exception:
            pass
        if isinstance(val, str):
            return [v.strip() for v in val.split(",")]
        return []

    for col in ["genres", "keywords", "cast", "crew"]:
        if col in df.columns:
            df[col] = df[col].apply(safe_parse)

    def make_soup(row):
        parts = []
        if isinstance(row.get("overview"), str):
            parts.append(row["overview"])
        for col in ["genres", "keywords", "cast", "crew"]:
            if isinstance(row.get(col), list):
                parts.extend(row[col])
        return " ".join(parts).lower()

    df["soup"] = df.apply(make_soup, axis=1)

    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = tfidf.fit_transform(df["soup"])
    sim    = cosine_similarity(matrix, matrix)

    indices = pd.Series(df.index, index=df["title"].str.lower()).drop_duplicates()
    return df, sim, indices


# ─── Recommendation Logic ────────────────────────────────────────
def recommend(title: str, df, sim, indices, n=8):
    key = title.lower()
    if key not in indices:
        return pd.DataFrame()
    idx   = indices[key]
    scores = list(enumerate(sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    idxs  = [i[0] for i in scores]
    result = df.iloc[idxs][["title", "genres", "cast", "crew",
                              "vote_average", "vote_count",
                              "release_date", "overview"]].copy()
    result["similarity"] = [round(s[1]*100, 1) for s in scores]
    return result


def genre_filter(df, genre):
    if genre == "All":
        return df
    def has_genre(g):
        if isinstance(g, list):
            return any(genre.lower() in x.lower() for x in g)
        return genre.lower() in str(g).lower()
    return df[df["genres"].apply(has_genre)]


# ─── Main UI ─────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("# 🎬 MOVIE RECOMMENDER")
    st.markdown("*Discover your next favourite film using AI-powered similarity matching*")
    st.markdown("---")

    movies, sim, indices = build_features(load_data())
    all_titles = sorted(movies["title"].tolist())

    # ── Sidebar ────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Filters")
        n_recs = st.slider("Number of recommendations", 4, 16, 8)
        all_genres = sorted({
            g for gs in movies["genres"]
            for g in (gs if isinstance(gs, list) else str(gs).split(","))
            if g.strip()
        })
        genre = st.selectbox("Filter by genre", ["All"] + all_genres)
        st.markdown("---")
        st.markdown("### 📊 Dataset Stats")
        st.metric("Total Movies", len(movies))
        st.metric("Genres Available", len(all_genres))

    # ── Search ─────────────────────────────────────────────────
    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.selectbox("🎥 Choose a movie you like", all_titles,
                                index=all_titles.index("The Dark Knight")
                                if "The Dark Knight" in all_titles else 0)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        find_btn = st.button("🔍 Find Similar")

    # ── Show selected movie info ────────────────────────────────
    sel_row = movies[movies["title"] == selected].iloc[0]
    genres_disp = sel_row["genres"]
    if isinstance(genres_disp, list):
        genres_disp = " · ".join(genres_disp)

    st.markdown(f"""
    <div class="movie-card">
        <div class="movie-title">📽 {sel_row['title']} ({sel_row.get('release_date', 'N/A')})</div>
        <div class="movie-meta">{sel_row.get('overview','')[:200]}...</div>
        <div style="margin-top:8px">{''.join(f'<span class="genre-tag">{g.strip()}</span>' for g in (genres_disp.split(' · ') if isinstance(genres_disp, str) else genres_disp))}</div>
        <div class="score-badge">⭐ {sel_row.get('vote_average','N/A')}/10</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Recommendations ────────────────────────────────────────
    if find_btn or True:   # always show
        recs = recommend(selected, movies, sim, indices, n=n_recs)
        if genre != "All":
            recs = genre_filter(recs, genre)

        if recs.empty:
            st.warning("No recommendations found. Try a different movie or genre filter.")
            return

        st.markdown(f"### 🍿 Because you liked **{selected}**...")
        cols = st.columns(2)
        for i, (_, row) in enumerate(recs.iterrows()):
            with cols[i % 2]:
                g = row["genres"]
                if isinstance(g, list):
                    g = " · ".join(g)
                tags = "".join(f'<span class="genre-tag">{x.strip()}</span>'
                               for x in str(g).split(" · ")[:4] if x.strip())
                cast_str = row.get("cast", "")
                if isinstance(cast_str, list):
                    cast_str = ", ".join(cast_str[:3])

                st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-title">🎬 {row['title']} ({row.get('release_date','N/A')})</div>
                    <div class="movie-meta">{str(row.get('overview',''))[:140]}...</div>
                    <div class="movie-meta" style="margin-top:6px">👤 {cast_str[:60]}</div>
                    <div style="margin-top:6px">{tags}</div>
                    <div style="display:flex;gap:8px;margin-top:8px">
                        <div class="score-badge">⭐ {row.get('vote_average','?')}/10</div>
                        <div class="score-badge">🎯 {row['similarity']}% match</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Top Rated section ──────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏆 Top Rated Movies")
    top = movies.nlargest(6, "vote_average")[["title", "vote_average",
                                               "genres", "release_date"]]
    cols3 = st.columns(3)
    for i, (_, row) in enumerate(top.iterrows()):
        with cols3[i % 3]:
            g = row["genres"]
            if isinstance(g, list): g = " · ".join(g[:2])
            st.markdown(f"""
            <div class="movie-card">
                <div class="movie-title">{row['title']}</div>
                <div class="movie-meta">{g} · {row.get('release_date','')}</div>
                <div class="score-badge">⭐ {row['vote_average']}/10</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<center><small>Built with ❤️ using Streamlit & Scikit-learn</small></center>",
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
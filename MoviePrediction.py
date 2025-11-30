# MoviePrediction_fixed.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
DATA_PATH = "indian_movies.csv"   # change if your file has different name
RANDOM_STATE = 42
# ----------------------------

# Common alternative names mapped to canonical names
COLUMN_ALIASES = {
    "title": ["title", "movie_title", "name", "Series_Title", "Name", "Title"],
    "genre": ["genre", "genres", "Genre", "Genres"],
    "year": ["year", "released_year", "release_year", "Released_Year", "Year", "date_published"],
    "duration": ["duration", "runtime", "time", "length", "Duration", "Runtime"],
    "rating": ["rating", "imdb_rating", "IMDB_RATING", "IMDB_Rating", "IMDB_Rating", "avg_vote", "IMDB_Rating", "imdbRating"],
    "votes": ["votes", "num_votes", "no_of_votes", "No_of_Votes", "votes_count"]
}

def load_df(path):
    # try multiple encodings
    for enc in ("utf-8", "latin1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"Loaded file with encoding: {enc}")
            return df
        except Exception as e:
            last_exc = e
    raise last_exc

def find_column(df, aliases):
    for a in aliases:
        if a in df.columns:
            return a
    return None

def auto_map_columns(df):
    # returns a dict canonical_name -> actual column name (or None)
    mapping = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        found = find_column(df, aliases)
        mapping[canonical] = found
    return mapping

def make_canonical_df(df, mapping):
    # create new df with canonical column names; if missing, create sensible defaults
    df2 = df.copy()
    # Title
    if mapping["title"]:
        df2["title"] = df2[mapping["title"]].astype(str)
    else:
        df2["title"] = "Unknown"
    # Genre
    if mapping["genre"]:
        df2["genre"] = df2[mapping["genre"]].astype(str)
    else:
        df2["genre"] = ""
    # Year -> numeric
    if mapping["year"]:
        df2["year"] = pd.to_numeric(df2[mapping["year"]], errors="coerce")
    else:
        df2["year"] = np.nan
    # Duration -> numeric minutes
    if mapping["duration"]:
        # attempt to extract numbers (e.g. "2h 30min" or "150" or "150 min")
        s = df2[mapping["duration"]].astype(str)
        # extract first number
        mins = s.str.extract(r'(\d{2,4})')
        df2["duration"] = pd.to_numeric(mins[0], errors="coerce")
    else:
        df2["duration"] = np.nan
    # Rating -> numeric target
    if mapping["rating"]:
        df2["rating"] = pd.to_numeric(df2[mapping["rating"]].astype(str).str.replace('/10','',regex=False), errors="coerce")
    else:
        df2["rating"] = np.nan
    # Votes -> numeric
    if mapping["votes"]:
        df2["votes"] = pd.to_numeric(df2[mapping["votes"]], errors="coerce")
    else:
        df2["votes"] = np.nan
    return df2

def preprocess_df(df):
    # Print columns & top rows for visibility
    print("\nDetected columns in file:")
    print(list(df.columns))
    print("\nFirst 5 rows (preview):")
    with pd.option_context('display.max_columns', None):
        print(df.head(5))

    mapping = auto_map_columns(df)
    print("\nAuto-mapping (canonical -> found column):")
    for k,v in mapping.items():
        print(f"  {k:8} -> {v}")

    dfc = make_canonical_df(df, mapping)

    # Basic cleaning
    # If no rating column at all, exit with friendly message
    if dfc["rating"].isna().all():
        print("\nERROR: No rating values detected in file. The script expects an IMDb rating column (e.g. 'rating', 'IMDB_Rating', 'avg_vote').")
        print("Please check your CSV or provide a file that contains movie ratings.")
        raise SystemExit(1)

    # Drop rows without ratings
    dfc = dfc.dropna(subset=["rating"]).reset_index(drop=True)

    # Fill numeric NaNs with medians or defaults
    for col in ["duration","year","votes"]:
        if dfc[col].isna().any():
            median = dfc[col].median()
            if np.isnan(median):
                # if a column has no values at all, fill with 0
                median = 0
            dfc[col] = dfc[col].fillna(median)

    # Simplify genre: take the first genre if pipe/comma separated
    dfc["genre_first"] = dfc["genre"].fillna("").astype(str).apply(lambda s: s.replace("|",",").split(",")[0].strip() if s.strip()!="" else "Unknown")

    return dfc

def encode_and_train(dfc):
    # Features: genre_first (label-encoded), title (label-encoded) optionally, numeric features
    FEATURES = ["title_enc", "genre_enc", "duration", "votes", "year"]
    encoders = {}

    # Label encode title and genre_first
    le_title = LabelEncoder()
    dfc["title_enc"] = le_title.fit_transform(dfc["title"].astype(str))
    encoders["title"] = le_title

    le_genre = LabelEncoder()
    dfc["genre_enc"] = le_genre.fit_transform(dfc["genre_first"].astype(str))
    encoders["genre"] = le_genre

    X = dfc[FEATURES]
    y = dfc["rating"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)

    model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    print(f"\nTrained RandomForest: test RMSE = {rmse:.4f}")
    return model, encoders, X_train.columns.tolist()

def predict_interactive(model, encoders, feature_cols):
    # Ask user if they'd like to predict sample movie
    ans = input("\nDo you want to predict rating for a new movie now? (y/n): ").strip().lower()
    if ans not in ("y","yes"):
        print("Exiting. You can re-run script anytime.")
        return
    title = input("Movie Title: ").strip()
    genre = input("Genre (e.g. Drama, Action): ").strip()
    try:
        duration = float(input("Duration (minutes): ").strip())
    except:
        duration = 0.0
    try:
        votes = float(input("Votes (numeric): ").strip())
    except:
        votes = 0.0
    try:
        year = float(input("Year (e.g. 2022): ").strip())
    except:
        year = 0.0

    # encode
    if title in encoders["title"].classes_:
        t_enc = encoders["title"].transform([title])[0]
    else:
        # unseen title -> use mode or 0
        t_enc = 0

    if genre in encoders["genre"].classes_:
        g_enc = encoders["genre"].transform([genre])[0]
    else:
        g_enc = 0

    sample = pd.DataFrame([[t_enc, g_enc, duration, votes, year]], columns=feature_cols)
    pred = model.predict(sample)[0]
    print(f"\nPredicted IMDb Rating: {pred:.2f}")

def main():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at: {DATA_PATH}")
        print("Place your Kaggle CSV in the same folder and name it:", DATA_PATH)
        return

    df = load_df(DATA_PATH)
    dfc = preprocess_df(df)
    model, encoders, feature_cols = encode_and_train(dfc)
    predict_interactive(model, encoders, feature_cols)

if __name__ == "__main__":
    main()

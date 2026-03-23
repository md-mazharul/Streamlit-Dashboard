# 🎬 CineScope — Movie Ratings Explorer

An interactive Streamlit dashboard for exploring movie ratings across age groups, genres, and decades. Built on the classic MovieLens dataset, CineScope lets you slice and filter 200,000+ ratings to uncover how different audiences experience film.

---

## 📸 Overview

CineScope provides four interactive views:

| Tab | Description |
|-----|-------------|
| **Overview** | Age distribution of raters and overall rating histogram |
| **🎭 Genre** | Ratings volume and average scores broken down by genre |
| **🔥 Heatmap** | Genre × Age group heatmap (average rating or count) |
| **📅 Year** | Ratings and average scores by release year |

---

## 🗂️ Project Structure

```
.
├── main.py               # Streamlit app
├── movie_ratings.csv     # Dataset (MovieLens-based)
└── requirements.txt      # Python dependencies
```

---

## 📊 Dataset

The dataset (`movie_ratings.csv`) contains ~212,000 rows with the following columns:

| Column | Description |
|--------|-------------|
| `user_id` | Unique user identifier |
| `movie_id` | Unique movie identifier |
| `rating` | Rating score (1–5) |
| `timestamp` | Date and time of the rating |
| `age` | Age of the user |
| `gender` | Gender of the user (M/F) |
| `occupation` | User's occupation |
| `zip_code` | User's zip code |
| `title` | Movie title and release year |
| `year` | Movie release year |
| `decade` | Release decade (e.g., 1990.0) |
| `genres` | Pipe-separated genre tags (e.g., `Action\|Thriller`) |
| `rating_year` | Year the rating was submitted |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/md-mazharul/Streamlit-Dashboard.git
cd Streamlit-Dashboard
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`.

> **Note:** Make sure `movie_ratings.csv` is in the same directory as `main.py`.

---

## 🧰 Dependencies

```
streamlit
pandas
numpy
plotly
```

Install all at once with:

```bash
pip install streamlit pandas numpy plotly
```

---

## 🎛️ Sidebar Controls

- **Age bins** — Customize the age group boundaries (comma-separated integers)
- **Heatmap metric** — Toggle between *Average rating* and *Number of ratings*
- **Filter by genre** — Select one or more genres to filter all views simultaneously
- **Filter by year** — Drag a range slider to focus on specific release years

---

## ✨ Features

- **Auto column detection** — The app intelligently detects age, rating, genre, and year columns regardless of exact naming
- **Multi-genre support** — Pipe-separated genres (e.g., `Action|Comedy`) are automatically expanded
- **Downloadable data** — Export the heatmap pivot table as a CSV directly from the dashboard
- **Responsive layout** — Built with Streamlit's wide layout and multi-column support

---
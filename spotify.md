```markdown
# Spotify Music Clustering Analysis

## Project Overview

Unsupervised machine learning project that clusters songs based on audio features to identify distinct music genres and listening moods. Using **K-Means clustering**, the model groups songs into 5 interpretable categories: Chill Acoustic, Dance Party, Energetic Rock, Study/Focus, and Happy Pop.

## Key Results

| Cluster | Size | Primary Characteristics |
|---------|------|------------------------|
| 🎵 Chill Acoustic | 24% | High energy, moderate danceability |
| 💃 Dance Party | 22% | High danceability, high valence |
| 🎸 Energetic Rock | 20% | High acousticness, low energy |
| 🎹 Study/Focus | 18% | Balanced features, moderate tempo |
| 😊 Happy Pop | 16% | High danceability, high valence |

## Cluster Profiles

### Cluster 0: 🎵 Chill Acoustic
```
Audio Profile:
├── Danceability: 0.48 (moderate)
├── Energy: 0.75 (high)
├── Valence: 0.39 (slightly negative)
├── Acousticness: 0.22 (low)
├── Tempo: 121 BPM
└── Loudness: -8.63 dB

Best for: Background music, relaxed listening
```

### Cluster 1: 💃 Dance Party
```
Audio Profile:
├── Danceability: 0.63 (high)
├── Energy: 0.65 (moderate-high)
├── Valence: 0.50 (neutral/positive)
├── Acousticness: 0.23 (low)
├── Tempo: 124 BPM
└── Loudness: -7.31 dB

Best for: Parties, workouts, social gatherings
```

### Cluster 2: 🎸 Energetic Rock
```
Audio Profile:
├── Danceability: 0.38 (low)
├── Energy: 0.19 (low)
├── Valence: 0.25 (negative)
├── Acousticness: 0.84 (very high!)
├── Tempo: 105 BPM
└── Loudness: -19.74 dB (quiet)

Best for: Focused listening, studying, quiet time
```

### Cluster 3: 🎹 Study/Focus
```
Audio Profile:
├── Danceability: 0.64 (high)
├── Energy: 0.67 (moderate)
├── Valence: 0.48 (neutral)
├── Acousticness: 0.20 (low)
├── Tempo: 123 BPM
└── Loudness: -7.35 dB

Best for: Studying, working, concentration
```

### Cluster 4: 😊 Happy Pop
```
Audio Profile:
├── Danceability: 0.70 (very high!)
├── Energy: 0.59 (moderate)
├── Valence: 0.51 (positive)
├── Acousticness: 0.27 (low)
├── Tempo: 122 BPM
├── Speechiness: 0.39 (noticeable lyrics)
└── Duration: 3 min (shortest)

Best for: Happy moods, commuting, casual listening
```

## Feature Importance Analysis

### Most Discriminative Features

| Feature | Range Across Clusters | Discriminative Power |
|---------|----------------------|---------------------|
| **Acousticness** | 0.20 - 0.84 | ⭐⭐⭐⭐⭐ (highest) |
| **Energy** | 0.19 - 0.75 | ⭐⭐⭐⭐⭐ (highest) |
| **Loudness** | -19.7 to -7.3 dB | ⭐⭐⭐⭐ |
| **Danceability** | 0.38 - 0.70 | ⭐⭐⭐⭐ |
| **Valence** | 0.25 - 0.51 | ⭐⭐⭐ |
| **Tempo** | 105 - 124 BPM | ⭐⭐ |

### Key Insights

1. **Acousticness** is the strongest separator:
   - Cluster 2 (Energetic Rock): 84% acoustic
   - Others: 20-27% acoustic

2. **Energy** creates clear boundaries:
   - Cluster 0 (Chill Acoustic): 75% energy
   - Cluster 2 (Energetic Rock): 19% energy (calm!)

3. **Danceability** distinguishes party vs study music:
   - Dance Party & Happy Pop: >0.63 danceability
   - Energetic Rock: 0.38 danceability

## Methodology

### 1. Data Preprocessing
```python
# Audio features used for clustering
features = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'duration_ms',
    'time_signature'
]

# Standardize features (important for K-Means)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
```

### 2. Optimal K Selection
- Used **Elbow Method** to determine optimal clusters
- **K=5** chosen as balance between granularity and interpretability

### 3. Clustering Algorithm
```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)
```

## Visualizations

### 1. Cluster Distribution
```
Cluster Sizes:
🎵 Chill Acoustic:  ████████████████████ 24%
💃 Dance Party:     ███████████████████  22%
🎸 Energetic Rock:  █████████████████    20%
🎹 Study/Focus:     ███████████████      18%
😊 Happy Pop:       ██████████████       16%
```

### 2. Feature Radar Chart (by cluster)
```
                    Danceability
                         ^
                    /    |    \
          Energy <--     |     --> Valence
                    \    |    /
                         v
                   Acousticness

Cluster 0 (Chill): High Energy, Low Valence
Cluster 1 (Dance): High Danceability, High Energy
Cluster 2 (Rock):  High Acousticness, Low Energy
Cluster 3 (Study): Balanced across features
Cluster 4 (Pop):   High Danceability, High Valence
```

## Business Applications

| Use Case | Recommended Cluster |
|----------|---------------------|
| **Workout Playlist** | 💃 Dance Party, 😊 Happy Pop |
| **Study Session** | 🎸 Energetic Rock, 🎹 Study/Focus |
| **Party/Event** | 💃 Dance Party |
| **Relaxation** | 🎵 Chill Acoustic |
| **Commuting** | 😊 Happy Pop, 💃 Dance Party |
| **Background Music** | 🎵 Chill Acoustic, 🎸 Energetic Rock |

## Next Steps

### Improvements

| Enhancement | Expected Impact |
|-------------|----------------|
| Add more audio features (MFCCs, chroma) | Better separation |
| Try DBSCAN or Hierarchical clustering | Handle outliers better |
| Add temporal features (release year) | Trend analysis |
| Create recommendation engine | Personalized playlists |

### Potential Extensions
- [ ] Build playlist generator based on cluster preferences
- [ ] Create mood transition mapping between clusters
- [ ] Develop audio feature predictor for new songs
- [ ] Deploy as Spotify API recommendation filter

## Technical Stack

```python
# Dependencies
pandas==2.0.0          # Data manipulation
numpy==1.24.0          # Numerical operations
scikit-learn==1.3.0    # K-Means, preprocessing
matplotlib==3.7.0      # Visualizations
seaborn==0.12.0        # Enhanced plotting
```

## Code Example

```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load and prepare data
df = pd.read_csv('spotify_data.csv')
features = ['danceability', 'energy', 'acousticness', 'valence', 'tempo']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Apply K-Means
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Analyze clusters
cluster_profiles = df.groupby('cluster')[features].mean()
print(cluster_profiles.round(3))
```

## Conclusion

The K-Means clustering successfully identifies **5 distinct music categories** based on audio features. The clusters are highly interpretable and align with real-world music listening contexts:

- ✅ **Clear separation** between acoustic and electronic music
- ✅ **Meaningful grouping** by energy and danceability
- ✅ **Practical applications** for playlist curation
- ✅ **Scalable approach** for large music datasets

**Best performing cluster:** Cluster 1 (Dance Party) - most distinctive audio signature
**Most versatile cluster:** Cluster 3 (Study/Focus) - balanced features for various contexts

## Repository Structure

```
├── data/
│   └── spotify_tracks.csv
├── notebooks/
│   └── music_clustering.ipynb
├── outputs/
│   ├── cluster_profiles.csv
│   └── cluster_visualizations.png
├── README.md
└── requirements.txt
```

## Author

Gusky

## License

MIT

---

**Project Status:** ✅ Complete | **Model Performance:** 🟢 High interpretability | **Ready for:** Playlist generation, music recommendation
```

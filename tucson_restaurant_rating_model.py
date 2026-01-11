import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==============================
# Load Yelp business data
# ==============================
rows = []
with open("data/yelp_academic_dataset_business.json", "r") as f:
    for line in f:
        rows.append(json.loads(line))

df = pd.DataFrame(rows)
print("Total businesses:", df.shape)

# ==============================
# Filter Tucson restaurants
# ==============================
tucson = df[df["city"] == "Tucson"]

tucson_restaurants = tucson[
    tucson["categories"].str.contains("Restaurant", case=False, na=False)
]

print("Tucson restaurants:", tucson_restaurants.shape)

# ==============================
# Keep restaurants with hours
# ==============================
tucson_with_hours = tucson_restaurants[
    tucson_restaurants["hours"].notna()
].copy()

print("Restaurants with hours:", tucson_with_hours.shape)

# ==============================
# Feature engineering: avg daily hours
# ==============================
def avg_daily_hours(hours):
    if not isinstance(hours, dict):
        return None

    total = 0
    count = 0

    for _, time_range in hours.items():
        try:
            open_t, close_t = time_range.split("-")
            open_h = int(open_t.split(":")[0])
            close_h = int(close_t.split(":")[0])

            if close_h < open_h:
                close_h += 24  # overnight

            total += max(close_h - open_h, 0)
            count += 1
        except:
            continue

    return total / count if count > 0 else None

tucson_with_hours["avg_daily_hours"] = (
    tucson_with_hours["hours"].apply(avg_daily_hours)
)

# ==============================
# Prepare ML dataset
# ==============================
features = ["review_count", "is_open", "avg_daily_hours"]
target = "stars"

ml_df = tucson_with_hours[features + [target, "name"]].dropna()

X = ml_df[features]
y = ml_df[target]

print("Final ML dataset size:", X.shape)

# ==============================
# Train / Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# Train Model
# ==============================
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# Evaluate Model
# ==============================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("-----------------")
print(f"Mean Absolute Error: {mae:.3f}")
print(f"R² Score: {r2:.3f}")

# ==============================
# Predict for all restaurants
# ==============================
ml_df["predicted_stars"] = model.predict(X)

top_predictions = (
    ml_df.sort_values("predicted_stars", ascending=False)
    .head(10)[
        ["name", "predicted_stars", "stars", "review_count", "avg_daily_hours"]
    ]
)

print("\nTOP PREDICTED TUCSON RESTAURANTS")
print("--------------------------------")
print(top_predictions.to_string(index=False))

print("\nDONE ✅")

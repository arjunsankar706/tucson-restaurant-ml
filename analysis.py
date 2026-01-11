import json
import pandas as pd

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
city = "Tucson"
tucson = df[df["city"] == city]

tucson_restaurants = tucson[
    tucson["categories"].str.contains("Restaurant", case=False, na=False)
]

print("Tucson restaurants:", tucson_restaurants.shape)

# ==============================
# Keep restaurants with hours
# ==============================
tucson_with_hours = tucson_restaurants[tucson_restaurants["hours"].notna()].copy()
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
# Show results
# ==============================
result = (
    tucson_with_hours[["name", "stars", "review_count", "avg_daily_hours"]]
    .dropna()
    .sort_values("avg_daily_hours", ascending=False)
    .head(10)
)

print("\nTop Tucson restaurants by average daily open hours:")
print(result)

print("\nDONE ✅")


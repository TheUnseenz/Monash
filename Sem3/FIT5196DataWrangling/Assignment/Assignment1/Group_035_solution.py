# %% [markdown]
# # FIT5196 Assessment 1 - EDA
# 
# Due Date: 23:55, Sunday, 15 September 2025
# 
# 
# ---
# 
# 
# 
# #### Group 35:
# Member 1: Adrian Leong Tat Wei, (27030768), atleo4@student.monash.edu, Contribution
# 
# Member 2: Jun Yuan, (35833645), jyua0050@student.monash.edu, Contribution
# 
# Member 3: Low Xuan Nan (35373849), alow0028@student.monash.edu, Contribution
# 
# ---

# %% [markdown]
# ### Table of Content
# 
# 1. Load, parse and merge data files
# 2. EDA
# 3. Key Key findings, insights and research questions
# 4. Reference
# 
# 

# %%
!pip install wordninja
!pip install cartopy

# %% [markdown]
# ## 1. Load, parse and merge data files
# 
# 
# 
# 
# 

# %%
# from google.colab import drive
# # import os

# # print(os.listdir('.'))

# drive.mount('/content/drive')
# base = "/content/drive/MyDrive/FIT5196/Assignment1/" # for colab

# %%
# begin here if running locally
import pandas as pd
import re

# for local drive
base = ""

# %% [markdown]
# ### 1.1 Load data files

# %%
# https://docs.python.org/3/library/xml.etree.elementtree.html
import xml.etree.ElementTree as ET

# Parse the XML file
tree = ET.parse(base + "Group035.xml")
root = tree.getroot()

# root is <FlickrData>, iterate over each <Record>
records = []
for record in root.findall("Record"):
    record_dict = {child.tag: child.text for child in record}
    records.append(record_dict)

print(type(records))   # <class 'list'>
print(records[0])      # print the first record

# Assuming records (from XML) is loaded as a list of dicts
df_xml = pd.DataFrame(records)


# %%
# https://www.geeksforgeeks.org/python/read-json-file-using-python/
import json

# Open and load the JSON file
with open(base + "Group035.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)

# json_data is now a list of dictionaries
print(type(json_data))  # <class 'list'>
print(json_data[0])     # print the first item

# Assuming json_data is loaded as a list of dicts
df_json = pd.DataFrame(json_data)

# %% [markdown]
# ### 1.2 Merge dataset

# %%
import numpy as np

def clean_text_content(text_input):
    """
    Clean text fields according to assignment specifications:
    - Convert to lowercase (except NaN)
    - Remove XML/JSON tags using regex
    - Remove non-English characters using regex
    - Replace null values with 'NaN'
    """
    if pd.isna(text_input) or text_input is None or str(text_input).strip() == '' or text_input == "null":
        return np.nan

    text_str = str(text_input).strip()
    if text_str == "" or text_str.lower() in {"nan", "null", "none", "n/a", "na"}:
        return np.nan

    text_str = text_str.lower()
    text_str = re.sub(r'<[^>]*>', '', text_str)
    text_str = re.sub(r'&[a-zA-Z0-9#]+;', '', text_str)
    text_str = re.sub(r'[^a-zA-Z0-9\s.,!?;:()\-\'"/\\@#$%&*+=~`|{}[\]^]+', '', text_str)
    text_str = re.sub(r'\s+', ' ', text_str)
    text_str = text_str.strip()

    if len(text_str) == 0:
        return np.nan
    return text_str

def clean_tags(tags_input):
    """
    Specialized tag cleaning: tokenize by comma, drop empties
    """
    if pd.isna(tags_input) or tags_input is None or str(tags_input).strip() == '' or tags_input == "null":
        return np.nan

    text_str = str(tags_input)
    if text_str.lower() == 'nan':
        return np.nan

    cleaned_text = clean_text_content(tags_input)
    if pd.isna(cleaned_text):
        return np.nan

    tags = [tag.strip() for tag in cleaned_text.split(',')]
    tags = [tag for tag in tags if tag and tag != '']

    if not tags:
        return np.nan

    return ','.join(tags)

# Clean alphanumeric columns
alphanumeric_columns = ["UserID", "secret"]
for col in alphanumeric_columns:
    for dataframe in [df_json, df_xml]:
        if col in dataframe.columns:
            dataframe[col] = dataframe[col].astype(str).str.strip()
            #dataframe[col] = dataframe[col].apply(clean_text_content)

# Clean numeric columns
numeric_columns = ["PostID", "server", "ispublic", "isfriend", "isfamily", "farm", "latitude", "longitude"]
for col in numeric_columns:
    for dataframe in [df_json, df_xml]:
        if col in dataframe.columns:
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')

# Clean datetime columns
datetime_columns = ["Post_date", "Taken_date", "min_taken_date"]
for col in datetime_columns:
    for dataframe in [df_json, df_xml]:
        if col in dataframe.columns:
            dataframe[col] = pd.to_datetime(dataframe[col], errors='coerce')

# Clean text columns (Member B focus)
text_columns = ["title", "description", "City", "Country"]
for col in text_columns:
    for dataframe in [df_json, df_xml]:
        if col in dataframe.columns:
            dataframe[col] = dataframe[col].apply(clean_text_content)

# Clean tags with specialized processing
for dataframe in [df_json, df_xml]:
    if "tags" in dataframe.columns:
        dataframe["tags"] = dataframe["tags"].apply(clean_tags)

# Merge datasets
df_all = pd.concat([df_json, df_xml], ignore_index=True)

print(f"Original shape: {df_all.shape}")
# Data cleaning: Drop duplicate posts (same PostID)
df_all = (
    df_all.sort_values("min_taken_date")
          .drop_duplicates(subset=["PostID"], keep="first")
)

print(f"After dropping duplicates: {df_all.shape}")


print(f"Dataset merged successfully: {len(df_all)} records, {len(df_all.columns)} columns")

# %%
import re
import wordninja

def rename_column(colname: str) -> str:
    """
    Convert column names into Title_Case with underscores.
    Handles camelCase, PascalCase, acronyms, and concatenated words.
    """
    # Step 1: Split camelCase / PascalCase into separate words
    # e.g. UserID -> ['User', 'ID'], isPublic -> ['is', 'Public']
    camel_split = re.sub(r'([a-z])([A-Z])', r'\1 \2', colname)

    # Step 2: Split on underscores (already separated words)
    tokens = re.split(r'[_\s]+', camel_split)

    final_tokens = []
    for token in tokens:
        if not token:
            continue
        # Step 3: Preserve acronyms (all caps, length > 1)
        if token.isupper() and len(token) > 1:
            final_tokens.append(token)
        else:
            # Step 4: Word segmentation for lowercase tokens
            if token.islower():
                split_words = wordninja.split(token)
            else:
                split_words = [token]
            # Step 5: Capitalize first letter of each segment
            final_tokens.extend([w.capitalize() for w in split_words])

    # Step 6: Join with underscores
    return "_".join(final_tokens)

# Example usage on your dataframe:
df_all.rename(columns=lambda c: rename_column(c), inplace=True)

print(df_all.columns)

# %%
df_all.to_csv('Group035_dataset.csv', index=False, na_rep="NaN")

# %% [markdown]
# ## 2. EDA

# %%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from collections import Counter
import numpy as np
import calendar
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
from shapely.geometry import LineString, Point
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
warnings.filterwarnings("ignore")

# Configure plotting settings
plt.rcParams['figure.figsize'] = (10, 6)
sns.set_style("whitegrid")

print("EDA libraries loaded successfully")

# %% [markdown]
# ### 2.1 Dataset overview

# %%
print("Dataset Structure and Dimensions")
print("=" * 40)

# Basic dataset information
print(f"Total number of records: {len(df_all)}")
print(f"Number of attributes: {len(df_all.columns)}")
print(f"Dataset shape: {df_all.shape}")

# Column information
summary = (
    pd.DataFrame({
        "Non-Null Count": df_all.count(),
        "Distinct Count": df_all.nunique()
    })
    .reset_index()
    .rename(columns={"index": "Column"})
    .set_index(pd.Index(range(1, len(df_all.columns)+1)))
)

print(summary.to_string(index=True, index_names=False))
# print(f"\nColumn names:")
# for i, col in enumerate(df_all.columns, 1):
#     print(f"{i:2d}. {col}")

# Memory usage
memory_usage = df_all.memory_usage(deep=True).sum() / 1024 / 1024
print(f"\nTotal memory usage: {memory_usage:.2f} MB")

# Data types summary
print(f"\nData types summary:")
dtype_counts = df_all.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f"  {dtype}: {count} columns")

print("\nFirst 5 records:")
print(df_all.head())

# %% [markdown]
# ### 2.2 Univariate analysis

# %%
# Metadata analysis

print(f"Range of values of Is_Family: [{min(df_all['Is_Family'])}, {max(df_all['Is_Family'])}]")
print(f"Range of values of Is_Friend: [{min(df_all['Is_Friend'])}, {max(df_all['Is_Friend'])}]")
print(f"Range of values of Is_Public: [{min(df_all['Is_Public'])}, {max(df_all['Is_Public'])}]")

def bucketize(n):
    if n == 1:
        return "1"
    elif 2 <= n <= 5:
        return "2-5"
    elif 6 <= n <= 10:
        return "6-10"
    elif 11 <= n <= 50:
        return "11-50"
    elif 51 <= n <= 100:
        return "51-100"
    elif 101 <= n <= 1000:
        return "101-1000"
    else:
        return "1001+"
# Order by magnitude
bucket_order = ["1", "2-5", "6-10", "11-50", "51-100", "101-1000", "1001+"]

# 1. Posts per user
posts_per_user = df_all.groupby("User_ID")["Post_ID"].count()
posts_per_user_bucketed = posts_per_user.map(bucketize).value_counts()
posts_per_user_bucketed = posts_per_user_bucketed.reindex(bucket_order).dropna()

# 2. Posts per server
posts_per_server = df_all.groupby("Server")["Post_ID"].count()
posts_per_server_bucketed = posts_per_server.map(bucketize).value_counts()
posts_per_server_bucketed = posts_per_server_bucketed.reindex(bucket_order).dropna()

# 3. Posts per farm (non-bucketed)
posts_per_farm = df_all.groupby("Farm")["Post_ID"].count()

# Plot in subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes = axes.flatten()

def plot_series(ax, series, title, xlabel, ylabel, logy=True):
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_ylim(1, 100000)

# 1. Posts per user
plot_series(axes[0], posts_per_user_bucketed, "Distribution of Users by Posts Made", "Posts per User (bucketed)", "Number of Users")

# 2. Posts per server
plot_series(axes[1], posts_per_server_bucketed, "Distribution of Servers by Posts Received", "Posts per Server (bucketed)", "Number of Servers")

# 4. Posts per farm (non-bucketed, alphabetical order OK)
plot_series(axes[2], posts_per_farm, "Posts per Farm", "Farm", "Number of Posts")

plt.tight_layout()
plt.show()



# %%
# Text length distributions
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
title_lengths = df_all['Title'].dropna().str.len()
plt.hist(title_lengths, bins=30, alpha=0.7)
plt.title('Title Length Distribution')
plt.xlabel('Characters')

plt.subplot(1, 3, 2)
desc_lengths = df_all['Description'].dropna().str.len()
plt.hist(desc_lengths, bins=30, alpha=0.7)
plt.title('Description Length Distribution')
plt.xlabel('Characters')

plt.subplot(1, 3, 3)
tag_counts = df_all['Tags'].dropna().apply(lambda x: len(x.split(',')))
plt.hist(tag_counts, bins=20, alpha=0.7)
plt.title('Number of Tags per Post')
plt.xlabel('Tag Count')

plt.tight_layout()
plt.show()

# %%
# Tag analysis
all_tags = []
for tags_str in df_all['Tags'].dropna():
    all_tags.extend(tags_str.split(','))

tag_freq = Counter(all_tags).most_common(20)

# Plot top tags
plt.figure(figsize=(10, 6))
tags_list, counts_list = zip(*tag_freq)
plt.bar(tags_list, counts_list)
plt.title('Top 20 Most Frequent Tags')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print(f"Total unique tags: {len(set(all_tags))}")
print(f"Posts with tags: {df_all['Tags'].notna().sum()} / {len(df_all)}")

# %%
# description length distributions
desc_lengths = df_all['Description'].dropna().str.len()

plt.figure(figsize=(8, 5))
plt.hist(desc_lengths, bins=40, alpha=0.7, color='lightblue')
plt.title('Description Length Distribution')
plt.xlabel('Characters')
plt.ylabel('Frequency')
plt.axvline(desc_lengths.mean(), color='red', linestyle='--', label=f'Mean: {desc_lengths.mean():.0f}')
plt.legend()
plt.show()

print(f"description statistics: mean={desc_lengths.mean():.0f}, max={desc_lengths.max()}")

# %%
# tag co-occurrence analysis
# get top tags first
all_tags = []
for tags in df_all['Tags'].dropna():
    all_tags.extend(tags.split(','))

top_tags = [tag for tag, count in Counter(all_tags).most_common(10)]

# build co-occurrence matrix
import numpy as np
cooc = np.zeros((10, 10))

for tags in df_all['Tags'].dropna():
    tag_list = tags.split(',')
    for i, tag1 in enumerate(top_tags):
        for j, tag2 in enumerate(top_tags):
            if tag1 in tag_list and tag2 in tag_list and i != j:
                cooc[i][j] += 1

# plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cooc, xticklabels=top_tags, yticklabels=top_tags,
            annot=True, fmt='.0f', cmap='Blues')
plt.title('Tag Co-occurrence (Top 10 Tags)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("most common tag combinations:")
for i in range(len(top_tags)):
    for j in range(i+1, len(top_tags)):
        if cooc[i][j] > 50:  # only show frequent combinations
            print(f"{top_tags[i]} + {top_tags[j]}: {int(cooc[i][j])} times")

# %%
# Geospatial EDA

# Distinct and Missing Country values
print("Number of distinct Country:", df_all["Country"].nunique())
print("Distinct Country:", df_all["Country"].dropna().unique())
print("Missing values for Country:", df_all["Country"].isna().sum())

# Distinct and Missing City values
print("Number of distinct City:", df_all["City"].nunique())
print("Distinct City:", df_all["City"].dropna().unique())
print("Missing values for City:", df_all["City"].isna().sum())

# %%
# Country names

# Regex helper
RE_SPLIT_MULTI = re.compile(r"[\/,&;|]|(?:\band\b)", re.I)

# List of the valid countries
valid_countries_list = ["Australia", "Ireland", "Italy", "United States",
                    "New Zealand", "United Kingdom", "Switzerland",
                   "Portugal", "Denmark", "Czech Republic", "Belgium",
                   "Canada", "Netherlands", "Japan", "Singapore", "Finland",
                   "Thailand", "Taiwan", "Russia", "China", "Hong Kong",
                   "Argentina", "France", "United Arab Emirates", "Sweden",
                   "Sri Lanka", "Spain", "Germany", "Austria", "Belarus",
                   "Papua New Guinea", "South Korea", "Hungary", "Vietnam",
                   "Bulgaria", "Poland", "Estonia", "Malaysia", "Luxembourg",
                   "Saudi Arabia", "Chile", "Liechtenstein", "Brazil",
                   "Greece", "Egypt", "Philippines", "Norway", "Kazakhstan"]

COUNTRY_RE = re.compile(r"^(?:" + "|".join(map(re.escape, valid_countries_list)) + r")$", re.I)

def country_check(country_val):
  """
  Check if a country name is valid based on valid_countries_list

  Returns:
    (NaN, invalid) -> if no value or not a valid country
    (country, valid) -> if valid_name = 1 and is valid
    (NaN, multi) -> if valid_name > 1
  """
  s = clean_text_content(country_val)
  if pd.isna(s):
    return np.nan, "invalid" # No value
  parts = [p.strip() for p in RE_SPLIT_MULTI.split(s) if p.strip()] # Split on multi separators defined in RE_SPLIT_MULTI
  valid_name = [p for p in parts if COUNTRY_RE.match(p)] # Keep as valid_names when country name(s) matched
  if len(valid_name) == 1:
    return valid_name[0], "valid" # Exactly one country name
  elif len(valid_name) > 1:
    return np.nan, "multi" # More than one country name
  else:
    return np.nan, "invalid" # Country names unmatched

# Apply the validator
tmp = df_all["Country"].apply(country_check)

# Create a df to store the values
df_country = pd.DataFrame({
    "Country": df_all["Country"],
    "Valid_country": tmp.apply(lambda x: x[0]),
    "Country_status": tmp.apply(lambda x: x[1])
})

# Data quality and accuracy
# Plot country_status bar chart
df_country_status = df_country["Country_status"].value_counts().reset_index()
df_country_status.columns = ["Country_status", "Count"]

sns.barplot(data = df_country_status, x = "Country_status", y = "Count", palette = "Set2")
plt.title("The Number of Records for each Country Name Status")
plt.ylabel("Number of Records")
plt.xlabel("Status of Country Name (Valid/Invalid/Multi)")
plt.yscale("log")
plt.gca().yaxis.set_major_formatter(ScalarFormatter())

total = df_country_status["Count"].sum()

# add label
for i, v in enumerate(df_country_status["Count"].values):
  pct = v / total * 100
  label = f"{v} ({pct:.3f}%)"
  plt.text(i, v * 1.05, label, ha = "center")

plt.show()


# %%
# Skewness of the dataset
# Plot country bar chart
df_country_name = df_country["Valid_country"].value_counts().reset_index().head(10)
df_country_name.columns = ["Country", "Count"]

sns.barplot(data = df_country_name, x = "Country", y = "Count", palette = "viridis")
plt.title("Top 10 Countries by Number of Records")
plt.xlabel("Country Name")
plt.ylabel("Number of Records")
plt.yscale("log")
plt.gca().yaxis.set_major_formatter(ScalarFormatter())
plt.xticks(rotation = 45, ha = "right")

total = df_country_status["Count"].sum()

# add label
for i, v in enumerate(df_country_name["Count"].values):
  pct = v / total * 100
  label = f"{v} ({pct:.1f}%)"
  plt.text(i, v * 1.05, label, ha = "center")

plt.show()

# %%
# Data quality and accuracy
# Longitude and Latitude

# Check min and max values for longitude
print(min(df_all["Longitude"]))
print(max(df_all["Longitude"]))
print(min(df_all["Latitude"]))
print(max(df_all["Latitude"]))

# Check if any longitude or latitude values fall outside the valid range
invalid_lon = df_all[(df_all["Longitude"] < -180) | (df_all["Longitude"] > 180)]
invalid_lat = df_all[(df_all["Latitude"] < -90) | (df_all["Latitude"] > 90)]

print("Invalid longitude values:", len(invalid_lon))
print("Invalid latitude values:", len(invalid_lat))

# %%
# Temporal EDA

# Missing Taken_Date value
print("Missing values for Taken_Date:", df_all["Taken_Date"].isna().sum())

# Missing Post_Date value
print("Missing values for Post_Date:", df_all["Post_Date"].isna().sum())

# Missing Min_Taken_Date value
print("Missing values for Min_Taken_Date:", df_all["Min_Taken_Date"].isna().sum())

# %%
# Count of Taken_Date
taken_per_year = df_all["Taken_Date"].dt.year.dropna().value_counts().sort_index()
taken_year = taken_per_year.index.astype(int) # ensure year is in integer form to avoid decimals
taken_counts = taken_per_year.values

plt.plot(taken_year, taken_counts, marker ='o', linestyle ='-')

plt.title("Photos Taken per Year")
plt.xlabel("Year")
plt.ylabel("Number of Photos Taken")

# Add label
for i, v in zip(taken_year, taken_counts):
  plt.text(i, v * 1.01, str(v), ha = "center", va = "bottom")

plt.show()

# %%
# Count of Post_Date
post_per_year = df_all["Post_Date"].dt.year.dropna().value_counts().sort_index()
post_year = post_per_year.index.astype(int) # ensure year is in integer form to avoid decimals
post_counts = post_per_year.values

plt.plot(post_year, post_counts, marker ='o', linestyle ='-')

plt.title("Photos Posted per Year")
plt.xlabel("Year")
plt.ylabel("Number of Photos Posted")
plt.xticks(post_year)
plt.yscale("log")
plt.gca().yaxis.set_major_formatter(ScalarFormatter())

# Add label
for i, v in zip(post_year, post_counts):
  plt.text(i, v * 1.1, str(v), ha = "center", va = "bottom")

plt.show()

# %%
# Count of Min_Taken_Date
min_taken_per_year = df_all["Min_Taken_Date"].dt.year.dropna().value_counts().sort_index()
min_year = min_taken_per_year.index.astype(int) # ensure year is in integer form to avoid decimals
min_counts = min_taken_per_year.values

plt.plot(min_year, min_counts, marker ='o', linestyle ='-')

plt.title("Min Photos Taken per Year")
plt.xlabel("Year")
plt.ylabel("Min Number of Photos Taken")
plt.gca().yaxis.set_major_formatter(ScalarFormatter())

# Add label
for i, v in zip(min_year, min_counts):
  plt.text(i, v * 1.01, str(v), ha = "center", va = "bottom")

plt.show()

# %% [markdown]
# ### 2.3 Bivariate analysis

# %%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def bucketize(n):
    if n == 1:
        return "1"
    elif 2 <= n <= 5:
        return "2-5"
    elif 6 <= n <= 10:
        return "6-10"
    elif 11 <= n <= 50:
        return "11-50"
    elif 51 <= n <= 100:
        return "51-100"
    elif 101 <= n <= 1000:
        return "101-1000"
    else:
        return "1001+"
# Order by magnitude
bucket_order = ["1", "2-5", "6-10", "11-50", "51-100", "101-1000", "1001+"]

# 4. Users per server
users_per_server = df_all.groupby("Server")["User_ID"].nunique()
users_per_server_bucketed = users_per_server.map(bucketize).value_counts()
users_per_server_bucketed = users_per_server_bucketed.reindex(bucket_order).dropna()

# 5. Users per farm (non-bucketed)
users_per_farm = df_all.groupby("Farm")["User_ID"].nunique()

# Plot in subplots
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes = axes.flatten()

def plot_series(ax, series, title, xlabel, ylabel, logy=True):
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_ylim(1, 100000)

# 4. Users per server
plot_series(axes[0], users_per_server_bucketed, "Distribution of Servers by User Count", "Users per Server (bucketed)", "Number of Servers")

# 5. Users per farm (non-bucketed)
plot_series(axes[1], users_per_farm, "Users per Farm", "Farm", "Number of Users")

plt.tight_layout()
plt.show()



# %%
# Geospatial EDA

# World-Map
plt.scatter(df_all["Longitude"], df_all["Latitude"],
            alpha = 0.5, s = 10)

plt.title("Global Map with Longitude and Latitude Values")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# World coordinate ranges
plt.xlim(-180, 180)
plt.ylim(-90, 90)

plt.grid(True, linestyle = "--", alpha = 0.5)
plt.show()

# %%
# Monthly count of Taken_Date
taken_2023 = df_all.loc[df_all["Taken_Date"].dt.year == 2023, "Taken_Date"]
taken_per_month = taken_2023.dt.month.value_counts().sort_index()

# Monthly count of Post_Date
post_2023 = df_all.loc[df_all["Post_Date"].dt.year == 2023, "Post_Date"]
post_per_month = post_2023.dt.month.value_counts().sort_index()

# Months
months = [calendar.month_name[i] for i in taken_per_month.index]

plt.plot(months, taken_per_month, marker ='^', linestyle ='--', label = "Photos Taken")
plt.plot(months, post_per_month, marker ='o', linestyle ='--', label = "Photos Posted")

plt.title("Monthly Trend of Photos Taken and Posted in 2023")
plt.xlabel("Month")
plt.ylabel("Number of Records")
plt.xticks(rotation = 45, ha = "right")
plt.legend()

plt.show()

# %% [markdown]
# ### 2.4 Multivariate analysis

# %%
# Temporal and Geospatial EDA

# Photo Density Heatmap
photo_heatmap_2023 = df_all[df_all["Post_Date"].dt.year == 2023]

# Set up map projection
fig, ax = plt.subplots(figsize=(12,8), subplot_kw={'projection': ccrs.PlateCarree()})

# World features
ax.add_feature(cfeature.LAND, facecolor = 'lightgrey')
ax.add_feature(cfeature.COASTLINE, linewidth = 0.5)
ax.add_feature(cfeature.BORDERS, linewidth = 0.5)

post_heatmap = sns.kdeplot(
    x = photo_heatmap_2023["Longitude"], y = photo_heatmap_2023["Latitude"],
    fill = True, cmap = "Reds", levels = 100, alpha = 0.7, ax = ax
)

# Add legend
contour = ax.collections[-1]
cbar = plt.colorbar(contour, ax = ax, orientation = "vertical", shrink = 0.6, pad = 0.02)
cbar.set_label("Density", rotation = -90, labelpad = 20)

plt.title("Photo Density Heatmap in Australia (2023)")

# Major cities coordinates
major_cities = {
    "Sydney": (-33.8688, 151.2093),
    "Melbourne": (-37.8136, 144.9631),
    "Brisbane": (-27.4698, 153.0251),
    "Perth": (-31.9505, 115.8605),
    "Adelaide": (-34.9285, 138.6007)
}

# Add labels
for city, (lat, lon) in major_cities.items():
    ax.text(lon, lat, city, transform = ccrs.PlateCarree(),
            fontsize = 10, ha ='center', va = 'bottom', color = 'black')

plt.show()

# %%
# User profile/Location/Time Multivariate EDA

# PARAMETERS
EPS_KM = 25                       # spatial threshold in km
CONSISTENT_POSTS = 2              # >= posts to consider a 'consistent' cluster (home)
CONSISTENT_SPAN_DAYS = 30         # >= days for consistent cluster
HOLIDAY_MAX_SPAN_DAYS = 30        # <= days for short-term holiday
MIN_POSTS_FOR_USER = 3            # users with fewer posts will be ignored
MIN_BUFFER_DEG = 0.5              # minimum buffer (deg) for zooming (≈ ~50 km lat)

# --- helper: cluster user posts using haversine DBSCAN (min_samples=1 allows singletons)
def cluster_user_posts(user_df, eps_km=EPS_KM):
    coords = user_df[['Latitude', 'Longitude']].values
    if len(coords) == 0:
        user_df = user_df.copy()
        user_df['Cluster'] = np.array([], dtype=int)
        return user_df
    coords_rad = np.radians(coords)
    db = DBSCAN(eps=eps_km / 6371.0, min_samples=1, metric='haversine').fit(coords_rad)
    user_df = user_df.copy()
    user_df['Cluster'] = db.labels_.astype(int)
    return user_df

# --- core: detect movements for a single user
def detect_movements(user_df,
                     eps_km=EPS_KM,
                     consistent_posts=CONSISTENT_POSTS,
                     consistent_span_days=CONSISTENT_SPAN_DAYS,
                     holiday_max_span_days=HOLIDAY_MAX_SPAN_DAYS,
                     min_posts_for_user=MIN_POSTS_FOR_USER):
    """
    Returns a list of movement dictionaries for the user (maybe empty).
    Each movement dict contains: category ('holiday'|'migration'), user_id, from, to, count, start, end
    """
    movements = []

    # Skip users with too few posts overall
    if len(user_df) < min_posts_for_user:
        return movements

    # Ensure sorted by date and dates exist
    user_df = user_df.sort_values('Taken_Date').reset_index(drop=True)

    # Spatial clustering (allow singletons so holidays with 1 post are valid clusters)
    clustered = cluster_user_posts(user_df, eps_km=eps_km)

    # Build clusters as a dict (exclude any -1 just in case)
    clusters = {lab: grp for lab, grp in clustered.groupby('Cluster') if lab != -1}

    # Your requested safe-check (keeps backward-compatible behavior)
    if not clusters or not isinstance(clusters, dict):
        return movements

    # Summarize all clusters (count, date range, centroid)
    all_clusters = []
    for label, g in clusters.items():
        # drop NaT in Taken_Date just in case
        valid_dates = g['Taken_Date'].dropna()
        if len(valid_dates) == 0:
            # cannot establish date ranges for this cluster — still include as zero-span
            start = end = pd.NaT
            span = 0
        else:
            start = valid_dates.min()
            end = valid_dates.max()
            span = (end - start).days

        all_clusters.append({
            'label': int(label),
            'center': (float(g['Latitude'].mean()), float(g['Longitude'].mean())),
            'start': start,
            'end': end,
            'span_days': int(span),
            'count': len(g)
        })

    # Split clusters by type:
    consistent_clusters = [c for c in all_clusters if c['count'] >= consistent_posts and c['span_days'] >= consistent_span_days]
    # short_clusters allow singletons or small-span clusters (holidays)
    short_clusters = [c for c in all_clusters if c['span_days'] <= holiday_max_span_days]

    # Need at least one consistent cluster to call a "home" or to reason about holiday-from-home
    if not consistent_clusters:
        return movements

    # Choose main cluster (home) = largest by count, tie-breaker by span_days
    main_cluster = max(consistent_clusters, key=lambda x: (x['count'], x['span_days']))
    home_center = main_cluster['center']

    # ------------ Category 2 (migration) detection ------------
    # Look for chronological switch between consistent clusters: earlier cluster(s) -> later cluster(s)
    if len(consistent_clusters) >= 2:
        consistent_sorted = sorted(consistent_clusters, key=lambda x: x['start'] if pd.notna(x['start']) else pd.Timestamp.min)
        first = consistent_sorted[0]
        last = consistent_sorted[-1]
        # require the earlier cluster ends before the later cluster starts (clear switch)
        if pd.notna(first['end']) and pd.notna(last['start']) and (first['end'] < last['start']):
            dist_km = geodesic(first['center'], last['center']).km
            if dist_km > eps_km:
                movements.append({
                    'category': 'migration',
                    'user_id': user_df['User_ID'].iloc[0],
                    'from': first['center'],
                    'to': last['center'],
                    'count': int(first['count'] + last['count']),
                    'start': first['start'],
                    'end': last['end']
                })
                # If migration found, we prioritize it and return (do not also label holiday)
                return movements

    # ------------ Category 1 (holiday / short trip) detection ------------
    # Any short-cluster sufficiently far from home -> holiday
    for sc in short_clusters:
        # Ignore if this short cluster is essentially the same as the main consistent cluster (very near)
        dist_km = geodesic(home_center, sc['center']).km
        if dist_km > eps_km:
            movements.append({
                'category': 'holiday',
                'user_id': user_df['User_ID'].iloc[0],
                'from': home_center,
                'to': sc['center'],
                'count': int(sc['count']),
                'start': sc['start'],
                'end': sc['end']
            })

    return movements


# === RUN over all users and collect results ===
def analyze_all_users(df_all):
    movements = []
    skipped_too_few_posts = 0
    skipped_no_consistent = 0
    processed_users = 0

    # ensure dates are proper
    df = df_all.copy()
    df['Taken_Date'] = pd.to_datetime(df['Taken_Date'], errors='coerce')

    for uid, user_df in df.groupby('User_ID'):
        if len(user_df) < MIN_POSTS_FOR_USER:
            skipped_too_few_posts += 1
            continue

        processed_users += 1
        user_moves = detect_movements(user_df)
        if not user_moves:
            skipped_no_consistent += 1
            continue
        movements.extend(user_moves)

    # summary
    print(f"Users processed (>= {MIN_POSTS_FOR_USER} posts): {processed_users}")
    print(f"Skipped (too few posts): {skipped_too_few_posts}")
    print(f"Skipped (no consistent clusters): {skipped_no_consistent}")
    print(f"Total movements found: {len(movements)}")

    return movements

# === Visualize (static map) ===

def plot_user_movements_static(movements):
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()

    # Base map
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="aliceblue")

    # Define gradient color ranges: origin → destination
    gradients = {
        "holiday": (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.8, 0.0])),   # blue → green
        "migration": (np.array([1.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))  # yellow → red
    }

    all_lats, all_lons = [], []

    for m in movements:
        start_loc = m.get("from")   # (lat, lon)
        end_loc   = m.get("to")     # (lat, lon)
        cat       = m.get("category")

        if not start_loc or not end_loc or cat not in gradients:
            continue

        start_lon, start_lat = start_loc[1], start_loc[0]
        end_lon, end_lat     = end_loc[1], end_loc[0]

        all_lats.extend([start_lat, end_lat])
        all_lons.extend([start_lon, end_lon])

        # Line thickness (log-scale)
        width = np.log1p(m.get("count", 1))/5
        if cat == "migration":
            width *= 5.0

        # Break into N segments
        N = 20
        lons = np.linspace(start_lon, end_lon, N)
        lats = np.linspace(start_lat, end_lat, N)

        # Start and end colors
        c_start, c_end = gradients[cat]

        for i in range(N - 1):
            frac = i / (N - 1)
            color = (1 - frac) * c_start + frac * c_end
            ax.plot(
                [lons[i], lons[i+1]],
                [lats[i], lats[i+1]],
                color=color,
                linewidth=width,
                transform=ccrs.PlateCarree(),
                alpha=0.15
            )

    # Zoom to bounding box
    if all_lons and all_lats:
        margin = 2.0
        ax.set_extent([
            min(all_lons) - margin, max(all_lons) + margin,
            min(all_lats) - margin, max(all_lats) + margin
        ], crs=ccrs.PlateCarree())

    # Legend
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color="blue", label="Holiday Origin"),
        mpatches.Patch(color="green", label="Holiday Destination"),
        mpatches.Patch(color="yellow", label="Migration Origin"),
        mpatches.Patch(color="red", label="Migration Destination")
    ]

    ax.legend(handles=legend_handles, loc="lower left", title="Movement Legend")
    plt.title("User Movements with Directionality", fontsize=14)
    plt.show()

movements = analyze_all_users(df_all)
plot_user_movements_static(movements)


# %% [markdown]
# ## 3. Key findings, insights and research questions
# 
# 

# %% [markdown]
# ### 3.1 Key findings

# %% [markdown]
# **Metadata Key Findings**  
# The count of distinct values in access fields Is_Family, Is_Friend, Is_Public show that the collected data is unanimously Is_Public.
# This makes sense, because it simply means that the dataset was collected from public data,
# so naturally, data which had Is_Family and Is_Friend tags simply could not be collected.
# ```python
# print(f"Range of values of Is_Family: [{min(df_all['Is_Family'])}, {max(df_all['Is_Family'])}]") # [0, 0]
# print(f"Range of values of Is_Friend: [{min(df_all['Is_Friend'])}, {max(df_all['Is_Friend'])}]") # [0, 0]
# print(f"Range of values of Is_Public: [{min(df_all['Is_Public'])}, {max(df_all['Is_Public'])}]") # [1, 1]
# ```
# Some of the Post_ID were not distinct (5352 out of the original 70k). They have been cleaned as they were found to be duplicate data.
# 
# **User post distribution:**  
# The distribution of the number of posts users make follows a log distribution. The vast majority of users make very few posts, but the users that do post a lot, post an incredible amount.
# 

# %% [markdown]
# **Textual Key Findings**
# 1. **Data completeness:** 95.0% posts have titles, 44.5% have descriptions, 65.8% have tags
# 2. **Text characteristics:** Average 27 chars per title, 175 chars per description, 11 tags per post  
# 3. **Popular topics:** Australia, landscape, melbourne, nature, nsw are most frequent tags
# 4. **Tag patterns:** Strong co-occurrence between location and nature tags observed

# %%
# analyze tag patterns for ML categories
nature_tags = ['landscape', 'nature', 'outdoor', 'beach', 'water', 'sky']
urban_tags = ['city', 'urban', 'melbourne', 'architecture', 'street']
australia_tags = ['australia', 'nsw', 'victoria', 'queensland']

nature_posts = sum(1 for tags_str in df_all['Tags'].dropna()
                   if any(tag in nature_tags for tag in tags_str.split(',')))
urban_posts = sum(1 for tags_str in df_all['Tags'].dropna()
                  if any(tag in urban_tags for tag in tags_str.split(',')))
australia_posts = sum(1 for tags_str in df_all['Tags'].dropna()
                      if any(tag in australia_tags for tag in tags_str.split(',')))

print(f"nature posts: {nature_posts:,}")
print(f"urban posts: {urban_posts:,}")
print(f"australia posts: {australia_posts:,}")
print(f"posts with title+tags: {df_all[['Title','Tags']].dropna().shape[0]:,}")

# %% [markdown]
# **Geospatial Key Findings**
# 
# 1. Only 50.665% of the values in Country column contain valid country names, while 49.312% with invalid country names and 0.023% are multivalued.
# 2. Australia accounts for the largest share of records in Country column, representing 48.1% of the dataset.
# 3. Scatter map of longitude and latitude shows that the dataset only contains coordinates from Australia.
# 
# **Temporal Key Findings**
# 
# 1. The data collection for temporal data is inconsistent as photo posting records exists as early as 2007, while photo-taking records only appear from 2017 onwards.
# 2. The number of photos taken, posted, and the minimum photos taken all peaked in 2019, with 14,628, 13,659, and 14,630 photos respectively.
# 3. Photos taken plummeted to only 6 photos in 2024. Photos posted remained less than 10 photos annually during 2007 to 2017, then surged in 2018 and 2019 with over 12,000 posts annually.
# 4. The Min_Taken_Date distribution mirrors the Taken_Date, with a peak in 2019 and a drop in 2020-2022, followed by recovery in 2023.
# 5. Photos taken in 2023 peaked during May (~ 1090) and showed another smaller rise in September (~ 980), while the lowest point was in June (~ 700).
# 6. Photos posted in 2023 peaked during May (~ 1030) but gradually declined in the second half of the year with a slight increase in October(~ 800 - 930 per month).
# 
# **Geospatial and Temporal Key Findings**
# 1. The highest concentration for photo activity in 2023 is clustered around major cities such as Sydney, Melbourne, and Brisbane, with secondary hotspot are visible in Adelaide and Perth.

# %% [markdown]
# **Geospatial Key Insights**
# 
# 1. The mismatch between Country column with Longitude and Latitude column highlights the need to rely on coordinates over country field.
# 
# **Temporal Key Insights**
# 1. The mismatch of year range in temporal records suggested either incomplete metadata or selective collection. Analyses involving long-term trends should therefore focus on the overlapping window from 2018 to 2023.
# 2. The sharp rise in 2018 for photo posting activity, suggested either change in platform usage or data collection. This aligns with the peak of photo-taking activity, showing consistency between taken and posted patterns.
# 3. The sharp rise in photo-taking activity from February to May corresponds with Australia's Autumn season where many major cultural festivals are held (Australia, 2022). In contrast, the steady activity observed from June to September aligns with Winter season, when landscape photography is popular due to clearer skies and tranquil landmarks (AdamC, 2024).
# 4. The photo-posting taking trend shows that there are some delayed or backlog posting during peak photo-taking period in April and September, suggesting that the delayed posting behaviour may occur during trips, festivals, or seasonal events.
# 
# **Geospatial and Temporal Key Insights**
# 
# 1. Photo-taking activity may strongly associated with population density (Australian Bureau of Statistics, 2025) and tourism activities, reflecting user behaviour centered around major cities.

# %% [markdown]
# **User, Location and Time Insights**  
# User movement can be identified by identifying user post location and dates. We find that a majority of movement for holiday is from major cities into the countryside, while a majority of long-term migration is from the countryside into major cities.
# 

# %% [markdown]
# ### 3.2 Machine Learning research questions and justification

# %% [markdown]
# **Question 1:** Can we predict post categories from title and description text?
# 
# From looking at the tags, I noticed that different types of posts have different tag patterns. Nature photos usually have tags like 'landscape', 'outdoor', 'beach', while city photos have 'city', 'melbourne', 'architecture'. Since most posts have titles (95%) and the tag patterns are quite different, we could probably train a model to predict the category by analyzing the words in titles and descriptions. The TF-IDF method could work well here to find important words for each category.

# %% [markdown]
# **Question 2:** Can we group similar posts together using their tags?
# 
# Since each post has about 11 tags on average, there's lots of information to compare posts. When I looked at the data, posts about similar topics tend to share many tags - like landscape photos often have 'nature', 'outdoor', 'australia' together. We could measure how similar two posts are by counting how many tags they share (Jaccard similarity) and then use clustering algorithms like K-means to group them. The co-occurrence analysis already shows that some tags naturally go together, which makes clustering possible.

# %% [markdown]
# **Question 3:** Who are the people posting extraordinary amounts of posts?  
# 
# When looking at the distribution of posts made by users, I notice that many users don't post a lot, but some users post an incredible volume of posts. Making an incredible volume of posts is a behaviour of interest for a multitude of reasons (For example, a study on students (Li et al., 2024) found that brain rot content, often associated with social media usage, significantly affects student academic anxiety, academic engagement, and mindfulness for the worse), so we want to analyze what type of people they are. Location of these users is an obvious trait to look for, but we can also look at the post categories they post, the language they use, to identify their behaviours.

# %% [markdown]
# **Question 4:** How can we better distribute server load?  
# 
# The number of users and posts are not very directly correlated with the number of servers and/or farms, some servers and/or farms serve many more people than others. This could be indicative of poor load balancing, which is an infrastructure problem that virtually every company wants to always improve on (if it is feasible to do so). By identifying the noteworthy servers and/or farms, and comparing their locations in proximity to their users, we can identify weak links in the infrastructure where some places may be overburdened, while other places may have too much resources for their value.  

# %% [markdown]
# **Question 5:** Can we predict photo taking and posting activity based on season?
# 
# The monthly trend showed fluctuations during seasons, highlighting photo taking and posting activities peak in May and dropping significantly in June. This reflects that seasonality influences user behaviour in both photo taking and posting. Therefore, Months input extracted from Post_Date and Taken_Date are the key predictors to predict the seasonal photo taking and posting activities. For instance, regression models such as Random Forest Regressor could be applied to predict the number of photos being taken or posted throughout the season, and using classification models to categorise months into high or low activity levels. The prediction may benefit platforms in resource planning to forecast server loads and storage needs by predicting when uploads will spike to avoid performance issues during seasonal surges.
# 

# %% [markdown]
# **Question 6:** Can we identify major hotspots of user activity in Australia during a certain year?
# 
# From the photo density heatmap for 2023, it showed high concentration of photo activity around Australia's major cities such as Sydney, Melbourne, and Brisbane, while rural area exhibited minimal activity. By using geospatial data such as Longitude and Latitude, pairing with Post_Date or Taken_Date allow us to identify the major hotspots of user activity in Australia over the years or a certain year with machine learning techniques such as K-means, XGBoost, etc. This can benefit tourism to target hotspots for event promotion and tourism campaigns in high-activity cities.

# %% [markdown]
# **Question 7:** Can we identify user movement through social media data?  
# 
# Based on the multivariate analysis using user, location and time data, we can identify where users have been, and by extension, where people have moved around to, and whether they are there for a short holiday or have moved there for long-term stay. This is tremendously helpful not just to tourism marketing and planning, but also for long-term government and urban planning.

# %% [markdown]
# ## 4. Reference

# %% [markdown]
# 1. AdamC. (2024, April 19). Is winter a good time to visit Australia? | ULTIMATE. Ultimate Adventure Travel. https://www.ultimate.travel/our-blog/is-winter-a-good-time-to-visit-australia/
# 2. Australia, T. (2022, March 16). Australia’s seasons - Tourism Australia. Www.australia.com. https://www.australia.com/en-my/facts-and-planning/when-to-go/australias-seasons.html
# 3. Australian Bureau of Statistics. (2025). Capital cities continue strong growth. Australian Bureau of Statistics. https://www.abs.gov.au/media-centre/media-releases/capital-cities-continue-strong-growth
# 4. IBM. (n.d.). Exploratory Data Analysis. Ibm.com. https://www.ibm.com/think/topics/exploratory-data-analysis
# 5. Li, G., Geng, Y., & Wu, T. (2024). Effects of short-form video app addiction on academic anxiety and academic engagement: The mediating role of mindfulness. Frontiers in Psychology, 15. https://doi.org/10.3389/fpsyg.2024.1428813



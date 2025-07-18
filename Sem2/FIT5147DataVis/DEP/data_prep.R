# prepare_data.R

# --- 0. Setup: Load Libraries ---
library(tidyverse)
library(readr)
library(dplyr)
library(sf) # For spatial data handling
library(rnaturalearth) # For map geometries
library(rnaturalearthdata) # For map geometries data
library(countrycode) # For converting country names/codes

# --- 1. Define File Paths ---
# Assuming your data files are in a 'data/' subdirectory relative to this script
# and 'data/metadata/' for income_groups.csv
data_dir <- "data/"
metadata_dir <- "data/metadata/"

gdp_per_capita_file <- paste0(data_dir, "gdp_per_capita.csv")
imports_gdp_ratio_file <- paste0(data_dir, "imports_gdp_ratio.csv")
exports_gdp_ratio_file <- paste0(data_dir, "exports_gdp_ratio.csv")
death_rate_air_pollution_file <- paste0(data_dir, "death_rate_air_pollution.csv")
mismanaged_plastic_waste_share_file <- paste0(data_dir, "mismanaged_plastic_waste_share.csv")
world_gdp_share_file <- paste0(data_dir, "world_gdp_share.csv")
income_groups_file <- paste0(metadata_dir, "income_groups.csv")

# --- 2. Load Raw Data ---
message("Loading raw data...")
gdp_data <- read_csv(gdp_per_capita_file, col_names = FALSE, skip = 1)
imports_data <- read_csv(imports_gdp_ratio_file, col_names = FALSE, skip = 1)
exports_data <- read_csv(exports_gdp_ratio_file, col_names = FALSE, skip = 1)
pollution_death_data <- read_csv(death_rate_air_pollution_file, col_names = FALSE, skip = 1)
plastic_waste_data <- read_csv(mismanaged_plastic_waste_share_file, col_names = FALSE, skip = 1)
percentage_world_gdp_data <- read_csv(world_gdp_share_file, col_names = FALSE, skip = 1)
income_groups_raw <- read_csv(income_groups_file)

# --- 3. Rename Columns and Clean Data ---
message("Renaming columns and cleaning data...")
gdp_data <- gdp_data %>%
    rename(country_name = 1, country_code = 2, year = 3, gdp_per_capita = 4) %>%
    mutate(gdp_per_capita = as.numeric(gdp_per_capita))

imports_data <- imports_data %>%
    rename(country_name = 1, country_code = 2, year = 3, imports_gdp_ratio = 4) %>%
    mutate(imports_gdp_ratio = as.numeric(imports_gdp_ratio))

exports_data <- exports_data %>%
    rename(country_name = 1, country_code = 2, year = 3, exports_gdp_ratio = 4) %>%
    mutate(exports_gdp_ratio = as.numeric(exports_gdp_ratio))

pollution_death_data <- pollution_death_data %>%
    rename(country_name = 1, country_code = 2, year = 3, death_rate_air_pollution = 4) %>%
    mutate(death_rate_air_pollution = as.numeric(death_rate_air_pollution))

plastic_waste_data <- plastic_waste_data %>%
    rename(country_name = 1, country_code = 2, year = 3, mismanaged_plastic_waste_share = 4) %>%
    mutate(mismanaged_plastic_waste_share = as.numeric(mismanaged_plastic_waste_share))

percentage_world_gdp_data <- percentage_world_gdp_data %>%
    rename(country_name = 1, country_code = 2, year = 3, world_gdp_share = 4) %>%
    mutate(world_gdp_share = as.numeric(world_gdp_share))

income_groups_clean <- income_groups_raw %>%
    rename(country_code = `Country Code`, income_group = IncomeGroup) %>%
    select(country_code, income_group)

# --- 4. Join All Data ---
message("Joining all data frames...")
combined_data_full <- gdp_data %>%
    full_join(imports_data, by = c("country_name", "country_code", "year")) %>%
    full_join(exports_data, by = c("country_name", "country_code", "year")) %>%
    full_join(plastic_waste_data, by = c("country_name", "country_code", "year")) %>%
    full_join(pollution_death_data, by = c("country_name", "country_code", "year")) %>%
    full_join(percentage_world_gdp_data, by = c("country_name", "country_code", "year")) %>%
    left_join(income_groups_clean, by = "country_code") %>%
    # Ensure all relevant columns are numeric and add log_gdp_per_capita
    mutate(
        across(
            c(
                gdp_per_capita, imports_gdp_ratio, exports_gdp_ratio,
                death_rate_air_pollution, mismanaged_plastic_waste_share, world_gdp_share
            ),
            ~ as.numeric(.)
        ),
        log_gdp_per_capita = log(gdp_per_capita)
    )

# --- 5. Calculate Growth Rates (from earliest non-NA year to 2019) ---
message("Calculating world GDP share growth from earliest non-NA year to 2019...")
combined_data_full <- combined_data_full %>%
    arrange(country_name, year) %>%
    group_by(country_name, country_code) %>%
    mutate(
        # Find the earliest non-NA world_gdp_share and its corresponding year
        # This ensures we pick a valid starting point for growth calculation
        earliest_valid_gdp_share = world_gdp_share[which(!is.na(world_gdp_share) & !is.na(year))][1],
        earliest_valid_year = year[which(!is.na(world_gdp_share) & !is.na(year))][1]
    ) %>%
    ungroup() %>%
    # Now calculate the growth for the 2019 data point
    group_by(country_name, country_code) %>%
    mutate(
        world_gdp_share_growth = case_when(
            year == 2019 &
                !is.na(world_gdp_share) & # Current (2019) share must not be NA
                !is.na(earliest_valid_gdp_share) & # Earliest valid share must not be NA
                (2019 - earliest_valid_year) > 0 ~ # Ensure a time difference for growth
                (world_gdp_share - earliest_valid_gdp_share) / (2019 - earliest_valid_year),
            TRUE ~ NA_real_ # Set to NA if conditions not met
        )
    ) %>%
    ungroup()

# --- 6. Filter to 2019 Data (as pollution data is only available then) ---
message("Filtering data to year 2019...")
combined_data_2019 <- combined_data_full %>%
    filter(year == 2019) %>%
    # Filter out rows with essential NAs for visualization
    filter(!is.na(death_rate_air_pollution) | !is.na(mismanaged_plastic_waste_share)) %>%
    # Ensure income_group is a factor with desired order
    filter(!is.na(income_group), income_group != "") %>%
    mutate(income_group = factor(income_group, levels = c("Low income", "Lower middle income", "Upper middle income", "High income")))

# --- 7. Calculate Residuals for Choropleth Maps ---
message("Calculating residuals for air pollution and plastic waste...")

# Model 1: Air Pollution Mortality vs log GDP
model_mortality_gdp <- lm(death_rate_air_pollution ~ log_gdp_per_capita, data = combined_data_2019)
combined_data_2019 <- combined_data_2019 %>%
    mutate(mortality_residual_gdp = residuals(model_mortality_gdp)[match(row_number(), as.numeric(names(residuals(model_mortality_gdp))))])

# Model 2: Air Pollution Mortality vs log GDP & Trade Ratios
model_mortality_gdp_trade <- lm(death_rate_air_pollution ~ log_gdp_per_capita + imports_gdp_ratio + exports_gdp_ratio, data = combined_data_2019)
combined_data_2019 <- combined_data_2019 %>%
    mutate(mortality_residual_gdp_trade = residuals(model_mortality_gdp_trade)[match(row_number(), as.numeric(names(residuals(model_mortality_gdp_trade))))])

# Model 3: Mismanaged Plastic Waste vs log GDP
model_plastic_gdp <- lm(mismanaged_plastic_waste_share ~ log_gdp_per_capita, data = combined_data_2019)
combined_data_2019 <- combined_data_2019 %>%
    mutate(plastic_residual_gdp = residuals(model_plastic_gdp)[match(row_number(), as.numeric(names(residuals(model_plastic_gdp))))])

# Determine the common range for air pollution residuals for consistent map scaling
combined_min_mortality_resid <- min(c(combined_data_2019$mortality_residual_gdp, combined_data_2019$mortality_residual_gdp_trade), na.rm = TRUE)
combined_max_mortality_resid <- max(c(combined_data_2019$mortality_residual_gdp, combined_data_2019$mortality_residual_gdp_trade), na.rm = TRUE)
max_abs_mortality_resid <- max(abs(combined_min_mortality_resid), abs(combined_max_mortality_resid), na.rm = TRUE)
scale_limits_mortality_resid <- c(-max_abs_mortality_resid, max_abs_mortality_resid)

# --- 8. Prepare Spatial Data for Maps and Add Lat/Lon/Continent ---
message("Preparing spatial data and adding lat/lon/continent coordinates...")
world_map <- ne_countries(scale = "medium", returnclass = "sf") %>%
    select(iso_a3, geometry, name, continent) %>% # Keep ISO code, geometry, country name, and continent
    filter(iso_a3 != "ATA") # Filter out Antarctica

# Add lat/lon for each country (centroid of polygon)
world_map_centroids <- world_map %>%
    st_centroid() %>%
    st_coordinates() %>%
    as.data.frame() %>%
    rename(lon = X, lat = Y)

world_map_with_coords <- cbind(world_map, world_map_centroids)

# Merge the 2019 data with map geometry and coordinates
final_data_for_viz <- world_map_with_coords %>%
    left_join(combined_data_2019, by = c("iso_a3" = "country_code")) %>%
    # Select and rename columns for clarity in the Shiny app
    select(
        country = name, # Use the full country name from rnaturalearth
        country_code = iso_a3,
        geometry,
        lat,
        lon,
        continent, # Add continent column
        income_group,
        gdp_per_capita,
        log_gdp_per_capita,
        death_rate_air_pollution,
        mismanaged_plastic_waste_share,
        world_gdp_share,
        world_gdp_share_growth, # This is the newly calculated growth
        imports_gdp_ratio,
        exports_gdp_ratio,
        mortality_residual_gdp,
        mortality_residual_gdp_trade,
        plastic_residual_gdp
    )

# --- 9. Save Pre-processed Data ---
output_file_path <- "preprocessed_pollution_data.rds"
saveRDS(final_data_for_viz, output_file_path)
message(paste("Pre-processed data saved to:", output_file_path))

# Also save the common scale limits for residuals
saveRDS(scale_limits_mortality_resid, "scale_limits_mortality_resid.rds")
message("Mortality residual scale limits saved.")

# Optional: Print structure of final data for verification
message("\nStructure of final_data_for_viz:")
print(str(final_data_for_viz))
message("\nHead of final_data_for_viz:")
print(head(final_data_for_viz))

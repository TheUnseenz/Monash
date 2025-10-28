library(plotly)
library(dplyr)
library(tibble)
library(jsonlite) # For more detailed printing of plotly object

# --- DIRECTLY ASSIGNED DATA AND INPUTS for Standalone Debugging ---

# Simulate base_plot_data (what filtered_data() would return)
set.seed(123)
dummy_countries <- paste0("Country", 1:50) # More countries for better spread
dummy_income_groups <- c("High income", "Upper middle income", "Lower middle income", "Low income")

base_plot_data_simulated <- tibble(
  country = sample(dummy_countries, 50, replace = FALSE),
  income_group = sample(dummy_income_groups, 50, replace = TRUE, prob = c(0.25, 0.35, 0.3, 0.1)),
  mismanaged_plastic_waste_share = runif(50, 0.01, 10), # Ensure values are positive and not too small
  world_gdp_share = runif(50, 0.01, 5)
)

# Introduce some NAs for filtering test (ensure some data remains)
base_plot_data_simulated$mismanaged_plastic_waste_share[c(2, 7, 15, 20, 35)] <- NA
base_plot_data_simulated$income_group[c(4, 10, 22, 40)] <- NA
base_plot_data_simulated$country[c(18, 24, 30)] <- NA
# Keep NAs for now, but ensure enough valid rows remain

# Manually ensure some values are above zero if the random generation could make them zero
base_plot_data_simulated$mismanaged_plastic_waste_share[base_plot_data_simulated$mismanaged_plastic_waste_share == 0] <- 0.001

# Simulate input values directly
input_treemap_income_toggle <- c("High income", "Upper middle income", "Lower middle income", "Low income")
input_treemap_grouping <- "income" # Set this to "income" to test the relevant branch

# Simulate income_colors_ordinal (ensure it covers all possible income groups)
income_colors_ordinal <- list(
  "High income" = "#440154",
  "Upper middle income" = "#3B528B",
  "Lower middle income" = "#21908C",
  "Low income" = "#5DC863"
)

# --- Debugging the 'income' grouping plot logic ---

# Use the directly simulated base_plot_data
plot_data <- base_plot_data_simulated %>%
  as_tibble() %>%
  filter(income_group %in% input_treemap_income_toggle) %>%
  filter(!is.na(mismanaged_plastic_waste_share),
         mismanaged_plastic_waste_share > 0, # Crucial: values must be > 0
         !is.na(income_group),
         !is.na(country),
         !is.na(world_gdp_share))

# --- DEBUGGING CHECKS ---
message("--- Debugging Report ---")
message(paste("Rows in plot_data after filtering:", nrow(plot_data)))

if (nrow(plot_data) == 0) {
  message("No data to display after filtering. Returning empty plot.")
  fig <- plot_ly() %>%
    layout(title = "No data to display for selected income groups/filters.",
           xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
           yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))
} else {
  message("Data is present. Proceeding with treemap generation.")
  
  # Check unique values in critical columns
  message(paste("Unique income_groups in plot_data:", toString(unique(plot_data$income_group))))
  message(paste("Number of unique countries in plot_data:", length(unique(plot_data$country))))
  
  # Check for NA or zero values in critical columns again (should be handled by filter but double check)
  if (any(is.na(plot_data$mismanaged_plastic_waste_share)) || any(plot_data$mismanaged_plastic_waste_share <= 0)) {
    warning("Issue: mismanaged_plastic_waste_share still has NAs or non-positive values after filter!")
    # Optionally filter again here or investigate source of issue
    plot_data <- plot_data %>% filter(!is.na(mismanaged_plastic_waste_share), mismanaged_plastic_waste_share > 0)
  }
  if (any(is.na(plot_data$income_group)) || any(plot_data$income_group == "")) {
    warning("Issue: income_group still has NAs or empty strings after filter!")
    plot_data <- plot_data %>% filter(!is.na(income_group), income_group != "")
  }
  if (any(is.na(plot_data$country)) || any(plot_data$country == "")) {
    warning("Issue: country still has NAs or empty strings after filter!")
    plot_data <- plot_data %>% filter(!is.na(country), country != "")
  }
  
  # Ensure colors can be mapped
  missing_colors <- setdiff(unique(plot_data$income_group), names(income_colors_ordinal))
  if (length(missing_colors) > 0) {
    warning(paste("Missing color definitions for income groups:", toString(missing_colors)))
    # Add a default color for missing ones to prevent error
    for (group in missing_colors) {
      income_colors_ordinal[[group]] <- "grey" # Fallback color
    }
  }
  
  # This is the 'income' grouping branch you want to debug
  message("Building plotly object...")
  fig <- plot_ly(
    data = plot_data,
    type = "treemap",
    source = "treemap_plotly_source",
    labels = ~country,          # Country names are leaf nodes
    parents = ~income_group,    # Income groups are parent nodes
    values = ~mismanaged_plastic_waste_share,
    branchvalues = "total", # Aggregates child values to parent
    marker = list(
      colors = sapply(plot_data$income_group, function(ig) income_colors_ordinal[[as.character(ig)]]),
      line = list(width = 1, color = 'white')
    ),
    textinfo = "label+value+percent parent",
    texttemplate = "<b>%{label}</b><br>%{value:.2f}%<br>%{percentParent} of group",
    hovertemplate = paste0(
      "<b>%{label}</b><br>",
      "Plastic Waste Share: %{value:.2f}%<br>",
      "World GDP Share: %{customdata:.2f}%<br>", # Use customdata for GDP share
      "%{percentParent} of group<br>",
      "<extra></extra>"
    ),
    customdata = plot_data$world_gdp_share, # Pass GDP share as customdata
    maxdepth = 2 # Show two levels (income group -> country)
  ) %>% layout(title = "Plastic Waste Share by Country, Grouped by Income Level")
  
  message("Plotly object created.")
}

# Display the plot
fig

# --- Advanced Debugging: Print Plotly JSON ---
# This can sometimes reveal more specific issues if the R object looks fine but plotly.js fails
message("\n--- Plotly JSON Representation (for advanced debugging) ---")
json_fig <- plotly_json(fig, pretty = TRUE)
print(json_fig)
writeLines(json_fig$data, "plotly_data.json")
writeLines(json_fig$layout, "plotly_layout.json")
message("--- Debugging complete ---")
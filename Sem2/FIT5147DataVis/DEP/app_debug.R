# app.R
library(shiny)
library(dplyr)
library(ggplot2)
library(plotly)
library(leaflet)
library(sf)
library(scales)
library(tidyr)
library(treemapify)
library(fmsb)
library(ggiraph)
library(forcats)  # For fct_rev
library(classInt)  # For Jenks breaks
library(shinyWidgets)  # For multi-select filters
library(ggforce) # Added for geom_parallel_sets

# Load preprocessed data
preprocessed_data <- readRDS("preprocessed_pollution_data.rds")
scale_limits_mortality_resid <- readRDS("scale_limits_mortality_resid.rds")

# Define consistent color palette for income groups
income_colors_ordinal <- c(
  "Low income" = "#C7E9C0",
  "Lower middle income" = "#A1D99B",
  "Upper middle income" = "#74C476",
  "High income" = "#238B45"
)

# Ensure income_group is properly factored
preprocessed_data <- preprocessed_data %>%
  mutate(income_group = factor(income_group, levels = names(income_colors_ordinal)))

# Get unique continents and countries
unique_continents <- sort(unique(preprocessed_data$continent))
all_countries <- sort(unique(preprocessed_data$country))

# Define map bounds to restrict dragging
world_map_bounds <- list(
  south_west = c(-60, -180), # Slightly adjusted to avoid too much white space
  north_east = c(85, 180)   # Slightly adjusted to avoid too much white space
)

# UI
ui <- fluidPage(
  titlePanel("Pollution & Development: An Interactive Narrative for Policymakers"),
  tags$style(HTML("
    .shiny-output-error { color: #ff0000; }
    .leaflet-container { background: #f0f0f0 !important; }
    .well { background-color: #f9f9f9; }
    .treemap-group { font-weight: bold; }
    .bootstrap-select .btn { background-color: white; }
  ")),
  
  fluidRow(
    column(12,
           p("This interactive visualization explores relationships between economic development, trade dynamics, and global pollution issues."),
           hr()
    )
  ),
  
  fluidRow(
    column(3,
           wellPanel(
             h4("Filter Countries"),
             selectInput("filter_income_group", "Income Group:",
                         choices = c("All" = "", levels(preprocessed_data$income_group))),
             sliderInput("filter_gdp_range", "GDP per Capita Range:",
                         min = floor(min(preprocessed_data$gdp_per_capita, na.rm = TRUE)),
                         max = ceiling(max(preprocessed_data$gdp_per_capita, na.rm = TRUE)),
                         value = c(floor(min(preprocessed_data$gdp_per_capita, na.rm = TRUE)),
                                   ceiling(max(preprocessed_data$gdp_per_capita, na.rm = TRUE))),
                         step = 1000,
                         pre = "$", sep = ","),
             pickerInput(
               inputId = "filter_continent",
               label = "Continent:", 
               choices = unique_continents,
               multiple = TRUE,
               options = list(`actions-box` = TRUE, `selected-text-format` = "count > 1")
             ),
             uiOutput("nested_country_filter")
           ),
           wellPanel(
             h4(htmlOutput("radar_chart_title")),
             plotOutput("interactive_radar_chart", height = 350)
           )
    ),
    column(9,
           tabsetPanel(
             id = "main_tabs",
             tabPanel("1. Income & Pollution",
                      h3("Income Disparity in Pollution Burden"),
                      p("Relationship between income groups, plastic waste contribution, and air pollution deaths"),
                      plotlyOutput("parallel_sets_plot", height = 500)
             ),
             tabPanel("2. Geographic Analysis",
                      h3("Geographic Distribution of Pollution Metrics"),
                      p("Spatial patterns of pollution impacts and residuals"),
                      selectInput("map_variable_selection", "Select Variable:",
                                  choices = c(
                                    "Air Pollution Residuals (vs GDP)" = "mortality_residual_gdp",
                                    "Air Pollution Residuals (vs GDP & Trade)" = "mortality_residual_gdp_trade",
                                    "Air Pollution Death Rate" = "death_rate_air_pollution",
                                    "Mismanaged Plastic Waste Share" = "mismanaged_plastic_waste_share"
                                  ),
                                  selected = "mortality_residual_gdp"),
                      leafletOutput("dynamic_choropleth_map", height = 500)
             ),
             tabPanel("3. Economic Impact",
                      h3("Plastic Waste vs Economic Share"),
                      p("Treemap visualization of countries grouped by income level"),
                      radioButtons("treemap_grouping", "Grouping:",
                                   choices = c("By Income Group" = "income", "By Country" = "country"),
                                   selected = "income", inline = TRUE),
                      girafeOutput("treemap_plot", height = "600px")
             ),
             tabPanel("4. Growth & Waste",
                      h3("Economic Growth vs Plastic Waste"),
                      p("Comparison of GDP share evolution and plastic waste contribution"),
                      plotlyOutput("growth_waste_plot", height = 600)
             ),
             tabPanel("5. Pollution Burden",
                      h3("Polluters vs Victims"),
                      p("Quadrant analysis of plastic waste contribution vs air pollution deaths"),
                      plotlyOutput("quadrant_plot", height = 500)
             )
           )
    )
  )
)

# Server
server <- function(input, output, session) {
  rv <- reactiveValues(selected_country_for_radar = NULL)
  
  # Dynamic country filter with multi-select
  output$nested_country_filter <- renderUI({
    if (is.null(input$filter_continent) || length(input$filter_continent) == 0) {
      countries_choices <- all_countries
    } else {
      countries_in_continent <- preprocessed_data %>%
        filter(continent %in% input$filter_continent) %>%
        pull(country) %>%
        unique() %>%
        sort()
      countries_choices <- countries_in_continent
    }
    
    pickerInput(
      inputId = "filter_country_nested",
      label = "Select Country(s):", 
      choices = countries_choices,
      selected = countries_choices,
      multiple = TRUE,
      options = list(`actions-box` = TRUE, `selected-text-format` = "count > 1")
    )
  })
  
  # Filter data based on inputs
  filtered_data <- reactive({
    data_filtered <- preprocessed_data
    
    if (input$filter_income_group != "") {
      data_filtered <- data_filtered %>%
        filter(income_group == input$filter_income_group)
    }
    
    data_filtered <- data_filtered %>%
      filter(gdp_per_capita >= input$filter_gdp_range[1] & 
               gdp_per_capita <= input$filter_gdp_range[2])
    
    if (!is.null(input$filter_continent) && length(input$filter_continent) > 0) {
      data_filtered <- data_filtered %>%
        filter(continent %in% input$filter_continent)
    }
    
    if (!is.null(input$filter_country_nested) && length(input$filter_country_nested) > 0) {
      data_filtered <- data_filtered %>%
        filter(country %in% input$filter_country_nested)
    }
    
    if (nrow(data_filtered) == 0) {
      return(data_filtered)
    }
    
    data_filtered
  })
  
  # Parallel sets plot - REVISED for order and error handling
  output$parallel_sets_plot <- renderPlotly({
    plot_data <- filtered_data() %>%
      as_tibble() %>%
      # Ensure data is clean for classification
      filter(!is.na(income_group), !is.na(mismanaged_plastic_waste_share), !is.na(death_rate_air_pollution))
    
    # Check if required columns exist and enough data for classification
    if (!all(c("income_group", "mismanaged_plastic_waste_share", "death_rate_air_pollution") %in% names(plot_data)) || nrow(plot_data) < 2) {
      return(NULL) # Not enough data for classIntervals or plotting
    }
    
    # Create categories using Jenks natural breaks. Handle cases where not enough unique values.
    # For plastic waste
    if (length(unique(plot_data$mismanaged_plastic_waste_share)) > 1) {
      jenks_breaks_plastic <- classIntervals(plot_data$mismanaged_plastic_waste_share, n = 4, style = "jenks")$brks
    } else {
      # Handle case with single unique value or not enough for 4 breaks
      jenks_breaks_plastic <- c(min(plot_data$mismanaged_plastic_waste_share), max(plot_data$mismanaged_plastic_waste_share) + .Machine$double.eps)
      if(length(unique(plot_data$mismanaged_plastic_waste_share)) == 1) jenks_breaks_plastic <- c(jenks_breaks_plastic[1]-0.01, jenks_breaks_plastic[1]+0.01)
    }
    
    # For air pollution
    if (length(unique(plot_data$death_rate_air_pollution)) > 1) {
      jenks_breaks_pollution <- classIntervals(plot_data$death_rate_air_pollution, n = 4, style = "jenks")$brks
    } else {
      # Handle case with single unique value or not enough for 4 breaks
      jenks_breaks_pollution <- c(min(plot_data$death_rate_air_pollution), max(plot_data$death_rate_air_pollution) + .Machine$double.eps)
      if(length(unique(plot_data$death_rate_air_pollution)) == 1) jenks_breaks_pollution <- c(jenks_breaks_pollution[1]-0.01, jenks_breaks_pollution[1]+0.01)
    }
    
    plot_data <- plot_data %>%
      mutate(
        plastic_waste_cat = cut(mismanaged_plastic_waste_share,
                                breaks = jenks_breaks_plastic,
                                labels = c("Very Low", "Low", "Medium", "High")[1:(length(jenks_breaks_plastic)-1)], # Adjust labels based on number of breaks
                                include.lowest = TRUE, right = TRUE,
                                ordered_result = TRUE), # Ensure ordered factor
        air_pollution_cat = cut(death_rate_air_pollution,
                                breaks = jenks_breaks_pollution,
                                labels = c("Very Low", "Low", "Medium", "High")[1:(length(jenks_breaks_pollution)-1)], # Adjust labels
                                include.lowest = TRUE, right = TRUE,
                                ordered_result = TRUE) # Ensure ordered factor
      ) %>%
      filter(!is.na(plastic_waste_cat), !is.na(air_pollution_cat), !is.na(income_group))
    
    if (nrow(plot_data) == 0) return(NULL)
    
    # REVISED: Convert to character before pivoting to avoid type compatibility issues
    # Store income_group separately for fill aesthetic after pivot
    plot_data_full_long <- plot_data %>%
      mutate(id = row_number(), # Unique ID for each row
             income_group_char = as.character(income_group), # Convert income group to character
             air_pollution_cat_char = as.character(air_pollution_cat), # Convert to character
             plastic_waste_cat_char = as.character(plastic_waste_cat)) %>% # Convert to character
      pivot_longer(cols = c(income_group_char, air_pollution_cat_char, plastic_waste_cat_char), # All 3 variables as character
                   names_to = "dimension_raw", values_to = "category") %>%
      mutate(
        dimension = factor(dimension_raw,
                           levels = c("income_group_char", "air_pollution_cat_char", "plastic_waste_cat_char"),
                           labels = c("Income Group", "Air Pollution Category", "Plastic Waste Category"))
      ) %>%
      select(id, dimension, category, income_group_fill = income_group) # Keep original income_group for fill
    
    # Create plot
    p <- ggplot(plot_data_full_long, 
                aes(x = dimension, id = id, split = category, 
                    # Correct tooltip for parallel sets
                    tooltip = paste("Income Group:", income_group_fill, "<br>",
                                    "Dimension:", dimension, "<br>",
                                    "Category:", category))) + 
      geom_parallel_sets(aes(fill = income_group_fill), alpha = 0.7, na.rm = TRUE) +
      geom_parallel_sets_axes(axis.width = 0.1, fill = "gray90", color = "gray30") +
      geom_parallel_sets_labels(colour = 'black', angle = 0, size = 3) + # Add labels to the axes
      scale_fill_manual(values = income_colors_ordinal) +
      labs(title = "Income Group, Air Pollution & Plastic Waste Relationships",
           fill = "Income Group") +
      theme_minimal() +
      theme(
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        panel.grid = element_blank(),
        axis.text.x = element_text(size = 11),
        plot.title = element_text(hjust = 0.5, size = 13)
      )
    
    # Convert to plotly, ensuring tooltip is correct
    ggplotly(p, tooltip = "tooltip") %>%
      layout(showlegend = TRUE)
  })
  
  # Choropleth map - FIXED with restricted bounds and correct variable selection
  output$dynamic_choropleth_map <- renderLeaflet({
    selected_var <- input$map_variable_selection
    map_title <- switch(selected_var,
                        "mortality_residual_gdp" = "Air Pollution Residuals (vs GDP)",
                        "mortality_residual_gdp_trade" = "Air Pollution Residuals (vs GDP & Trade)",
                        "death_rate_air_pollution" = "Air Pollution Death Rate",
                        "mismanaged_plastic_waste_share" = "Mismanaged Plastic Waste Share"
    )
    legend_title <- switch(selected_var,
                           "mortality_residual_gdp" = "Residual Value",
                           "mortality_residual_gdp_trade" = "Residual Value",
                           "death_rate_air_pollution" = "Deaths per 100k",
                           "mismanaged_plastic_waste_share" = "% Global Waste"
    )
    
    limits <- if (selected_var %in% c("mortality_residual_gdp", "mortality_residual_gdp_trade")) {
      scale_limits_mortality_resid
    } else {
      # Use range of the specific variable for other plots
      range(preprocessed_data[[selected_var]], na.rm = TRUE)
    }
    
    pal <- if (selected_var %in% c("mortality_residual_gdp", "mortality_residual_gdp_trade")) {
      colorNumeric("RdBu", domain = limits, reverse = TRUE, na.color = "#808080")
    } else if (selected_var == "death_rate_air_pollution") {
      colorNumeric("YlOrRd", domain = limits, na.color = "#808080")
    } else { # mismanaged_plastic_waste_share
      colorNumeric("Blues", domain = limits, na.color = "#808080")
    }
    
    map_data <- filtered_data()
    
    leaflet(map_data, 
            options = leafletOptions(maxBounds = list(world_map_bounds$south_west, world_map_bounds$north_east),
                                     minZoom = 2, maxZoom = 8, 
                                     zoomControl = TRUE # Keep zoom control
            )) %>%
      addProviderTiles(providers$CartoDB.Positron) %>%
      addPolygons(
        fillColor = ~pal(get(selected_var)),
        weight = 1,
        opacity = 1,
        color = "white",
        fillOpacity = 0.7,
        highlightOptions = highlightOptions(
          weight = 2,
          color = "#666",
          fillOpacity = 0.9,
          bringToFront = TRUE),
        label = ~paste0(country, ": ", 
                        ifelse(selected_var %in% c("mortality_residual_gdp", "mortality_residual_gdp_trade"),
                               round(get(selected_var), 2),
                               ifelse(selected_var == "mismanaged_plastic_waste_share",
                                      paste0(round(get(selected_var), 2), "%"),
                                      round(get(selected_var), 2)))),
        layerId = ~country # crucial for linking click events to country
      ) %>%
      addLegend(pal = pal, values = ~get(selected_var), 
                title = legend_title, opacity = 0.8) %>%
      setView(lng = 0, lat = 30, zoom = 2) # Initial view
  })
  
  # Update radar chart on map click
  observeEvent(input$dynamic_choropleth_map_shape_click, {
    click <- input$dynamic_choropleth_map_shape_click
    rv$selected_country_for_radar <- click$id # 'id' contains the country name from layerId
  })
  
  # Treemap plot for Tab 3 - FIXED with click events and ordering
  output$treemap_plot <- renderGirafe({
    plot_data <- filtered_data() %>%
      as_tibble() %>%
      filter(!is.na(world_gdp_share), world_gdp_share > 0, !is.na(income_group), !is.na(mismanaged_plastic_waste_share)) %>%
      arrange(desc(world_gdp_share)) # Order by largest GDP share first for better layout
    
    if (nrow(plot_data) == 0) return(NULL)
    
    # Create plot based on grouping choice
    if (input$treemap_grouping == "income") {
      # Group by income group
      p <- ggplot(plot_data, 
                  aes(area = world_gdp_share, fill = income_group,
                      subgroup = income_group, subgroup2 = country, # subgroup2 for country within income
                      tooltip = paste(country, "<br>GDP Share:", round(world_gdp_share, 2), "%<br>Plastic Waste:", round(mismanaged_plastic_waste_share, 2), "%"),
                      data_id = country)) + # data_id on country for click
        geom_treemap(color = "white", start = "topleft") +
        geom_treemap_subgroup_border(color = "black", size = 1.5) +
        geom_treemap_subgroup_text(place = "centre", grow = FALSE, alpha = 1, colour = "black",
                                   fontface = "bold", min.size = 0, padding.x = grid::unit(3, "mm"),
                                   label = "income_group") +  # Fixed label specification
        geom_treemap_text(aes(label = country),
                          colour = "white", place = "centre", reflow = TRUE, size = 10,
                          min.size = 6, padding.x = grid::unit(1, "mm"), 
                          padding.y = grid::unit(1, "mm"))
    } else {
      # No grouping - just countries
      p <- ggplot(plot_data, 
                  aes(area = world_gdp_share, fill = income_group,
                      tooltip = paste(country, "<br>GDP Share:", round(world_gdp_share, 2), "%<br>Plastic Waste:", round(mismanaged_plastic_waste_share, 2), "%"),
                      data_id = country)) + # data_id on country for click
        geom_treemap(color = "white", start = "topleft") +
        geom_treemap_text(aes(label = paste(country, "\n", round(world_gdp_share, 2), "%")),
                          colour = "white", place = "centre", reflow = TRUE, size = 10,
                          min.size = 6)
    }
    
    # Common elements
    p <- p +
      scale_fill_manual(values = income_colors_ordinal) +
      labs(fill = "Income Group") +
      theme_minimal() +
      theme(legend.position = "bottom")
    
    # Make interactive
    girafe(ggobj = p, 
           options = list(
             opts_tooltip(use_fill = TRUE, opacity = 0.9, css = "font-size:12px;"),
             opts_selection(type = "single", only_shiny = FALSE, css = "stroke:gold;stroke-width:3px;") # Ensure selection is active
           ),
           width_svg = 10, height_svg = 8)
  })
  
  # Observe treemap clicks and update radar chart
  observeEvent(input$treemap_plot_selected, {
    rv$selected_country_for_radar <- input$treemap_plot_selected
  })
  
  # Growth & Waste plot for Tab 4 - Fixed with proper ordering, tooltip, and hidden labels
  output$growth_waste_plot <- renderPlotly({
    plot_data <- filtered_data() %>%
      as_tibble() %>%
      filter(!is.na(world_gdp_share_growth), 
             !is.na(mismanaged_plastic_waste_share),
             !is.na(world_gdp_share)) %>%
      mutate(
        # Calculate past GDP share (from 1990)
        past_world_gdp_share = world_gdp_share - (world_gdp_share_growth * (2019 - 1990)), # Assumes 1990 is the base year from data_prep.R
        # Ensure no negative values
        past_world_gdp_share = pmax(past_world_gdp_share, 0)
      ) %>%
      # Order countries by plastic waste share (descending) - highest at top
      arrange(desc(mismanaged_plastic_waste_share)) %>%
      # Keep top 20 countries for readability
      head(20) %>%
      mutate(country = forcats::fct_rev(factor(country, levels = unique(country))))
    
    if (nrow(plot_data) == 0) return(NULL)
    
    # Create plot with countries ordered by plastic waste (highest at top)
    p <- ggplot(plot_data, aes(y = country)) +
      # Past GDP point with custom tooltip
      geom_point(aes(x = past_world_gdp_share, color = "Past GDP Share",
                     text = paste("Country:", country, "<br>Past GDP Share:", round(past_world_gdp_share, 1), "%")),
                 shape = 1, size = 3) +
      # Current GDP point with custom tooltip
      geom_point(aes(x = world_gdp_share, color = "Current GDP Share",
                     text = paste("Country:", country, "<br>Current GDP Share:", round(world_gdp_share, 1), "%")),
                 shape = 16, size = 4) +
      # Plastic waste point with custom tooltip
      geom_point(aes(x = mismanaged_plastic_waste_share, fill = mismanaged_plastic_waste_share, 
                     color = "Plastic Waste Share",
                     text = paste("Country:", country, "<br>Plastic Waste Share:", round(mismanaged_plastic_waste_share, 1), "%")), 
                 shape = 23, size = 5) +
      # Connect points with lines
      geom_segment(aes(x = past_world_gdp_share, xend = world_gdp_share,
                       y = country, yend = country),
                   color = "gray", linetype = "dashed") +
      geom_segment(aes(x = world_gdp_share, xend = mismanaged_plastic_waste_share,
                       y = country, yend = country),
                   color = "gray", linetype = "dashed") +
      scale_color_manual(values = c(
        "Past GDP Share" = "blue",
        "Current GDP Share" = "blue",
        "Plastic Waste Share" = "red"
      )) +
      scale_fill_gradient(low = "lightpink", high = "darkred", name = "Plastic Waste %") +
      labs(title = "Economic Share Evolution vs Plastic Waste Contribution (Top 20 Countries)",
           x = "Percentage Value",
           y = "Country",
           color = "Metric") +
      theme_minimal() +
      theme(
        axis.text.y = element_text(size = 10),
        plot.title = element_text(hjust = 0.5, size = 14),
        legend.position = "bottom"
      )
    
    # Convert to plotly with click events, using 'text' for tooltip
    ggplotly(p, tooltip = "text", source = "growth_waste_plot") %>% # Added source ID
      layout(hoverlabel = list(bgcolor = "white")) %>%
      event_register("plotly_click") # Register the event
  })
  
  # Update radar chart on growth & waste plot click - PRESERVED YOUR FIX
  observeEvent(event_data("plotly_click", source = "growth_waste_plot"), { # Added source ID
    click_data <- event_data("plotly_click", source = "growth_waste_plot")
    if (!is.null(click_data)) {
      # Get the list of countries in the current plot order
      countries_in_plot_order <- filtered_data() %>% 
        as_tibble() %>% # Ensure it's a tibble for filtering and arranging
        filter(!is.na(world_gdp_share_growth), 
               !is.na(mismanaged_plastic_waste_share),
               !is.na(world_gdp_share)) %>%
        mutate(
          past_world_gdp_share = world_gdp_share - (world_gdp_share_growth * (2019 - 1990)), 
          past_world_gdp_share = pmax(past_world_gdp_share, 0)
        ) %>%
        arrange(desc(mismanaged_plastic_waste_share)) %>%
        head(20) %>%
        pull(country) %>%
        as.character() # Ensure it's character vector for indexing
      
      # Plotly's y-axis index starts from 0 for the first element, so 0-indexed
      # R's vectors are 1-indexed. So, click_data$y (0-indexed) + 1 gives the R index.
      clicked_index <- click_data$y + 1
      
      if (clicked_index %in% seq_along(countries_in_plot_order)) {
        rv$selected_country_for_radar <- countries_in_plot_order[clicked_index]
      }
    }
  })
  
  # Quadrant plot with click event - FIXED source for event_data
  output$quadrant_plot <- renderPlotly({
    plot_data <- filtered_data() %>%
      as_tibble() %>%
      filter(!is.na(mismanaged_plastic_waste_share), !is.na(death_rate_air_pollution)) %>%
      mutate(
        tooltip_text = paste0(country, "\n",
                              "Deaths: ", round(death_rate_air_pollution, 1), "\n",
                              "Plastic Waste: ", round(mismanaged_plastic_waste_share, 2), "%")
      )
    
    if (nrow(plot_data) == 0) return(NULL)
    
    median_death <- median(plot_data$death_rate_air_pollution, na.rm = TRUE)
    median_waste <- median(plot_data$mismanaged_plastic_waste_share, na.rm = TRUE)
    
    p <- ggplot(plot_data, 
                aes(x = mismanaged_plastic_waste_share, 
                    y = death_rate_air_pollution,
                    text = tooltip_text,
                    key = country)) +  # Add key for plotly click to identify country
      geom_point(aes(color = income_group), size = 4, alpha = 0.7) +
      geom_vline(xintercept = median_waste, linetype = "dashed", color = "gray50") +
      geom_hline(yintercept = median_death, linetype = "dashed", color = "gray50") +
      annotate("text", x = max(plot_data$mismanaged_plastic_waste_share, na.rm = TRUE)*0.9, # Added na.rm
               y = median_death*1.1, label = "High Suffering", size = 4, color = "gray30") +
      annotate("text", x = max(plot_data$mismanaged_plastic_waste_share, na.rm = TRUE)*0.9, # Added na.rm
               y = median_death*0.9, label = "Low Suffering", size = 4, color = "gray30") +
      annotate("text", x = median_waste*1.1, 
               y = max(plot_data$death_rate_air_pollution, na.rm = TRUE)*0.9, # Added na.rm
               label = "High Polluting", angle = 90, size = 4, color = "gray30") +
      annotate("text", x = median_waste*0.9, 
               y = max(plot_data$death_rate_air_pollution, na.rm = TRUE)*0.9, # Added na.rm
               label = "Low Polluting", angle = 90, size = 4, color = "gray30") +
      scale_color_manual(values = income_colors_ordinal) +
      scale_x_log10() +
      labs(title = "Pollution Burden: Who Pollutes vs Who Suffers?",
           x = "Mismanaged Plastic Waste Share (log scale)",
           y = "Air Pollution Death Rate",
           color = "Income Group") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5))
    
    ggplotly(p, tooltip = "text", source = "quadrant_plot") %>% # Specify source ID
      event_register("plotly_click") # Register the event
  })
  
  # Update radar chart on quadrant plot click - FIXED source ID
  observeEvent(event_data("plotly_click", source = "quadrant_plot"), { # Use the specific source ID
    click_data <- event_data("plotly_click", source = "quadrant_plot")
    if (!is.null(click_data)) {
      rv$selected_country_for_radar <- click_data$key # 'key' contains the country name
    }
  })
  
  # Radar chart - FIXED (mostly for data handling and label consistency)
  output$radar_chart_title <- renderText({
    if (!is.null(rv$selected_country_for_radar)) {
      paste("Country Profile:", rv$selected_country_for_radar)
    } else {
      "Select a country on any visualization for details"
    }
  })
  
  output$interactive_radar_chart <- renderPlot({
    req(rv$selected_country_for_radar)
    
    country_data <- preprocessed_data %>%
      as_tibble() %>%
      filter(country == rv$selected_country_for_radar)
    
    if (nrow(country_data) == 0) {
      return(NULL)
    }
    
    # Prepare radar data
    radar_data <- country_data %>%
      select(
        `GDP per Capita` = gdp_per_capita,
        `Air Pollution Deaths` = death_rate_air_pollution,
        `Plastic Waste %` = mismanaged_plastic_waste_share,
        `Imports % GDP` = imports_gdp_ratio,
        `Exports % GDP` = exports_gdp_ratio
      ) %>%
      # Ensure all variables are numeric, convert to matrix for fmsb
      mutate_all(as.numeric)
    
    # Check for NA values in radar_data and handle if necessary
    if(any(is.na(radar_data))) {
      # Replace NA with 0 or a reasonable mid-point if 0 is misleading
      # For now, let's use 0 as a placeholder, but consider implications.
      # A better approach might be to impute or exclude axis if all are NA
      radar_data[is.na(radar_data)] <- 0 
      warning(paste("NA values present for", rv$selected_country_for_radar, "in radar data. Replaced with 0 for radar chart."))
    }
    
    # Calculate min/max for scaling from the entire preprocessed_data
    max_vals <- c(
      max(preprocessed_data$gdp_per_capita, na.rm = TRUE),
      max(preprocessed_data$death_rate_air_pollution, na.rm = TRUE),
      max(preprocessed_data$mismanaged_plastic_waste_share, na.rm = TRUE),
      max(preprocessed_data$imports_gdp_ratio, na.rm = TRUE),
      max(preprocessed_data$exports_gdp_ratio, na.rm = TRUE)
    )
    
    min_vals <- c(
      min(preprocessed_data$gdp_per_capita, na.rm = TRUE),
      min(preprocessed_data$death_rate_air_pollution, na.rm = TRUE),
      min(preprocessed_data$mismanaged_plastic_waste_share, na.rm = TRUE),
      min(preprocessed_data$imports_gdp_ratio, na.rm = TRUE),
      min(preprocessed_data$exports_gdp_ratio, na.rm = TRUE)
    )
    
    # Ensure min/max values are properly aligned with radar_data columns
    names(max_vals) <- colnames(radar_data)
    names(min_vals) <- colnames(radar_data)
    
    # Create radar chart data frame, ensuring it's a matrix for fmsb::radarchart
    radar_df <- as.data.frame(rbind(max_vals, min_vals, as.numeric(radar_data[1,]))) # Ensure single row
    colnames(radar_df) <- colnames(radar_data)
    rownames(radar_df) <- c("max", "min", "country")
    
    # Get color based on income group
    country_income <- country_data$income_group
    chart_color <- income_colors_ordinal[as.character(country_income)] # Ensure char for lookup
    
    # Create radar chart
    fmsb::radarchart(
      radar_df,
      axistype = 1,
      pcol = chart_color,
      pfcol = scales::alpha(chart_color, 0.3),
      plwd = 2,
      cglcol = "grey",
      cglty = 1,
      axislabcol = "grey",
      vlcex = 0.9,
      title = "" # Title set by output$radar_chart_title
    )
    
    legend("topright", 
           legend = paste(country_data$country, "-", country_income),
           bty = "n", pch = 20, col = chart_color, text.col = "black")
  })
}

# Run the application
shinyApp(ui = ui, server = server)
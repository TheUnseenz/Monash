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
library(ggforce) # For geom_parallel_sets (though no longer used, keeping for library load consistency if it's useful elsewhere)

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
  south_west = c(-60, -180),
  north_east = c(85, 180)
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
             tabPanel("1. Income and Pollution", # Renamed tab
                      h3("Geographic Distribution and Economic Drivers of Pollution"),
                      p("Spatial patterns of pollution impacts, residuals, and their relationship with GDP."),
                      selectInput("map_variable_selection", "Select Map Variable and Corresponding Scatter Plot:",
                                  choices = c(
                                    "Air Pollution Death Rate" = "death_rate_air_pollution",
                                    "Air Pollution Residuals (vs GDP)" = "mortality_residual_gdp",
                                    "Mismanaged Plastic Waste Share" = "mismanaged_plastic_waste_share",
                                    "Mismanaged Plastic Waste Residuals (vs GDP)" = "plastic_residual_gdp" # Added new residual
                                  ),
                                  selected = "death_rate_air_pollution"),
                      leafletOutput("dynamic_choropleth_map", height = 500),
                      tags$hr(), # Separator between map and scatter plot
                      h4(htmlOutput("scatter_plot_title")), # Dynamic title for scatter plot
                      plotlyOutput("combined_scatter_plot", height = 400) # New scatter plot output
             ),
             # Original Tab 3 is now Tab 2
             tabPanel("2. Economic Impact",
                      h3("Plastic Waste vs Economic Share"),
                      p("Treemap visualization of countries grouped by income level"),
                      radioButtons("treemap_grouping", "Grouping:",
                                   choices = c("By Income Group" = "income", "By Country" = "country"),
                                   selected = "income", inline = TRUE),
                      girafeOutput("treemap_plot", height = "600px")
             ),
             # Original Tab 4 is now Tab 3
             tabPanel("3. Growth & Waste",
                      h3("Economic Growth vs Plastic Waste"),
                      p("Comparison of GDP share evolution and plastic waste contribution"),
                      plotlyOutput("growth_waste_plot", height = 600)
             ),
             # Original Tab 5 is now Tab 4
             tabPanel("4. Pollution Burden",
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
  
  # Choropleth map - UPDATED for new variable and legend title
  output$dynamic_choropleth_map <- renderLeaflet({
    selected_var <- input$map_variable_selection
    map_title <- switch(selected_var,
                        "mortality_residual_gdp" = "Air Pollution Residuals (vs GDP)",
                        "plastic_residual_gdp" = "Mismanaged Plastic Waste Residuals (vs GDP)", # Added new residual
                        "death_rate_air_pollution" = "Air Pollution Death Rate",
                        "mismanaged_plastic_waste_share" = "Mismanaged Plastic Waste Share"
    )
    legend_title <- switch(selected_var,
                           "mortality_residual_gdp" = "Residual Value",
                           "plastic_residual_gdp" = "Residual Value", # Added new residual
                           "death_rate_air_pollution" = "Deaths per 100k",
                           "mismanaged_plastic_waste_share" = "% Global Waste"
    )
    
    limits <- if (selected_var %in% c("mortality_residual_gdp", "plastic_residual_gdp")) { # Include new residual
      # For residuals, use a common scale (if applicable) or range of filtered data
      # Let's assume plastic_residual_gdp might have its own range, or we make one common for all residuals
      # For simplicity, let's use range of filtered data for plastic_residual_gdp if distinct limits are not pre-calculated
      if (selected_var == "mortality_residual_gdp") scale_limits_mortality_resid
      else range(preprocessed_data[[selected_var]], na.rm = TRUE) # Using preprocessed for consistent scale
    } else {
      range(preprocessed_data[[selected_var]], na.rm = TRUE) # Use range of the specific variable for other plots
    }
    
    pal <- if (selected_var %in% c("mortality_residual_gdp", "plastic_residual_gdp")) { # Include new residual
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
                                     zoomControl = TRUE
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
                        ifelse(selected_var %in% c("mortality_residual_gdp", "plastic_residual_gdp"), # Updated
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
    rv$selected_country_for_radar <- click$id
  })
  
  # Scatter plot logic based on map selection
  output$scatter_plot_title <- renderText({
    selected_var <- input$map_variable_selection
    if (selected_var %in% c("death_rate_air_pollution", "mortality_residual_gdp")) {
      "Air Pollution Death Rate vs. GDP per Capita"
    } else if (selected_var %in% c("mismanaged_plastic_waste_share", "plastic_residual_gdp")) {
      "Mismanaged Plastic Waste Share vs. GDP per Capita"
    } else {
      "" # Should not happen with defined choices
    }
  })
  
  output$combined_scatter_plot <- renderPlotly({
    plot_data <- filtered_data() %>%
      as_tibble() %>%
      filter(!is.na(log_gdp_per_capita), !is.na(income_group)) # Filter NA for essential columns
    
    selected_var <- input$map_variable_selection
    
    if (nrow(plot_data) == 0) {
      return(NULL)
    }
    
    y_var_name <- ""
    y_axis_label <- ""
    
    if (selected_var %in% c("death_rate_air_pollution", "mortality_residual_gdp")) {
      plot_data <- plot_data %>% filter(!is.na(death_rate_air_pollution))
      y_var_name <- "death_rate_air_pollution"
      y_axis_label <- "Air Pollution Death Rate (per 100k)"
    } else if (selected_var %in% c("mismanaged_plastic_waste_share", "plastic_residual_gdp")) {
      plot_data <- plot_data %>% filter(!is.na(mismanaged_plastic_waste_share))
      y_var_name <- "mismanaged_plastic_waste_share"
      y_axis_label <- "Mismanaged Plastic Waste Share (%)"
    } else {
      return(NULL) # Should not happen
    }
    
    if (nrow(plot_data) == 0) return(NULL) # Check again after variable-specific filter
    
    p <- ggplot(plot_data, 
                aes(x = log_gdp_per_capita, y = .data[[y_var_name]],
                    color = income_group,
                    text = paste0("Country: ", country,
                                  "<br>GDP per Capita (log): ", round(log_gdp_per_capita, 2),
                                  "<br>", y_axis_label, ": ", round(.data[[y_var_name]], 2),
                                  "<br>Income Group: ", income_group))) +
      geom_point(alpha = 0.7, size = 3) +
      geom_smooth(method = "lm", se = FALSE, color = "darkblue", linetype = "dashed") + # Linear regression line
      scale_color_manual(values = income_colors_ordinal) +
      labs(x = "Log GDP per Capita",
           y = y_axis_label,
           color = "Income Group") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5))
    
    ggplotly(p, tooltip = "text") %>%
      layout(hoverlabel = list(bgcolor = "white"))
  })
  
  # Treemap plot for Tab 2 (formerly Tab 3) - FIXED subgroup label warning
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
                      subgroup = income_group, subgroup2 = country,
                      tooltip = paste(country, "<br>GDP Share:", round(world_gdp_share, 2), "%<br>Plastic Waste:", round(mismanaged_plastic_waste_share, 2), "%"),
                      data_id = country)) +
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
                      data_id = country)) +
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
             opts_selection(type = "single", only_shiny = FALSE, css = "stroke:gold;stroke-width:3px;")
           ),
           width_svg = 10, height_svg = 8)
  })
  
  # Observe treemap clicks and update radar chart
  observeEvent(input$treemap_plot_selected, {
    rv$selected_country_for_radar <- input$treemap_plot_selected
  })
  
  # Growth & Waste plot for Tab 3
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
    ggplotly(p, tooltip = "text", source = "growth_waste_plot") %>%
      layout(hoverlabel = list(bgcolor = "white")) %>%
      event_register("plotly_click") # Register the event
  })
  
  # Update radar chart on growth & waste plot click
  observeEvent(event_data("plotly_click", source = "growth_waste_plot"), {
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
  
  # Quadrant plot for Tab 4
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
                    key = country)) +
      geom_point(aes(color = income_group), size = 4, alpha = 0.7) +
      geom_vline(xintercept = median_waste, linetype = "dashed", color = "gray50") +
      geom_hline(yintercept = median_death, linetype = "dashed", color = "gray50") +
      scale_color_manual(values = income_colors_ordinal) +
      scale_x_log10() +
      labs(title = "Pollution Burden: Who Pollutes vs Who Suffers?",
           x = "Mismanaged Plastic Waste Share (log scale)",
           y = "Air Pollution Death Rate",
           color = "Income Group") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5))
    
    # Convert to plotly and add annotations
    plotly_plot <- ggplotly(p, tooltip = "text", source = "quadrant_plot") %>%
      layout(
        annotations = list(
          list(
            x = max(plot_data$mismanaged_plastic_waste_share, na.rm = TRUE)*0.9,
            y = median_death*1.1,
            text = "High Suffering",
            showarrow = FALSE,
            font = list(size = 12, color = "gray30")
          ),
          list(
            x = max(plot_data$mismanaged_plastic_waste_share, na.rm = TRUE)*0.9,
            y = median_death*0.9,
            text = "Low Suffering",
            showarrow = FALSE,
            font = list(size = 12, color = "gray30")
          ),
          list(
            x = median_waste*1.1,
            y = max(plot_data$death_rate_air_pollution, na.rm = TRUE)*0.9,
            text = "High Polluting",
            showarrow = FALSE,
            font = list(size = 12, color = "gray30"),
            textangle = -90
          ),
          list(
            x = median_waste*0.9,
            y = max(plot_data$death_rate_air_pollution, na.rm = TRUE)*0.9,
            text = "Low Polluting",
            showarrow = FALSE,
            font = list(size = 12, color = "gray30"),
            textangle = -90
          )
        )
      ) %>%
      event_register("plotly_click")
    
    plotly_plot
  })
  
  # Update radar chart on quadrant plot click
  observeEvent(event_data("plotly_click", source = "quadrant_plot"), {
    click_data <- event_data("plotly_click", source = "quadrant_plot")
    if (!is.null(click_data)) {
      rv$selected_country_for_radar <- click_data$key # 'key' contains the country name
    }
  })
  
  # Radar chart
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
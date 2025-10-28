# app.R

# --- 1. LOAD LIBRARIES ---
# Core libraries for Shiny app functionality, data manipulation, and plotting.
# List of required packages
required_packages <- c("shiny", "dplyr", "ggplot2", "plotly", "leaflet",
                       "sf", "scales", "fmsb", "forcats", "shinyWidgets")

# Check and install packages if they are not already installed
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

library(shiny)
library(dplyr)
library(ggplot2)
library(plotly)
library(leaflet)
library(sf)
library(scales)
library(fmsb)
library(forcats)
library(shinyWidgets)

# --- 2. INITIAL SETUP ---
# Load preprocessed data objects.
preprocessed_data <- readRDS("preprocessed_pollution_data.rds")
scale_limits_mortality_resid <- readRDS("scale_limits_mortality_resid.rds")

# Define a consistent color palette for income groups to be used across all charts.
income_colors_ordinal <- c(
  "Low income" = "#D73027",
  "Lower middle income" = "#FC8D59",
  "Upper middle income" = "#91BFDB",
  "High income" = "#1A9850"
)

# Ensure the income_group column is a factor with a defined order for plotting.
preprocessed_data <- preprocessed_data %>%
  mutate(income_group = factor(income_group, levels = names(income_colors_ordinal)))

# Get unique values for filter dropdowns.
unique_continents <- sort(unique(preprocessed_data$continent))
income_group_levels <- levels(preprocessed_data$income_group)
all_countries <- sort(unique(preprocessed_data$country))

# Define map boundaries to prevent the user from panning too far.
world_map_bounds <- list(
  south_west = c(-60, -180),
  north_east = c(85, 180)
)


# --- 3. UI DEFINITION ---
ui <- fluidPage(
  tags$style(HTML("
    /* Global sans-serif font for body text */
    body {
      font-family: 'Arial', sans-serif;
    }

    /* Serif font for all heading levels */
    h1, h2, h3, h4, h5, h6, .title {
      font-family: 'Georgia', serif;
    }
  ")),
  titlePanel("Pollution & Development: An Interactive Narrative for Policymakers"),
  
  # Custom CSS for improved aesthetics.
  tags$style(HTML("
    .shiny-output-error { color: #ff0000; }
    .leaflet-container { background: #f0f0f0 !important; }
    .well { background-color: #f9f9f9; }
    .treemap-group { font-weight: bold; }
    .bootstrap-select .btn { background-color: white; }
  ")),
  
  fluidRow(
    column(12,
           p("This interactive visualization explores relationships between economic development and global pollution issues."),
           hr()
    )
  ),
  
  fluidRow(
    # Left column: Filters and the radar chart.
    column(3,
           wellPanel(
             h4("Filter Countries"),
             pickerInput(
               inputId = "filter_income_group",
               label = "Income Group:",
               choices = income_group_levels,
               selected = income_group_levels, # Default to all selected
               multiple = TRUE,
               options = list(`actions-box` = TRUE, `selected-text-format` = "count > 1")
             ),
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
               selected = unique_continents,
               multiple = TRUE,
               options = list(`actions-box` = TRUE, `selected-text-format` = "count > 1")
             ),
             uiOutput("nested_country_filter") # Country filter dynamically updates based on continent selection.
           ),
           wellPanel(
             h4(htmlOutput("radar_chart_title")),
             plotOutput("interactive_radar_chart", height = 350)
           )
    ),
    # Right column: Main tabset panel for visualizations.
    column(9,
           tabsetPanel(
             id = "main_tabs",
             tabPanel("1. Income and Pollution",
                      h3("Geographic Distribution and Economic Drivers of Pollution"),
                      p("Spatial patterns of pollution impacts, residuals, and their relationship with GDP."),
                      selectInput("map_variable_selection", "Select Map Variable and Corresponding Scatter Plot:",
                                  choices = c(
                                    "Air Pollution Death Rate" = "death_rate_air_pollution",
                                    "Air Pollution Residuals (vs GDP)" = "mortality_residual_gdp",
                                    "Mismanaged Plastic Waste Share" = "mismanaged_plastic_waste_share",
                                    "Mismanaged Plastic Waste Residuals (vs GDP)" = "plastic_residual_gdp"
                                  ),
                                  selected = "death_rate_air_pollution"),
                      leafletOutput("dynamic_choropleth_map", height = 500),
                      tags$hr(),
                      h4(htmlOutput("scatter_plot_title")),
                      plotlyOutput("combined_scatter_plot", height = 400)
             ),
             tabPanel("2. Economic Impact",
                      h3("Plastic Waste vs Economic Share"),
                      p("Treemap visualization of countries grouped by income level"),
                      radioButtons("treemap_grouping", "Grouping:",
                                   choices = c("By Income Group" = "income", "By Country" = "country"),
                                   selected = "income", inline = TRUE),
                      plotlyOutput("treemap_plot", height = "600px")
             ),
             tabPanel("3. Growth & Waste",
                      h3("Economic Growth vs Plastic Waste"),
                      p("Comparison of GDP share evolution and plastic waste contribution"),
                      plotlyOutput("growth_waste_plot", height = 600)
             ),
             tabPanel("4. Pollution Burden",
                      h3("Polluters vs Victims"),
                      p("Quadrant analysis of plastic waste contribution vs air pollution deaths"),
                      plotlyOutput("quadrant_plot", height = 500)
             )
           )
    )
  )
)


# --- 4. SERVER LOGIC ---
server <- function(input, output, session) {
  
  # Reactive value to store the country selected from any plot for the radar chart.
  rv <- reactiveValues(selected_country_for_radar = NULL)
  
  # Renders a multi-select dropdown for countries, dynamically filtered by selected continents.
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
  
  # A reactive expression that filters the main dataset based on all user inputs.
  filtered_data <- reactive({
    data_filtered <- preprocessed_data
    
    # An empty selection means no filter is applied (i.e., all are selected).
    if (!is.null(input$filter_income_group) && length(input$filter_income_group) > 0) {
      data_filtered <- data_filtered %>%
        filter(income_group %in% input$filter_income_group)
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
  
  # Renders the main choropleth map (Leaflet).
  output$dynamic_choropleth_map <- renderLeaflet({
    selected_var <- input$map_variable_selection
    
    # Set titles and color palettes dynamically based on the selected variable.
    map_title <- switch(selected_var,
                        "mortality_residual_gdp" = "Air Pollution Residuals (vs GDP)",
                        "plastic_residual_gdp" = "Mismanaged Plastic Waste Residuals (vs GDP)",
                        "death_rate_air_pollution" = "Air Pollution Death Rate",
                        "mismanaged_plastic_waste_share" = "Mismanaged Plastic Waste Share"
    )
    legend_title <- switch(selected_var,
                           "mortality_residual_gdp" = "Residual Value",
                           "plastic_residual_gdp" = "Residual Value",
                           "death_rate_air_pollution" = "Deaths per 100k",
                           "mismanaged_plastic_waste_share" = "% Global Waste"
    )
    
    # Use pre-calculated limits for residuals for a consistent scale.
    limits <- if (selected_var %in% c("mortality_residual_gdp", "plastic_residual_gdp")) {
      if (selected_var == "mortality_residual_gdp") scale_limits_mortality_resid
      else range(preprocessed_data[[selected_var]], na.rm = TRUE)
    } else {
      range(preprocessed_data[[selected_var]], na.rm = TRUE)
    }
    
    pal <- if (selected_var %in% c("mortality_residual_gdp", "plastic_residual_gdp")) {
      colorNumeric("RdBu", domain = limits, reverse = TRUE, na.color = "#808080")
    } else if (selected_var == "death_rate_air_pollution") {
      colorNumeric("YlOrRd", domain = limits, na.color = "#808080")
    } else {
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
                        ifelse(selected_var %in% c("mortality_residual_gdp", "plastic_residual_gdp"),
                               round(get(selected_var), 2),
                               ifelse(selected_var == "mismanaged_plastic_waste_share",
                                      paste0(round(get(selected_var), 2), "%"),
                                      round(get(selected_var), 2)))),
        layerId = ~country # Set layerId to country name for click events.
      ) %>%
      addLegend(pal = pal, values = ~get(selected_var),
                title = legend_title, opacity = 0.8) %>%
      setView(lng = 0, lat = 30, zoom = 2)
  })
  
  # An observer that updates the selected country when a polygon on the map is clicked.
  observeEvent(input$dynamic_choropleth_map_shape_click, {
    click <- input$dynamic_choropleth_map_shape_click
    rv$selected_country_for_radar <- click$id
  })
  
  # Renders the title for the scatter plot, which changes based on the map's variable selection.
  output$scatter_plot_title <- renderText({
    selected_var <- input$map_variable_selection
    if (selected_var %in% c("death_rate_air_pollution", "mortality_residual_gdp")) {
      "Air Pollution Death Rate vs. GDP per Capita"
    } else if (selected_var %in% c("mismanaged_plastic_waste_share", "plastic_residual_gdp")) {
      "Mismanaged Plastic Waste Share vs. GDP per Capita"
    } else {
      ""
    }
  })
  
  # Renders the scatter plot, which is linked to the map's dropdown menu.
  output$combined_scatter_plot <- renderPlotly({
    plot_data <- filtered_data() %>%
      as_tibble() %>%
      filter(!is.na(log_gdp_per_capita), !is.na(income_group))
    
    selected_var <- input$map_variable_selection
    if (nrow(plot_data) == 0) return(NULL)
    
    # Determine which variable to plot on the y-axis based on map selection.
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
      return(NULL)
    }
    
    if (nrow(plot_data) == 0) return(NULL)
    
    p <- ggplot(plot_data,
                aes(x = log_gdp_per_capita, y = .data[[y_var_name]],
                    color = income_group,
                    text = paste0("Country: ", country,
                                  "<br>GDP per Capita (log): ", round(log_gdp_per_capita, 2),
                                  "<br>", y_axis_label, ": ", round(.data[[y_var_name]], 2),
                                  "<br>Income Group: ", income_group))) +
      geom_point(alpha = 0.7, size = 3) +
      geom_smooth(method = "lm", se = FALSE, color = "darkblue", linetype = "dashed") +
      scale_color_manual(values = income_colors_ordinal) +
      labs(x = "Log GDP per Capita",
           y = y_axis_label,
           color = "Income Group") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5))
    
    ggplotly(p, tooltip = "text") %>%
      layout(hoverlabel = list(bgcolor = "white"))
  })
  
  # Renders the treemap plot for the "Economic Impact" tab.
  output$treemap_plot <- renderPlotly({
    req(filtered_data())
    income_groups <- names(income_colors_ordinal)
    
    plot_data <- filtered_data() %>%
      as_tibble() %>%
      filter(!is.na(world_gdp_share), world_gdp_share > 0,
             !is.na(income_group), !is.na(mismanaged_plastic_waste_share)) %>%
      mutate(
        income_group = factor(income_group, levels = income_groups),
        color_value = income_colors_ordinal[as.character(income_group)],
        label = paste0(
          country, "<br>GDP Share: ", round(world_gdp_share, 2),
          "%<br>Plastic Waste: ", round(mismanaged_plastic_waste_share, 2), "%"
        ),
        id = country,
        parent = if (input$treemap_grouping == "income") as.character(income_group) else ""
      ) %>%
      arrange(desc(mismanaged_plastic_waste_share))
    
    # If grouping by income, create parent nodes for the income groups themselves.
    if (input$treemap_grouping == "income") {
      parent_sums <- plot_data %>%
        group_by(income_group) %>%
        summarise(mismanaged_plastic_waste_share = sum(mismanaged_plastic_waste_share, na.rm = TRUE)) %>%
        mutate(
          id = as.character(income_group),
          parent = "",
          label = as.character(income_group),
          country = NA,
          color_value = income_colors_ordinal[as.character(income_group)]
        )
      
      plot_data <- plot_data %>%
        mutate(value = mismanaged_plastic_waste_share)
      
      parent_nodes <- parent_sums %>%
        rename(value = mismanaged_plastic_waste_share)
      
      plot_data <- bind_rows(parent_nodes, plot_data)
    } else {
      plot_data <- plot_data %>%
        mutate(value = mismanaged_plastic_waste_share)
    }
    
    p <- plot_ly(
      data = plot_data,
      type = "treemap",
      ids = ~id,
      labels = ~label,
      parents = ~parent,
      values = ~value,
      branchvalues = 'total', # Ensures children sum up to parent total.
      hovertext = ~label,
      hoverinfo = "text",
      customdata = ~country,
      source = "treemap_plot", # Source name for linking click events.
      textinfo = "label",
      marker = list(
        colors = plot_data$color_value,
        colorscale = NULL,
        showscale = FALSE
      )
    )
    
    # A workaround to show a categorical legend for the treemap using invisible scatter points.
    legend_data <- tibble(
      income_group = factor(income_groups, levels = income_groups),
      color_value = income_colors_ordinal[income_groups]
    )
    
    p <- add_trace(
      p,
      data = legend_data,
      type = "scatter",
      mode = "markers",
      x = ~seq_along(income_group),
      y = ~seq_along(income_group),
      marker = list(
        color = ~color_value,
        size = 10
      ),
      name = ~income_group,
      legendgroup = ~income_group,
      showlegend = TRUE,
      hoverinfo = "none",
      inherit = FALSE
    ) %>%
      layout(
        showlegend = TRUE,
        xaxis = list(visible = FALSE),
        yaxis = list(visible = FALSE)
      )
    
    return(p)
  })
  
  # Observer to update radar chart when a country is clicked on the treemap.
  observeEvent(event_data("plotly_click", source = "treemap_plot"), {
    click_data <- event_data("plotly_click", source = "treemap_plot")
    if (!is.null(click_data$customdata)) {
      rv$selected_country_for_radar <- click_data$customdata
    }
  })
  
  # Renders the dumbbell plot for the "Growth & Waste" tab.
  output$growth_waste_plot <- renderPlotly({
    plot_data <- filtered_data() %>%
      as_tibble() %>%
      filter(!is.na(world_gdp_share_growth),
             !is.na(mismanaged_plastic_waste_share),
             !is.na(world_gdp_share)) %>%
      mutate(
        past_world_gdp_share = world_gdp_share - (world_gdp_share_growth * (2019 - 1990)),
        past_world_gdp_share = pmax(past_world_gdp_share, 0)
      ) %>%
      arrange(desc(mismanaged_plastic_waste_share)) %>%
      head(20) %>% # Limit to top 20 for readability.
      mutate(country = forcats::fct_rev(factor(country, levels = unique(country))))
    
    if (nrow(plot_data) == 0) return(NULL)
    
    p <- ggplot(plot_data, aes(y = country)) +
      geom_point(aes(x = past_world_gdp_share, color = "Past GDP Share",
                     text = paste("Country:", country, "<br>Past GDP Share:", round(past_world_gdp_share, 1), "%")),
                 shape = 1, size = 3) +
      geom_point(aes(x = world_gdp_share, color = "Current GDP Share",
                     text = paste("Country:", country, "<br>Current GDP Share:", round(world_gdp_share, 1), "%")),
                 shape = 16, size = 4) +
      geom_point(aes(x = mismanaged_plastic_waste_share, fill = mismanaged_plastic_waste_share,
                     color = "Plastic Waste Share",
                     text = paste("Country:", country, "<br>Plastic Waste Share:", round(mismanaged_plastic_waste_share, 1), "%")),
                 shape = 23, size = 5) +
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
    
    ggplotly(p, tooltip = "text", source = "growth_waste_plot") %>%
      layout(hoverlabel = list(bgcolor = "white")) %>%
      event_register("plotly_click")
  })
  
  # Observer to update radar chart from the growth & waste plot.
  observeEvent(event_data("plotly_click", source = "growth_waste_plot"), {
    click_data <- event_data("plotly_click", source = "growth_waste_plot")
    if (!is.null(click_data)) {
      # This logic re-calculates the plot's data to find the clicked country by its y-axis position.
      countries_in_plot_order <- filtered_data() %>%
        as_tibble() %>%
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
        as.character()
      
      clicked_index <- click_data$y + 1
      
      if (clicked_index %in% seq_along(countries_in_plot_order)) {
        rv$selected_country_for_radar <- countries_in_plot_order[clicked_index]
      }
    }
  })
  
  # Renders the quadrant scatter plot for the "Pollution Burden" tab.
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
                    key = country)) + # 'key' is used for plotly click events.
      geom_point(aes(color = income_group), size = 4, alpha = 0.7) +
      geom_vline(xintercept = median_waste, linetype = "dashed", color = "gray50") +
      geom_hline(yintercept = median_death, linetype = "dashed", color = "gray50") +
      scale_color_manual(values = income_colors_ordinal) +
      scale_x_log10() + # Using a log scale for the x-axis to better separate points.
      labs(title = "Pollution Burden: Who Pollutes vs Who Suffers?",
           x = "Mismanaged Plastic Waste Share (log scale)",
           y = "Air Pollution Death Rate",
           color = "Income Group") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5))
    
    # Add annotations for the four quadrants directly to the Plotly object.
    plotly_plot <- ggplotly(p, tooltip = "text", source = "quadrant_plot") %>%
      layout(
        annotations = list(
          list(
            x = max(plot_data$mismanaged_plastic_waste_share, na.rm = TRUE)*0.9,
            y = median_death*1.1, text = "High Suffering", showarrow = FALSE, font = list(size = 12, color = "gray30")
          ),
          list(
            x = max(plot_data$mismanaged_plastic_waste_share, na.rm = TRUE)*0.9,
            y = median_death*0.9, text = "Low Suffering", showarrow = FALSE, font = list(size = 12, color = "gray30")
          ),
          list(
            x = median_waste*1.1, y = max(plot_data$death_rate_air_pollution, na.rm = TRUE)*0.9,
            text = "High Polluting", showarrow = FALSE, font = list(size = 12, color = "gray30"), textangle = -90
          ),
          list(
            x = median_waste*0.9, y = max(plot_data$death_rate_air_pollution, na.rm = TRUE)*0.9,
            text = "Low Polluting", showarrow = FALSE, font = list(size = 12, color = "gray30"), textangle = -90
          )
        )
      ) %>%
      event_register("plotly_click")
    
    plotly_plot
  })
  
  # Observer to update radar chart when a point on the quadrant plot is clicked.
  observeEvent(event_data("plotly_click", source = "quadrant_plot"), {
    click_data <- event_data("plotly_click", source = "quadrant_plot")
    if (!is.null(click_data)) {
      rv$selected_country_for_radar <- click_data$key # The country name was stored in 'key'.
    }
  })
  
  # Renders the title for the radar chart dynamically.
  output$radar_chart_title <- renderText({
    if (!is.null(rv$selected_country_for_radar)) {
      paste("Country Profile:", rv$selected_country_for_radar)
    } else {
      "Select a country for details"
    }
  })
  
  # Renders the radar chart for the selected country.
  output$interactive_radar_chart <- renderPlot({
    req(rv$selected_country_for_radar)
    
    country_data <- preprocessed_data %>%
      as_tibble() %>%
      filter(country == rv$selected_country_for_radar)
    
    if (nrow(country_data) == 0) return(NULL)
    
    # Select and format the data for the radar chart.
    radar_data <- country_data %>%
      select(
        `GDP per Capita` = gdp_per_capita,
        `Air Pollution Deaths` = death_rate_air_pollution,
        `Plastic Waste %` = mismanaged_plastic_waste_share,
        `Imports % GDP` = imports_gdp_ratio,
        `Exports % GDP` = exports_gdp_ratio
      ) %>%
      mutate_all(as.numeric)
    
    # Handle potential NA values in the selected country's data.
    if(any(is.na(radar_data))) {
      radar_data[is.na(radar_data)] <- 0
      warning(paste("NA values present for", rv$selected_country_for_radar, "in radar data. Replaced with 0 for radar chart."))
    }
    
    # Define max and min values for each axis, scaled to the entire dataset.
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
    names(max_vals) <- colnames(radar_data)
    names(min_vals) <- colnames(radar_data)
    
    # Combine max, min, and country data into the format required by the fmsb library.
    radar_df <- as.data.frame(rbind(max_vals, min_vals, as.numeric(radar_data[1,])))
    colnames(radar_df) <- colnames(radar_data)
    rownames(radar_df) <- c("max", "min", "country")
    
    country_income <- country_data$income_group
    chart_color <- income_colors_ordinal[as.character(country_income)]
    
    # Generate the radar chart.
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
      title = "" # Title is handled by the UI output.
    )
    
    legend("topright",
           legend = paste(country_data$country, "-", country_income),
           bty = "n", pch = 20, col = chart_color, text.col = "black")
  })
}

# --- 5. RUN APPLICATION ---
shinyApp(ui = ui, server = server)
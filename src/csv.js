/**
 * HYDERABAD DUAL-SEASON PHYRES EXTRACTOR
 * Extracts 14-day windows for both Summer (April) and Winter (December)
 */

// 1. Define ROI & Load Nodes
var region = ee.Geometry.Rectangle([78.30, 17.30, 78.60, 17.50]);
// IMPORTANT: Point this to your new Asset
var nodes = ee.FeatureCollection("projects/uhi-paper-488715/assets/Hyd_Nodes"); 

Map.centerObject(region, 12);

// 2. The Master Extraction Function
function extractHybridData(era5_start, era5_end, no2_start, no2_end, l9_start, l9_end, exportName) {
  
  // A. ERA5 Weather
  var era5 = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
    .filterBounds(region)
    .filterDate(era5_start, era5_end)
    .select([
      'temperature_2m', 'u_component_of_wind_10m', 'v_component_of_wind_10m',
      'surface_net_solar_radiation_hourly', 'surface_net_thermal_radiation_hourly',
      'surface_latent_heat_flux_hourly', 'leaf_area_index_high_vegetation',
      'leaf_area_index_low_vegetation'
    ]);

  // B. Sentinel-5P Traffic Proxy
  var no2 = ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
    .filterBounds(region)
    .filterDate(no2_start, no2_end)
    .select('NO2_column_number_density')
    .mean()
    .rename('traffic_no2_proxy');

  // C. Landsat 9 (LST, NDVI, NDBI)
  var l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    .filterBounds(region)
    .filterDate(l9_start, l9_end)
    .sort('CLOUD_COVER')
    .first(); // Gets clearest image in the window

  var lst = l9.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('LST_landsat');
  var ndvi = l9.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');
  var ndbi = l9.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI');
  var landSurface = ee.Image.cat([lst, ndvi, ndbi]);

  // D. Map and Join
  var hourlyMaster = era5.map(function(image) {
    var combined = image.addBands(no2).addBands(landSurface);
    
    return combined.reduceRegions({
      collection: nodes,
      reducer: ee.Reducer.mean(),
      scale: 30
    }).map(function(f) {
      return f.set('timestamp', image.get('system:time_start'));
    });
  }).flatten();

  // E. Create Export Task
  Export.table.toDrive({
    collection: hourlyMaster,
    description: exportName,
    fileFormat: 'CSV'
  });
}

// =================================================================
// 3. EXECUTE: SUMMER RUN (April 2025)
// =================================================================
extractHybridData(
  '2025-04-01', '2025-04-15', // ERA5 Window (Target)
  '2025-03-01', '2025-04-15', // NO2 Window (Monthly context)
  '2025-01-01', '2025-04-15', // Landsat Window (Find clearest day)
  'Hyderabad_Summer_April_PhyRes'
);

// =================================================================
// 4. EXECUTE: WINTER RUN (December 2025)
// matches BLR exactly
// =================================================================
extractHybridData(
  '2025-12-02', '2025-12-16', // ERA5 Window (Target)
  '2025-11-01', '2025-12-16', // NO2 Window (Monthly context)
  '2025-10-01', '2025-12-31', // Landsat Window (Find clearest day)
  'Hyderabad_Winter_Dec_PhyRes'
);

print("✅ Tasks generated! Go to the 'Tasks' tab to start both downloads.");
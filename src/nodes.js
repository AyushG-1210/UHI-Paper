/**
 * PHASE 1: HYDERABAD NODES (1:1 BLR MIRROR)
 */
var region = ee.Geometry.Rectangle([78.30, 17.30, 78.60, 17.50]);

// 1. Create 500m Grid
var grid = region.coveringGrid(ee.Projection('EPSG:4326').atScale(500));

// 2. Load Google Open Buildings (Height & Count)
var heightImg = ee.ImageCollection('GOOGLE/Research/open-buildings-temporal/v1')
  .select('building_height').filterDate('2023-01-01', '2023-12-31').median();
var buildingsV3 = ee.FeatureCollection('GOOGLE/Research/open-buildings/v3/polygons').filterBounds(region);
var countImg = buildingsV3.reduceToImage({properties:['confidence'], reducer:ee.Reducer.count()});

// 3. Bake Building Data into Grid
var gridStats = heightImg.addBands(countImg.rename('building_count')).reduceRegions({
  collection: grid,
  reducer: ee.Reducer.mean(),
  scale: 4 
});

// 4. Format IDs and Set Baseline Properties
var gridList = gridStats.toList(gridStats.size());
var finalizedNodes = ee.FeatureCollection(gridList.map(function(f) {
  var feat = ee.Feature(f);
  return feat.set({
    'id': ee.Number(gridList.indexOf(f)).add(1).format('%d'), // Forces integer-strings
    'avg_building_height': feat.get('building_height'),
    'building_count': feat.get('building_count')
  });
}));

// 5. Export GeoJSON (For Colab/Python)
Export.table.toDrive({
  collection: finalizedNodes,
  description: 'Hyd_Nodes_PhyRes_Final',
  fileFormat: 'GeoJSON'
});

// 6. Export Asset (For Phase 2 Script)
Export.table.toAsset({
  collection: finalizedNodes,
  description: 'Hyd_Nodes_Final_Mirror',
  assetId: 'projects/uhi-paper-488715/assets/Hyd_Nodes_Final_Mirror'
});
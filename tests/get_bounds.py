import geopandas as gpd

shp = gpd.read_file(r"C:\Users\tscho\OneDrive - Scientific Network South Tyrol\Projekte\Wasserbilanz Südtirol - v2\data\watersheds_poly\Stations_watersheds.shp")
print(shp.crs)
shp_re = shp.to_crs(32632)

for nam, geom in zip(shp_re['Name'], shp_re['geometry']):
    print(nam, geom.bounds)
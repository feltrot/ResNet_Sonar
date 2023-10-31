

# script to load and test all sorts of marine data

# load libraries
library(terra)

#set working directory (wd)
wd <- "C:/Users/felix.trotter/OneDrive - The European Marine Energy Centre Ltd/Documents/Technical_Environment/CollisionRiskModelling/"

### At-Sea Density Maps for Grey and Harbour Seals in the British Isles (2020) (dataset)
# =======================================================================================
### URL: https://risweb.st-andrews.ac.uk/portal/en/datasets/atsea-density-maps-for-grey-and-harbour-seals-in-the-british-isles-2020-dataset(dcebb865-3177-4498-ac9d-13a0f10b74e1).html


# load the data
# landscape map Scotland
LSC_Scotl <- terra::vect("C:/Users/felix.trotter/OneDrive - The European Marine Energy Centre Ltd/Documents/Technical_Environment/shapefiles/LandscapeMap_Scotland/LSCMAP_SCOTLAND.shp")
# filter Orkney
Orkney <- terra::subset(LSC_Scotl, LSC_Scotl$NAME == "Orkney")
# save Orkney shapefile (will be used more often)
writeVector(Orkney, "C:/Users/felix.trotter/OneDrive - The European Marine Energy Centre Ltd/Documents/Technical_Environment/shapefiles/Orkney.shp", overwrite=TRUE)

#GreySeal_MeanCount <- terra::vect(paste0(wd, "Data/Seal_Density_Shapefiles_UTM30N_20210205/Hg_Sea_Mean.shp"))
HarbSeal_MeanCount_sixhm <- terra::rast(paste0(wd, "Data/Seal_finescale_Density_Shapefiles_Orkney/GeoTIFFs/PvSeaUsage.tif"))

# reproject harbour seal datset
HS_MeanC <- terra::project(HarbSeal_MeanCount_sixhm, crs(Orkney))
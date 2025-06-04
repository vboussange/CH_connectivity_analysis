"""Ecological Connectivity Importance (ECI) calculation for biodiversity analysis."""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(str(Path(__file__).parent / "../src/"))
from group_preprocessing import GROUP_INFO
from NSDM import NSDM_PATH
from utils_raster import load_raster, CRS_CH


class ECICalculator:
    """Calculator for Ecological Connectivity Importance."""
    
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.base_path = Path(__file__).parent / "../../data/processed" / config["hash"]
        self.eci_path = self.base_path / "ECI"
        self.ref_raster = load_raster(NSDM_PATH[25] / "Rattus.norvegicus_reg_covariate_ensemble.tif")
        
        # Create output directories
        self.eci_path.mkdir(parents=True, exist_ok=True)
        (self.eci_path / "CH3Div_per_taxa").mkdir(parents=True, exist_ok=True)
        (self.eci_path / "CH3Div_all").mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def rescale(data: xr.DataArray) -> xr.DataArray:
        """Rescale data to [0, 1] range."""
        data_min = data.min()
        data_max = data.max()
        return (data - data_min) / (data_max - data_min)
    
    @staticmethod
    def safe_add(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
        """Safely add two arrays, handling NaN values."""
        return np.where(np.isnan(a), b, np.where(np.isnan(b), a, a + b))
    
    def load_elasticities(self, habitat: str, group: str) -> Dict[str, xr.DataArray]:
        """Load elasticity rasters for a given habitat and group."""
        path_elasticities = self.base_path / "elasticities" / habitat / group
        tif_files = list(path_elasticities.glob("*.tif"))
        
        elasticities = {}
        for tif_file in tif_files:
            logger.debug(f"Reading file: {tif_file}")
            elasticities[tif_file.stem] = rioxarray.open_rasterio(tif_file)
        
        return elasticities
    
    def calculate_eci_for_group(self, habitat: str, group: str) -> xr.DataArray:
        """Calculate eci for a specific habitat and taxonomic group."""
        logger.info(f"Processing {habitat} species for group: {group}")
        
        elasticities = self.load_elasticities(habitat, group)
        
        if not elasticities:
            raise ValueError(f"No elasticity files found for {habitat}/{group}")
        
        # Sum quality and permeability elasticities
        group_summed_elasticity = elasticities[f"elasticity_quality_{group}_{habitat}"]
        
        if f"elasticity_permeability_{group}_{habitat}" in elasticities:
            group_summed_elasticity += elasticities[f"elasticity_permeability_{group}_{habitat}"]
        
        # Calculate ECI with log transformation and rescaling
        eci = self.rescale(np.log(group_summed_elasticity + 1e-5))
        
        return eci.rio.set_crs(CRS_CH).rio.reproject_match(self.ref_raster)
    
    def aggregate_groups(self, eci_by_group: Dict[str, xr.DataArray]) -> xr.DataArray:
        """Aggregate ECI across taxonomic groups."""
        group_names = list(eci_by_group.keys())
        aggregated = eci_by_group[group_names[0]].copy()
        
        for group in group_names[1:]:
            if self.config["aggregation"] == "max":
                aggregated = xr.ufuncs.fmax(aggregated, eci_by_group[group])
            elif self.config["aggregation"] == "mean":
                aggregated = xr.apply_ufunc(self.safe_add, aggregated, eci_by_group[group])
            else:
                raise ValueError(f"Unknown aggregation method: {self.config['aggregation']}")
        
        return self.rescale(aggregated)
    
    def save_raster(self, data: xr.DataArray, filepath: Path, create_plot: bool = False) -> None:
        """Save raster data and optionally create a plot."""
        logger.info(f"Saving raster to: {filepath}")
        data.rio.to_raster(str(filepath.with_suffix(".tif")), compress="zstd")
        
        if create_plot:
            self._create_plot(data, filepath.with_suffix(".png"))
    
    def _create_plot(self, data: xr.DataArray, output_path: Path) -> None:
        """Create and save a plot of the raster data."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Coarsen data for faster plotting
        plot_data = data.coarsen(x=10, y=10, boundary="trim").mean()
        
        plot_data.plot(
            ax=ax,
            cmap="magma",
            cbar_kwargs={"label": "Ecological Connectivity Importance", "shrink": 0.7}
        )
        
        ax.set_aspect("equal")
        ax.set_title("Ecological Connectivity Importance")
        ax.set_axis_off()
        
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def run_analysis(self, habitats: List[str]) -> xr.DataArray:
        """Run the complete ECI analysis pipeline."""
        eci_all = {}
        
        # Calculate ECI per taxonomic group and habitat
        for habitat in habitats:
            eci_habitat = {}
            
            for group in GROUP_INFO:
                try:
                    eci = self.calculate_eci_for_group(habitat, group)
                    
                    # Save per-group result
                    output_file = (self.eci_path / "CH3Div_per_taxa" / 
                                 f"CH3Div_ECI_unadjusted_{group}_{habitat}")
                    self.save_raster(eci, output_file)
                    
                    eci_habitat[group] = eci
                    
                except Exception as e:
                    logger.error(f"Failed to process {habitat}/{group}: {e}")
                    continue
            
            eci_all[habitat] = eci_habitat
        
        # Aggregate per habitat
        habitat_aggregated = {}
        for habitat, eci_habitat in eci_all.items():
            if not eci_habitat:
                logger.warning(f"No valid groups found for habitat: {habitat}")
                continue
                
            logger.info(f"Aggregating ECI for habitat: {habitat}")
            aggregated = self.aggregate_groups(eci_habitat)
            
            # Save habitat-aggregated result
            output_file = self.eci_path / "CH3Div_all" / f"CH3Div_ECI_unadjusted_{habitat}"
            self.save_raster(aggregated, output_file)
            
            habitat_aggregated[habitat] = aggregated
        
        # Final aggregation across all habitats
        if len(habitat_aggregated) > 1:
            logger.info("Aggregating ECI across all habitats")
            final_aggregated = self.aggregate_groups(habitat_aggregated)
        else:
            final_aggregated = list(habitat_aggregated.values())[0]
        
        # Save final result with plot
        output_file = self.eci_path / "CH3Div_all" / "CH3Div_ECI_unadjusted_all"
        self.save_raster(final_aggregated, output_file, create_plot=True)
        
        return final_aggregated


if __name__ == "__main__":
    config = {
        "hash": "3dcf5b2",
        "aggregation": "max",
    }
    
    habitats = ["aquatic", "terrestrial"]
    
    calculator = ECICalculator(config)
    result = calculator.run_analysis(habitats)
    logger.info("Analysis completed successfully")

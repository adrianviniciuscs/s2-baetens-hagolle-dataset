# Cloud Detection Dataset - Technical Documentation

## Algorithm Description

### Patch Extraction

The dataset generator implements a sliding window approach to extract fixed-size patches from large Sentinel-2 tiles and their corresponding cloud masks.

**Algorithm:**

```
For each image-mask pair:
    1. Verify spatial dimensions compatibility
    2. Calculate grid of patch positions based on patch_size and stride
    3. For each position (i, j):
        a. Extract patch from image: window = [i:i+patch_size, j:j+patch_size]
        b. Extract corresponding mask patch
        c. Compute no_data ratio in mask
        d. If no_data_ratio < threshold:
            - Save image patch
            - Save mask patch
            - Record metadata
```

**Parameters:**
- `patch_size`: Side length of square patches (pixels)
- `stride`: Step size between consecutive patches (pixels)
- `nodata_threshold`: Maximum allowed proportion of no_data pixels

**Overlap calculation:**
```
overlap = (patch_size - stride) / patch_size
```

Example: `patch_size=256`, `stride=128` gives 50% overlap.

### Image-Mask Matching

The system matches Sentinel-2 images to reference masks using filename parsing:

**Filename pattern:**
```
Sentinel-2 image: S2[A/B]_TILE_DATE.tif
Reference mask:   S2[A/B]_MSIL1C_DATETIME_*_TILE_*.tif
```

**Matching algorithm:**
1. Extract tile ID and date from image filename
2. Search mask directory for scenes containing both tile and date
3. Verify mask file exists at expected path: `{scene}/Classification/classification_map.tif`

### Quality Filtering

Patches undergo quality filtering to remove low-information samples:

**Criteria:**
- No_data pixel ratio: Patches with `>30%` no_data pixels are rejected (configurable)
- Boundary handling: Incomplete patches at image edges are discarded
- Dimension verification: Only patches with valid spatial correspondence are retained

### Train/Validation Split

Dataset partitioning uses stratified random sampling:

**Method:**
1. Extract all valid patches from all images
2. Shuffle patch list using fixed random seed
3. Split based on configured ratio (default: 80/20)

**Properties:**
- Random but reproducible (fixed seed)
- Split occurs at patch level, not image level
- Validation set proportion configurable via `val_split` parameter

## Data Format

### Image Format

Images are stored as NumPy arrays in `.npy` format:

**Specifications:**
- Shape: `(C, H, W)` where C = number of bands, H = height, W = width
- Data type: Preserve original Sentinel-2 data type (typically uint16)
- Band order: Maintains Sentinel-2 band ordering
- Coordinate system: Implicit (geographic information not preserved)

**Sentinel-2 bands:**
```
Band 1:  Coastal aerosol (443 nm) - 60m
Band 2:  Blue (490 nm) - 10m
Band 3:  Green (560 nm) - 10m
Band 4:  Red (665 nm) - 10m
Band 5:  Red Edge 1 (705 nm) - 20m
Band 6:  Red Edge 2 (740 nm) - 20m
Band 7:  Red Edge 3 (783 nm) - 20m
Band 8:  NIR (842 nm) - 10m
Band 8A: Red Edge 4 (865 nm) - 20m
Band 9:  Water vapour (945 nm) - 60m
Band 10: SWIR - Cirrus (1375 nm) - 60m
Band 11: SWIR 1 (1610 nm) - 20m
Band 12: SWIR 2 (2190 nm) - 20m
```

### Mask Format

Masks are stored as NumPy arrays in `.npy` format:

**Specifications:**
- Shape: `(H, W)`
- Data type: uint8
- Values: Integer class labels [0-7]
- No overlap: Each pixel belongs to exactly one class

**Class definitions:**

| Value | Class | Description |
|-------|-------|-------------|
| 0 | no_data | Invalid/missing data |
| 1 | not_used | Reserved (not used in practice) |
| 2 | low_clouds | Low-altitude cloud cover |
| 3 | high_clouds | High-altitude cloud cover |
| 4 | cloud_shadows | Shadow cast by clouds |
| 5 | land | Clear land surface |
| 6 | water | Water bodies |
| 7 | snow | Snow/ice cover |

## Performance Considerations

### Memory Usage

Memory requirements scale with:
- Number of images processed simultaneously
- Patch size
- Number of spectral bands

**Estimation:**
```
Memory per patch ≈ patch_size² × num_bands × bytes_per_pixel
Example: 256² × 13 × 2 bytes ≈ 1.7 MB per patch
```

### Processing Time

Processing time depends on:
- Image dimensions
- Patch size and stride (determines number of patches)
- I/O speed
- Quality filtering (reduces output but increases processing)

**Typical rates:**
- Patch extraction: ~100-500 patches/second (SSD storage)
- Full tile processing: 1-5 minutes per 10980×10980 image

### Disk Space

Output size estimation:
```
Dataset size = n_patches × (image_patch_size + mask_patch_size)

Example:
- 1000 patches
- 256×256 patches
- 13 bands
- uint16 images, uint8 masks
= 1000 × (256² × 13 × 2 + 256² × 1) bytes
≈ 2.3 GB
```

## Error Handling

The system implements robust error handling:

**Common issues:**
1. **Missing mask**: Image skipped, warning logged
2. **Dimension mismatch**: Processes intersection of valid areas
3. **Corrupt file**: Exception caught, image skipped, error logged
4. **No valid patches**: Warning logged, no output generated

**Logging levels:**
- ERROR: Critical failures preventing processing
- WARNING: Non-critical issues (missing masks, dimension mismatches)
- INFO: Normal processing progress
- DEBUG: Detailed diagnostic information

## Reproducibility

The system ensures reproducible results through:

1. **Fixed random seed**: Controls train/val split randomization
2. **Deterministic sorting**: Image processing order is consistent
3. **Configuration versioning**: All parameters stored in output summary

**Best practices:**
- Always specify `random_seed` in configuration
- Document configuration file version
- Record system information (Python version, library versions)

## Validation

Dataset quality can be verified through:

1. **Patch count verification**: Compare expected vs actual patch counts
2. **Class distribution analysis**: Verify mask class balance
3. **Spatial continuity**: Check patches from same image maintain coherence
4. **Visual inspection**: Sample visualization of image-mask pairs

The system generates `dataset_summary.txt` containing verification information.


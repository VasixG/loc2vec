# LOC2VEC

This project processes OpenStreetMap (OSM) data to generate images based on geographic locations (latitude/longitude), and uses a Triplet Neural Network to train embeddings based on these images as in word2vec. You can run the project either using Docker or directly from the command line.

---

## **Table of Contents**
1. [Requirements](#requirements)
2. [How to Use](#how-to-use)
    - [With Docker](#with-docker)
    - [Without Docker (Command Line)](#without-docker)
3. [Configuration Parameters](#configuration-parameters)


---

## **Requirements**

### Dependencies:

- Python 3.9 or later
- Required libraries (if not using Docker):
  - `geopandas`
  - `numpy`
  - `matplotlib`
  - `osmnx`
  - `shapely`
  - `Pillow`
  - `srai`
  - `pyyaml`
  - `geopy`
  - `torch`
  - `scikit-learn`

You can install the required libraries with:

```bash
pip install -r requirements.txt
```
## **How to Use**

### **With Docker**

1. **Build the Docker image**:
   ```bash
   docker compose build 
   ```
2. **Run the Docker image**:
   ```bash
   docker compose up
   ```

### **With CMD**

1. Install dependecies
```bash
pip install -r requirements.txt
```
2. Run the preprocessing step:
```bash
python src/preprocessing.py --config preprocessing_config.yaml
```
3. Run the training step:
```bash
python src/training.py --config training_config.yaml
```

## **Preprocessing configuration Parameters**
## **Preprocessing Configuration Parameters**

| Parameter      | Type   | Default          | Description                                                                 |
|----------------|--------|------------------|-----------------------------------------------------------------------------|
| `location`     | string | "Moscow, Russia" | The geographic location from which to extract OSM data.                     |
| `grid_spacing` | int    | 1000             | Spacing (in meters) between grid points.                                    |
| `poi_tags`     | dict   | `{amenity: [...], railway: [...], highway: [...]}` | Specifies which OSM POI tags to load (e.g., `amenity`, `railway`, `highway`). |
| `tag_groups`   | list   | `[amenity, railway, highway]` | Groups of tags to organize images for triplet creation.                       |

### Example Preprocessing `config.yaml`

```yaml
location: "Moscow, Russia"
grid_spacing: 1000
poi_tags:
  amenity: 
    - school
    # - hospital
    # - restaurant
    # - cafe
    # - bank
  railway: 
    - rail
    - subway
  highway: 
    - motorway
    # - trunk
    # - primary
tag_groups:
  - 
    - amenity_school
    # - amenity_hospital
    # - amenity_restaurant
    # - amenity_cafe
    # - amenity_bank
  -
    - railway_rail
    - railway_subway
  -
    - highway_motorway
    # - highway_trunk
    # - highway_primary
```

## **Training Configuration Parameters**

The configuration for both the preprocessing and training processes is stored in `training_config.yaml`. Below is a table of the configuration parameters:

| Parameter        | Type    | Default | Description                                                                 |
|------------------|---------|---------|-----------------------------------------------------------------------------|
| `n_channels`     | int     | 3       | Number of channels( images with tags data )    |
| `batch_size`     | int     | 128     | Batch size used for training the Triplet Network.                           |
| `lr`             | float   | 0.001   | Learning rate for the optimizer.                                            |
| `margin`         | float   | 0.0005  | Margin value for the Triplet Loss function.                                 |
| `n_epochs`       | int     | 3       | Number of epochs to train the model.                                        |
| `scheduler_step` | int     | 8       | Step size for learning rate decay in the scheduler.                         |
| `scheduler_gamma`| float   | 0.1     | Multiplicative factor for learning rate decay in the scheduler.             |
| `log_interval`   | int     | 100     | Interval for logging the training progress.                                 |
| `radius_near`    | int     | 3000    | Radius (in meters) to find nearby points for triplet creation.              |
| `radius_far`     | int     | 5000    | Radius (in meters) to find faraway points for triplet creation.             |
| `test_size`      | float   | 0.2     | Fraction of data to be used for testing.                                    |
| `random_seed`    | int     | 42      | Random seed for reproducibility.                                            |


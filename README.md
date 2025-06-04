# Reproducibility Repository – On the power of CNNs to detect slums in Brazil

This repository contains the code for reproducing the experiments presented in the following article published in *Computers, Environment and Urban Systems*:

> **"On the power of CNNs to detect slums in Brazil"**  
> [https://www.sciencedirect.com/science/article/pii/S0198971525000596](https://www.sciencedirect.com/science/article/pii/S0198971525000596)

## Data

Reference data for subnormal agglomerates (AGNS) was collected from the Brazilian Institute of Geography and Statistics (IBGE) and is available at:

🔗 [https://www.ibge.gov.br/geociencias/organizacao-do-territorio/tipologias-do-territorio/15788-favelas-e-comunidades-urbanas.html?edicao=27720&t=acesso-ao-produto](https://www.ibge.gov.br/geociencias/organizacao-do-territorio/tipologias-do-territorio/15788-favelas-e-comunidades-urbanas.html?edicao=27720&t=acesso-ao-produto)

## Pre-processing

The `pre_processing/` folder contains the scripts used to acquire the satellite images:

- `download_sentinel_2_ee.ipynb`:  
  Downloads Sentinel-2 satellite images (10m spatial resolution) using [Google Earth Engine](https://earthengine.google.com/).  
  Preprocessed images are available at:  
  🔗 [https://data.mendeley.com/preview/xg7p7rfrrp?a=e3fbc6b4-0007-40c7-8e23-46f54f8d26a4](https://data.mendeley.com/preview/xg7p7rfrrp?a=e3fbc6b4-0007-40c7-8e23-46f54f8d26a4)

- `download_google_images.ipynb`:  
  Downloads very high resolution (VHR) images from Google Maps using the Google Maps Static API.  
  ⚠️ You must include your own Google Maps API key to use this script.  
  ⚠️ Due to licensing restrictions, the downloaded Google images are **not provided** in this repository.

## Models and Experiments

The `models/` folder contains the deep learning code used for training and evaluation:

- `train_city_model.py`: Trains a separate model for each city.
- `evaluate_cross_city.py`: Evaluates each city-trained model on the remaining cities.
- `all_cities.py`: Trains a single model using data from all cities and evaluates it.
- `all_cities_one-leave-out.py`: Trains using data from 5 cities and evaluates on the one left out (leave-one-city-out setup).

## Citation

If you use this code in your research, please cite the original article.

> da Silva, J. P., Rodrigues-Jr, J. F., & de Albuquerque, J. P. (2025). *On the power of CNNs to detect slums in Brazil*. Computers, Environment and Urban Systems, 121, 102306.  
> [https://doi.org/10.1016/j.compenvurbsys.2025.102306](https://doi.org/10.1016/j.compenvurbsys.2025.102306)

Dataset citation:

> da Silva, João Pedro (2025), “High-Resolution Sentinel-2 Images of Brazilian in 2024”, Mendeley Data, V2, doi: 10.17632/xg7p7rfrrp.2  
> [https://doi.org/10.17632/xg7p7rfrrp.1](https://doi.org/10.17632/xg7p7rfrrp.1)

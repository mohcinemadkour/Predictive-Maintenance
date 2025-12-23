# Running the Streamlit Dashboard

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the App

To run the Streamlit dashboard, execute the following command from the project directory:

```bash
streamlit run app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

## Features

The Streamlit app includes:

- **Dataset Overview**: View basic statistics and raw data
- **Feature Statistics**: Visualize standard deviation and variance
- **Correlation Analysis**: Explore correlations with Time to Failure (TTF)
- **Interactive Feature Explorer**: Select and visualize specific features
- **Correlation Matrix**: Generate heatmaps for selected features
- **Data Export**: Download processed data as CSV

## Troubleshooting

If you encounter any issues:

1. Make sure you're in the correct directory
2. Verify that `data/train.csv` exists
3. Ensure all dependencies are installed: `pip install -r requirements.txt`

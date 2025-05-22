# Credit Card Fraud Detection - Interactive Analysis

This project provides an interactive web application for understanding credit card fraud detection through data visualization and analysis. It combines insights from machine learning models and data analysis to create an educational experience about fraud detection in financial transactions.

## Features

- Interactive visualizations of transaction data
- Real-time fraud risk prediction
- Comprehensive data analysis
- Model performance metrics
- Responsive design for all devices

## Project Structure

```
credit-card-fraud/
├── index.html              # Main HTML file
├── styles/
│   └── main.css           # CSS styles
├── scripts/
│   ├── visualizations.js  # Visualization logic
│   ├── model.js          # Model interaction logic
│   └── process_data.py   # Data processing script
├── data/
│   ├── processed_data.json    # Processed dataset
│   ├── model_metrics.json     # Model performance metrics
│   └── roc_curve.png         # ROC curve visualization
└── assets/
    └── images/            # Additional images
```

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/gamechaser782/gamechaser782.github.io.git
   cd gamechaser782.github.io/projects/credit-card-fraud
   ```

2. Install Python dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```

3. Download the dataset:
   - Download the credit card fraud dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   - Place the `creditcard.csv` file in the project root directory

4. Process the data:
   ```bash
   python scripts/process_data.py
   ```

5. Serve the website:
   - The website can be served using any static file server
   - For local development, you can use Python's built-in server:
     ```bash
     python -m http.server 8000
     ```
   - Then visit `http://localhost:8000` in your browser

## Usage

1. **Data Analysis Section**
   - View key statistics about the dataset
   - Explore transaction distributions
   - Analyze feature correlations

2. **Visualizations**
   - Interactive correlation heatmap
   - Transaction amount distribution
   - Time-based analysis of transactions
   - Fraud vs. normal transaction patterns

3. **Fraud Detection Model**
   - View model performance metrics
   - Try the model with sample transactions
   - Understand risk factors

## Technologies Used

- HTML5, CSS3, JavaScript
- Bootstrap 5 for responsive design
- D3.js and Plotly.js for visualizations
- Python for data processing
- scikit-learn for model metrics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Credit Card Fraud Dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Insights from the [Fraud Detection Handbook](https://fraud-detection-handbook.github.io/fraud-detection-handbook/)
- Temporian and Keras for the original model implementation 
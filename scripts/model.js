class FraudDetectionModel {
    constructor() {
        this.model = null;
        this.modelMetrics = null;
        this.riskParameters = null;
        this.initializeModel();
        this.setupEventListeners();
    }

    async initializeModel() {
        try {
            // Load both model metrics and processed data for risk parameters
            const [metricsResponse, dataResponse] = await Promise.all([
                fetch('data/model_metrics.json'),
                fetch('data/processed_data.json')
            ]);
            
            this.modelMetrics = await metricsResponse.json();
            const processedData = await dataResponse.json();
            this.riskParameters = processedData.risk_parameters;
            
            this.displayModelMetrics();
        } catch (error) {
            console.error('Error loading model data:', error);
        }
    }

    displayModelMetrics() {
        const metrics = this.modelMetrics;
        const metricsHtml = `
            <div class="metrics-grid">
                <div class="metric-item">
                    <h4>AUC Score</h4>
                    <p>${metrics.auc_score.toFixed(3)}</p>
                </div>
                <div class="metric-item">
                    <h4>Precision</h4>
                    <p>${metrics.precision.toFixed(3)}</p>
                </div>
                <div class="metric-item">
                    <h4>Recall</h4>
                    <p>${metrics.recall.toFixed(3)}</p>
                </div>
                <div class="metric-item">
                    <h4>F1 Score</h4>
                    <p>${metrics.f1_score.toFixed(3)}</p>
                </div>
            </div>
            <div class="roc-curve">
                <img src="data/roc_curve.png" alt="ROC Curve" class="img-fluid">
            </div>
        `;
        document.getElementById('model-metrics').innerHTML = metricsHtml;
    }

    setupEventListeners() {
        const form = document.getElementById('prediction-form');
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.handlePrediction();
        });
    }

    async handlePrediction() {
        const amount = document.getElementById('amount').value;
        const time = document.getElementById('time').value;
        
        // Convert time to hour of day (0-23)
        const [hours] = time.split(':').map(Number);
        
        try {
            // In a real application, this would be an API call to your backend
            // For this demo, we'll simulate a prediction
            const prediction = await this.simulatePrediction(amount, hours);
            this.displayPrediction(prediction);
        } catch (error) {
            console.error('Error making prediction:', error);
            this.displayError('Error making prediction. Please try again.');
        }
    }

    async simulatePrediction(amount, hour) {
        // Use the risk parameters from the data analysis
        const { amount_thresholds, time_risk_factors, hourly_fraud_rates } = this.riskParameters;
        
        let riskScore = 0;
        
        // Amount-based risk (using actual data quantiles)
        if (amount > amount_thresholds.high) {
            riskScore += 0.4;  // High amount risk
        } else if (amount > amount_thresholds.medium) {
            riskScore += 0.2;  // Medium amount risk
        } else if (amount > amount_thresholds.low) {
            riskScore += 0.1;  // Low amount risk
        }
        
        // Time-based risk (using actual fraud rates by hour)
        const hourFraudRate = hourly_fraud_rates[hour];
        riskScore += hourFraudRate * 0.3;  // Weight the hourly fraud rate
        
        // Additional time-based risk factors
        if (time_risk_factors.high_risk_hours.includes(hour)) {
            riskScore += 0.15;  // Higher risk during early morning
        } else if (time_risk_factors.medium_risk_hours.includes(hour)) {
            riskScore += 0.05;  // Moderate risk during afternoon
        }
        
        // Add small random variation (max 5%)
        riskScore += (Math.random() - 0.5) * 0.05;
        
        // Normalize risk score to [0,1] range
        riskScore = Math.min(1, Math.max(0, riskScore));
        
        // Calculate confidence based on how far the score is from the decision boundary
        const confidence = Math.abs(riskScore - 0.5) * 2;
        
        return {
            riskScore: riskScore,
            isFraud: riskScore > 0.6,  // Higher threshold for fraud detection
            confidence: confidence
        };
    }

    displayPrediction(prediction) {
        const resultDiv = document.getElementById('prediction-result');
        const riskColor = prediction.isFraud ? '#e74c3c' : '#2ecc71';
        const riskText = prediction.isFraud ? 'High Risk' : 'Low Risk';
        const riskLevel = prediction.riskScore > 0.8 ? 'Very High' :
                         prediction.riskScore > 0.6 ? 'High' :
                         prediction.riskScore > 0.4 ? 'Medium' :
                         prediction.riskScore > 0.2 ? 'Low' : 'Very Low';
        
        const html = `
            <div class="prediction-result" style="border-left: 4px solid ${riskColor}">
                <h4>Prediction Result</h4>
                <p><strong>Risk Level:</strong> ${riskLevel}</p>
                <p><strong>Risk Score:</strong> ${(prediction.riskScore * 100).toFixed(1)}%</p>
                <p><strong>Confidence:</strong> ${(prediction.confidence * 100).toFixed(1)}%</p>
                <div class="risk-meter">
                    <div class="risk-bar" style="width: ${prediction.riskScore * 100}%; background-color: ${riskColor}"></div>
                </div>
                <div class="risk-explanation mt-3">
                    <small class="text-muted">
                        ${this.getRiskExplanation(prediction.riskScore, riskLevel)}
                    </small>
                </div>
            </div>
        `;
        
        resultDiv.innerHTML = html;
    }

    getRiskExplanation(riskScore, riskLevel) {
        const explanations = {
            'Very High': 'This transaction shows multiple high-risk indicators. Manual review recommended.',
            'High': 'This transaction has several risk factors. Additional verification may be needed.',
            'Medium': 'This transaction has some risk factors. Standard verification procedures apply.',
            'Low': 'This transaction shows minimal risk factors. Standard processing recommended.',
            'Very Low': 'This transaction shows very low risk. Standard processing recommended.'
        };
        return explanations[riskLevel] || 'Standard processing recommended.';
    }

    displayError(message) {
        const resultDiv = document.getElementById('prediction-result');
        resultDiv.innerHTML = `
            <div class="alert alert-danger" role="alert">
                ${message}
            </div>
        `;
    }
}

// Initialize the model when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const model = new FraudDetectionModel();
}); 
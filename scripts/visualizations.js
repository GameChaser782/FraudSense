// Data processing and visualization functions
class FraudVisualizations {
    constructor() {
        this.data = null;
        this.loadData();
    }

    async loadData() {
        try {
            // Load the processed data - updated path for correct directory structure
            const response = await fetch('data/processed_data.json');
            this.data = await response.json();
            this.initializeVisualizations();
        } catch (error) {
            console.error('Error loading data:', error);
        }
    }

    initializeVisualizations() {
        this.createCorrelationHeatmap();
        this.createTransactionDistribution();
        this.createAmountDistribution();
        this.createTimeAnalysis();
        this.updateDatasetStats();
    }

    createCorrelationHeatmap() {
        const correlations = this.data.correlations;
        const features = Object.keys(correlations);
        
        const data = [{
            z: features.map(f1 => 
                features.map(f2 => correlations[f1][f2])
            ),
            x: features,
            y: features,
            type: 'heatmap',
            colorscale: [
                [0, '#3498db'],    // Negative correlations
                [0.5, '#ffffff'],  // No correlation
                [1, '#e74c3c']     // Positive correlations
            ],
            zmin: -1,
            zmax: 1,
            showscale: true,
            colorbar: {
                title: 'Correlation',
                titleside: 'right'
            }
        }];

        const layout = {
            title: 'Feature Correlations',
            height: 700,
            margin: { t: 50, b: 100, l: 100, r: 50 },
            xaxis: { 
                tickangle: 45,
                tickfont: { size: 10 }
            },
            yaxis: { 
                autorange: 'reversed',
                tickfont: { size: 10 }
            },
            plot_bgcolor: 'white',
            paper_bgcolor: 'white'
        };

        Plotly.newPlot('correlation-heatmap', data, layout);
    }

    createTransactionDistribution() {
        const fraudCount = this.data.transaction_counts.fraud;
        const normalCount = this.data.transaction_counts.normal;

        const data = [{
            values: [normalCount, fraudCount],
            labels: ['Normal', 'Fraudulent'],
            type: 'pie',
            hole: 0.4,
            marker: {
                colors: ['#3498db', '#e74c3c']
            }
        }];

        const layout = {
            title: 'Transaction Distribution',
            height: 400,
            showlegend: true,
            legend: {
                orientation: 'h',
                y: -0.1
            }
        };

        Plotly.newPlot('transaction-dist', data, layout);
    }

    createAmountDistribution() {
        const amounts = this.data.amount_distribution;
        
        // Create histogram with log scale
        const data = [{
            x: amounts.values,
            type: 'histogram',
            name: 'Transaction Amount (log scale)',
            marker: {
                color: '#3498db',
                line: {
                    color: 'white',
                    width: 1
                }
            },
            opacity: 0.7
        }];

        const layout = {
            title: 'Transaction Amount Distribution (Log Scale)',
            xaxis: { 
                title: 'Log(Amount + 1)',
                showgrid: true,
                gridcolor: '#f0f0f0'
            },
            yaxis: { 
                title: 'Frequency',
                showgrid: true,
                gridcolor: '#f0f0f0'
            },
            height: 400,
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            margin: { t: 50, b: 50, l: 50, r: 20 }
        };

        Plotly.newPlot('amount-dist', data, layout);
    }

    createTimeAnalysis() {
        const timeData = this.data.time_analysis;
        
        // Create line plot with area fill
        const data = [{
            x: timeData.hours,
            y: timeData.fraud_counts,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Fraudulent Transactions',
            line: { 
                color: '#e74c3c',
                width: 2
            },
            marker: {
                size: 6,
                color: '#e74c3c'
            },
            fill: 'tonexty'
        }, {
            x: timeData.hours,
            y: timeData.normal_counts,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Normal Transactions',
            line: { 
                color: '#3498db',
                width: 2
            },
            marker: {
                size: 6,
                color: '#3498db'
            },
            fill: 'tonexty'
        }, {
            x: timeData.hours,
            y: timeData.fraud_rates.map(rate => rate * 100),
            type: 'scatter',
            mode: 'lines',
            name: 'Fraud Rate (%)',
            line: {
                color: '#2ecc71',
                width: 2,
                dash: 'dot'
            },
            yaxis: 'y2'
        }];

        const layout = {
            title: 'Transaction Frequency and Fraud Rate by Hour',
            xaxis: { 
                title: 'Hour of Day',
                range: [0, 23],
                showgrid: true,
                gridcolor: '#f0f0f0'
            },
            yaxis: { 
                title: 'Number of Transactions',
                showgrid: true,
                gridcolor: '#f0f0f0'
            },
            yaxis2: {
                title: 'Fraud Rate (%)',
                overlaying: 'y',
                side: 'right',
                showgrid: false,
                range: [0, Math.max(...timeData.fraud_rates) * 100 * 1.1]
            },
            height: 500,
            showlegend: true,
            legend: {
                x: 1.1,
                y: 1
            },
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            margin: { t: 50, b: 50, l: 50, r: 80 }
        };

        Plotly.newPlot('time-analysis', data, layout);
    }

    updateDatasetStats() {
        const stats = this.data.dataset_stats;
        const statsHtml = `
            <div class="stats-grid">
                <div class="stat-item">
                    <h4>Total Transactions</h4>
                    <p>${stats.total_transactions.toLocaleString()}</p>
                </div>
                <div class="stat-item">
                    <h4>Fraud Rate</h4>
                    <p>${(stats.fraud_rate * 100).toFixed(3)}%</p>
                </div>
                <div class="stat-item">
                    <h4>Average Amount</h4>
                    <p>$${stats.avg_amount.toFixed(2)}</p>
                </div>
                <div class="stat-item">
                    <h4>Median Amount</h4>
                    <p>$${stats.median_amount.toFixed(2)}</p>
                </div>
                <div class="stat-item">
                    <h4>Max Amount</h4>
                    <p>$${stats.max_amount.toFixed(2)}</p>
                </div>
                <div class="stat-item">
                    <h4>Amount Std Dev</h4>
                    <p>$${stats.std_amount.toFixed(2)}</p>
                </div>
            </div>
        `;
        document.getElementById('dataset-stats').innerHTML = statsHtml;
    }
}

// Initialize visualizations when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const visualizations = new FraudVisualizations();
});

// Add smooth scrolling for navigation
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
}); 
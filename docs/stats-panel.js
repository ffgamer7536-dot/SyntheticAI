/**
 * Statistics Panel Component
 * Displays per-class metrics and inference statistics
 */

const CLASS_NAMES = [
    'Trees',
    'Lush Bushes',
    'Dry Grass',
    'Dry Bushes',
    'Ground Clutter',
    'Flowers',
    'Logs',
    'Rocks',
    'Landscape',
    'Sky'
];

const CLASS_COLORS_RGB = {
    0: 'rgb(0, 255, 0)',
    1: 'rgb(34, 139, 34)',
    2: 'rgb(255, 255, 0)',
    3: 'rgb(210, 180, 140)',
    4: 'rgb(165, 42, 42)',
    5: 'rgb(255, 20, 147)',
    6: 'rgb(160, 82, 45)',
    7: 'rgb(128, 128, 128)',
    8: 'rgb(0, 128, 255)',
    9: 'rgb(0, 0, 255)',
};

class StatsPanel {
    constructor() {
        this.lastMetrics = null;
    }

    /**
     * Display inference results in stats panel
     */
    displayResults(results, containerSelector = '#statsPanel') {
        const container = document.querySelector(containerSelector);
        if (!container) return;

        this.lastMetrics = results.metrics;

        // Update inference time
        const timeDisplay = document.querySelector('#inferenceTime');
        if (timeDisplay) {
            timeDisplay.textContent = results.inference_time_ms.toFixed(2);
        }

        // Update coverage chart
        this.displayCoverageChart(results.coverage);

        // Update metrics table
        this.displayMetricsTable(results.metrics);

        // Update mean IoU
        this.updateMeanIoU(results.metrics.mean_iou);
    }

    /**
     * Display per-class coverage as a visual chart
     */
    displayCoverageChart(coverage) {
        const chartContainer = document.getElementById('coverageChart');
        if (!chartContainer) return;

        chartContainer.innerHTML = '';

        // Sort coverage by value and get top classes
        const sorted = Object.entries(coverage)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 10);

        sorted.forEach(([classIdx, percent]) => {
            const idx = parseInt(classIdx);
            const className = CLASS_NAMES[idx];
            const color = CLASS_COLORS_RGB[idx];

            const item = document.createElement('div');
            item.className = 'coverage-item';
            item.style.borderLeftColor = color;

            const name = document.createElement('span');
            name.className = 'coverage-item-name';
            name.textContent = className;

            const value = document.createElement('span');
            value.className = 'coverage-item-value';
            value.textContent = `${percent.toFixed(1)}%`;

            item.appendChild(name);
            item.appendChild(value);
            chartContainer.appendChild(item);
        });
    }

    /**
     * Display per-class metrics in table
     */
    displayMetricsTable(metrics) {
        const tbody = document.getElementById('metricsBody');
        if (!tbody) return;

        tbody.innerHTML = '';

        const coverage = metrics.per_class_coverage || {};
        const iouValues = metrics.per_class_iou || {};

        CLASS_NAMES.forEach((name, idx) => {
            const row = document.createElement('tr');

            // Class name cell
            const nameCell = document.createElement('td');
            nameCell.style.display = 'flex';
            nameCell.style.alignItems = 'center';
            nameCell.style.gap = '8px';

            const colorBox = document.createElement('div');
            colorBox.style.width = '16px';
            colorBox.style.height = '16px';
            colorBox.style.borderRadius = '2px';
            colorBox.style.backgroundColor = CLASS_COLORS_RGB[idx];

            const nameSpan = document.createElement('span');
            nameSpan.textContent = name;

            nameCell.appendChild(colorBox);
            nameCell.appendChild(nameSpan);
            row.appendChild(nameCell);

            // Coverage cell
            const coverageCell = document.createElement('td');
            const coverageVal = coverage[name] !== undefined ? coverage[name] : 0;
            coverageCell.textContent = `${coverageVal.toFixed(2)}%`;
            row.appendChild(coverageCell);

            // IoU cell
            const iouCell = document.createElement('td');
            const iouVal = iouValues[name];
            if (iouVal !== null && iouVal !== undefined) {
                iouCell.textContent = iouVal ? `${(iouVal * 100).toFixed(2)}%` : 'N/A';
                iouCell.style.color = iouVal > 0.5 ? '#10b981' : '#f87171';
            } else {
                iouCell.textContent = 'N/A';
                iouCell.style.color = '#9ca3af';
            }
            row.appendChild(iouCell);

            tbody.appendChild(row);
        });
    }

    /**
     * Update mean IoU display
     */
    updateMeanIoU(meanIoU) {
        const display = document.getElementById('meanIoU');
        if (!display) return;

        if (meanIoU !== null && meanIoU !== undefined) {
            display.textContent = `${(meanIoU * 100).toFixed(2)}%`;
            display.style.color = meanIoU > 0.5 ? '#10b981' : '#f87171';
        } else {
            display.textContent = 'N/A';
            display.style.color = '#9ca3af';
        }
    }

    /**
     * Display live coverage stats for webcam
     */
    displayLiveCoverage(coverage) {
        const chartContainer = document.getElementById('liveCoverageChart');
        if (!chartContainer) return;

        chartContainer.innerHTML = '';

        // Sort coverage by value and get top 5
        const sorted = Object.entries(coverage)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5);

        sorted.forEach(([classIdx, percent]) => {
            const idx = parseInt(classIdx);
            const className = CLASS_NAMES[idx];
            const color = CLASS_COLORS_RGB[idx];

            const item = document.createElement('div');
            item.className = 'coverage-item';
            item.style.borderLeftColor = color;

            const name = document.createElement('span');
            name.className = 'coverage-item-name';
            name.textContent = className;

            const barContainer = document.createElement('div');
            barContainer.style.display = 'flex';
            barContainer.style.alignItems = 'center';
            barContainer.style.gap = '8px';
            barContainer.style.marginTop = '5px';

            const bar = document.createElement('div');
            bar.style.flex = '1';
            bar.style.height = '6px';
            bar.style.background = 'rgba(0, 0, 0, 0.3)';
            bar.style.borderRadius = '3px';
            bar.style.overflow = 'hidden';

            const barFill = document.createElement('div');
            barFill.style.height = '100%';
            barFill.style.width = `${Math.min(percent, 100)}%`;
            barFill.style.background = color;
            barFill.style.transition = 'width 0.2s ease';

            bar.appendChild(barFill);

            const value = document.createElement('span');
            value.className = 'coverage-item-value';
            value.textContent = `${percent.toFixed(1)}%`;

            barContainer.appendChild(bar);
            barContainer.appendChild(value);

            item.appendChild(name);
            item.appendChild(barContainer);
            chartContainer.appendChild(item);
        });
    }

    /**
     * Update webcam stats display
     */
    updateWebcamStats(inferenceTimeMs, fps, dominantClass) {
        const fpsDisplay = document.getElementById('fpsDisplay');
        const timeDisplay = document.getElementById('webcamInferenceTime');
        const classDisplay = document.getElementById('dominantClass');

        if (fpsDisplay) fpsDisplay.textContent = fps.toFixed(1);
        if (timeDisplay) timeDisplay.textContent = inferenceTimeMs.toFixed(2);
        if (classDisplay && dominantClass) {
            classDisplay.textContent = CLASS_NAMES[dominantClass];
            classDisplay.style.color = CLASS_COLORS_RGB[dominantClass];
        }
    }

    /**
     * Create a formatted metrics summary
     */
    createSummary(metrics) {
        const summary = {
            totalClasses: 10,
            meanIoU: metrics.mean_iou,
            coverage: metrics.per_class_coverage,
            classIoU: metrics.per_class_iou
        };

        return summary;
    }

    /**
     * Export metrics as JSON
     */
    exportMetricsJSON(metrics) {
        const json = JSON.stringify(metrics, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `metrics_${new Date().toISOString()}.json`;
        link.click();
        URL.revokeObjectURL(url);
    }

    /**
     * Export metrics as CSV
     */
    exportMetricsCSV(metrics) {
        const coverage = metrics.per_class_coverage || {};
        const iouValues = metrics.per_class_iou || {};

        let csv = 'Class,Coverage (%),IoU\n';
        CLASS_NAMES.forEach((name, idx) => {
            const cov = coverage[name] || 0;
            const iou = iouValues[name] || 'N/A';
            csv += `"${name}",${cov.toFixed(2)},${iou}\n`;
        });

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `metrics_${new Date().toISOString()}.csv`;
        link.click();
        URL.revokeObjectURL(url);
    }
}

// Create instance
const statsPanel = new StatsPanel();

/**
 * Segmentation Viewer Component
 * Renders segmentation predictions as colored overlays
 */

// Class colors mapping (RGB)
const CLASS_COLORS = {
    0: [0, 255, 0],       // Trees - Green
    1: [34, 139, 34],     // Lush Bushes - Dark Green
    2: [255, 255, 0],     // Dry Grass - Yellow
    3: [210, 180, 140],   // Dry Bushes - Tan
    4: [165, 42, 42],     // Ground Clutter - Brown
    5: [255, 20, 147],    // Flowers - Deep Pink
    6: [160, 82, 45],     // Logs - Sienna
    7: [128, 128, 128],   // Rocks - Gray
    8: [0, 128, 255],     // Landscape - Orange
    9: [0, 0, 255],       // Sky - Blue
};

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

class SegmentationViewer {
    constructor() {
        this.predictions = null;
        this.originalImage = null;
        this.predictionImage = null;
    }

    /**
     * Display image on canvas
     */
    displayImage(canvas, imageSrc) {
        const ctx = canvas.getContext('2d');
        
        if (typeof imageSrc === 'string') {
            // Base64 image
            const img = new Image();
            img.onload = () => {
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
            };
            img.src = `data:image/jpeg;base64,${imageSrc}`;
        } else {
            // Canvas or image element
            canvas.width = imageSrc.width;
            canvas.height = imageSrc.height;
            ctx.drawImage(imageSrc, 0, 0);
        }
    }

    /**
     * Decode predictions from base64 image
     * Assumes each pixel's red channel contains class index
     */
    decodePredictionsFromImage(imageBase64) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0);

                const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                const data = imgData.data;
                const predictions = new Uint8Array(canvas.width * canvas.height);

                for (let i = 0; i < data.length; i += 4) {
                    predictions[i / 4] = data[i]; // Use red channel as class index
                }

                resolve({
                    data: predictions,
                    width: canvas.width,
                    height: canvas.height
                });
            };
            img.src = `data:image/jpeg;base64,${imageBase64}`;
        });
    }

    /**
     * Create color-coded segmentation from predictions array
     */
    createColoredSegmentation(predictions, width, height) {
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');

        const imageData = ctx.createImageData(width, height);
        const data = imageData.data;

        for (let i = 0; i < predictions.length; i++) {
            const classIdx = Math.min(predictions[i], 9);
            const color = CLASS_COLORS[classIdx];

            const pixelIdx = i * 4;
            data[pixelIdx] = color[0];     // R
            data[pixelIdx + 1] = color[1]; // G
            data[pixelIdx + 2] = color[2]; // B
            data[pixelIdx + 3] = 255;      // A
        }

        ctx.putImageData(imageData, 0, 0);
        return canvas;
    }

    /**
     * Create blended overlay of original and segmentation
     */
    createBlendedOverlay(originalBase64, segmentationBase64, alpha = 0.6) {
        return new Promise((resolve) => {
            const originalImg = new Image();
            const segmentationImg = new Image();

            let loadedCount = 0;

            const onImagesLoaded = () => {
                loadedCount++;
                if (loadedCount === 2) {
                    const canvas = document.createElement('canvas');
                    canvas.width = originalImg.width;
                    canvas.height = originalImg.height;
                    const ctx = canvas.getContext('2d');

                    ctx.drawImage(originalImg, 0, 0);
                    ctx.globalAlpha = alpha;
                    ctx.drawImage(segmentationImg, 0, 0);
                    ctx.globalAlpha = 1.0;

                    resolve(canvas);
                }
            };

            originalImg.onload = onImagesLoaded;
            originalImg.src = `data:image/jpeg;base64,${originalBase64}`;

            segmentationImg.onload = onImagesLoaded;
            segmentationImg.src = `data:image/jpeg;base64,${segmentationBase64}`;
        });
    }

    /**
     * Display inference results on canvases
     */
    displayResults(originalCanvas, predictionCanvas, originalBase64, predictionBase64) {
        this.displayImage(originalCanvas, originalBase64);
        this.displayImage(predictionCanvas, predictionBase64);
    }

    /**
     * Display side-by-side comparison
     */
    displayComparison(comparisonCanvas, comparisonBase64) {
        this.displayImage(comparisonCanvas, comparisonBase64);
    }

    /**
     * Create legend showing all classes and their colors
     */
    createLegend() {
        const legendContainer = document.getElementById('classLegend');
        if (!legendContainer) return;

        legendContainer.innerHTML = '';

        CLASS_NAMES.forEach((name, idx) => {
            const color = CLASS_COLORS[idx];
            const item = document.createElement('div');
            item.className = 'legend-item';

            const colorBox = document.createElement('div');
            colorBox.className = 'legend-color';
            colorBox.style.backgroundColor = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;

            const label = document.createElement('span');
            label.className = 'legend-label';
            label.textContent = name;

            item.appendChild(colorBox);
            item.appendChild(label);
            legendContainer.appendChild(item);
        });
    }

    /**
     * Draw segmentation on canvas with streaming updates
     */
    drawSegmentationFrame(canvas, segmentationBase64, originalBase64 = null, showOverlay = true) {
        return new Promise((resolve) => {
            const segImg = new Image();
            segImg.onload = () => {
                canvas.width = segImg.width;
                canvas.height = segImg.height;
                const ctx = canvas.getContext('2d');

                if (showOverlay && originalBase64) {
                    const origImg = new Image();
                    origImg.onload = () => {
                        ctx.drawImage(origImg, 0, 0);
                        ctx.globalAlpha = 0.7;
                        ctx.drawImage(segImg, 0, 0);
                        ctx.globalAlpha = 1.0;
                        resolve();
                    };
                    origImg.src = `data:image/jpeg;base64,${originalBase64}`;
                } else {
                    ctx.drawImage(segImg, 0, 0);
                    resolve();
                }
            };
            segImg.src = `data:image/jpeg;base64,${segmentationBase64}`;
        });
    }

    /**
     * Create mini visualization for stats panel
     */
    createMiniVisualization(base64Data, maxWidth = 200) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                const scale = maxWidth / img.width;
                canvas.width = maxWidth;
                canvas.height = img.height * scale;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                resolve(canvas.toDataURL());
            };
            img.src = `data:image/jpeg;base64,${base64Data}`;
        });
    }
}

// Create instance
const segmentationViewer = new SegmentationViewer();

// Initialize legend on page load
document.addEventListener('DOMContentLoaded', () => {
    segmentationViewer.createLegend();
});

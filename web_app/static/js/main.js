// Global variables
let selectedFile = null;
let previewImage = null;
let imageCanvas = null;
let canvasContext = null;
let imageElement = null;
let boxes = []; // Store exemplar boxes [{x1, y1, x2, y2}, ...]
let isDrawing = false;
let startX = 0;
let startY = 0;
let currentBox = null;

// DOM elements
const uploadBox = document.getElementById('uploadBox');
const imageInput = document.getElementById('imageInput');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const imageCanvasContainer = document.getElementById('imageCanvasContainer');
const imageCanvasEl = document.getElementById('imageCanvas');
const boxCountEl = document.getElementById('boxCount');
const clearBoxesBtn = document.getElementById('clearBoxesBtn');
const previewBtn = document.getElementById('previewBtn');
const confirmBtn = document.getElementById('confirmBtn');
const cancelPreviewBtn = document.getElementById('cancelPreviewBtn');
const clearBtn = document.getElementById('clearBtn');
const loadingIndicator = document.getElementById('loadingIndicator');
const previewSection = document.getElementById('previewSection');
const previewGrid = document.getElementById('previewGrid');
const resultsSection = document.getElementById('resultsSection');
const errorMessage = document.getElementById('errorMessage');

// Store preview data
let previewData = null;

// Event listeners
uploadBox.addEventListener('click', (e) => {
    if (e.target === uploadBox || e.target === uploadPlaceholder || uploadPlaceholder.contains(e.target)) {
        imageInput.click();
    }
});
uploadBox.addEventListener('dragover', handleDragOver);
uploadBox.addEventListener('dragleave', handleDragLeave);
uploadBox.addEventListener('drop', handleDrop);
imageInput.addEventListener('change', handleFileSelect);
previewBtn.addEventListener('click', previewBoxes);
confirmBtn.addEventListener('click', confirmAndProcess);
cancelPreviewBtn.addEventListener('click', cancelPreview);
clearBtn.addEventListener('click', clearAll);
clearBoxesBtn.addEventListener('click', clearBoxes);

// Canvas drawing related events
if (imageCanvasEl) {
    imageCanvasEl.addEventListener('mousedown', startDrawing);
    imageCanvasEl.addEventListener('mousemove', draw);
    imageCanvasEl.addEventListener('mouseup', stopDrawing);
    imageCanvasEl.addEventListener('mouseleave', stopDrawing);
}

// Drag and drop handling
function handleDragOver(e) {
    e.preventDefault();
    uploadBox.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// File selection handling
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select an image file');
        return;
    }
    
    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showError('File size cannot exceed 16MB');
        return;
    }
    
    selectedFile = file;
    
    // Display preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage = e.target.result;
        
        // Create image element for canvas
        imageElement = new Image();
        imageElement.onload = () => {
            setupCanvas();
            hideError();
        };
        imageElement.src = previewImage;
    };
    reader.readAsDataURL(file);
}

// Setup Canvas
function setupCanvas() {
    if (!imageElement) return;
    
    // Calculate canvas size (maintain aspect ratio, max width 800px)
    const maxWidth = 800;
    const maxHeight = 600;
    let canvasWidth = imageElement.width;
    let canvasHeight = imageElement.height;
    
    if (canvasWidth > maxWidth) {
        canvasHeight = (canvasHeight * maxWidth) / canvasWidth;
        canvasWidth = maxWidth;
    }
    if (canvasHeight > maxHeight) {
        canvasWidth = (canvasWidth * maxHeight) / canvasHeight;
        canvasHeight = maxHeight;
    }
    
    imageCanvasEl.width = canvasWidth;
    imageCanvasEl.height = canvasHeight;
    canvasContext = imageCanvasEl.getContext('2d');
    
    // Draw image
    canvasContext.drawImage(imageElement, 0, 0, canvasWidth, canvasHeight);
    
    // Show canvas container, hide upload placeholder
    uploadPlaceholder.style.display = 'none';
    imageCanvasContainer.style.display = 'block';
    
    // Reset boxes
    boxes = [];
    updateBoxCount();
    previewData = null;
    
    // Update button state
    previewBtn.disabled = boxes.length === 0;
    previewSection.style.display = 'none';
    resultsSection.style.display = 'none';
}

// Start drawing
function startDrawing(e) {
    if (boxes.length >= 3) {
        showError('Maximum 3 exemplar boxes allowed');
        return;
    }
    
    const rect = imageCanvasEl.getBoundingClientRect();
    const scaleX = imageCanvasEl.width / rect.width;
    const scaleY = imageCanvasEl.height / rect.height;
    
    startX = (e.clientX - rect.left) * scaleX;
    startY = (e.clientY - rect.top) * scaleY;
    isDrawing = true;
}

// Drawing in progress
function draw(e) {
    if (!isDrawing) return;
    
    const rect = imageCanvasEl.getBoundingClientRect();
    const scaleX = imageCanvasEl.width / rect.width;
    const scaleY = imageCanvasEl.height / rect.height;
    
    const currentX = (e.clientX - rect.left) * scaleX;
    const currentY = (e.clientY - rect.top) * scaleY;
    
    // Redraw canvas
    canvasContext.clearRect(0, 0, imageCanvasEl.width, imageCanvasEl.height);
    canvasContext.drawImage(imageElement, 0, 0, imageCanvasEl.width, imageCanvasEl.height);
    
    // Draw existing boxes
    drawAllBoxes();
    
    // Draw currently drawing box
    const x1 = Math.min(startX, currentX);
    const y1 = Math.min(startY, currentY);
    const x2 = Math.max(startX, currentX);
    const y2 = Math.max(startY, currentY);
    
    drawBox(x1, y1, x2, y2, '#ff0000', true);
    currentBox = {x1, y1, x2, y2};
}

// Stop drawing
function stopDrawing(e) {
    if (!isDrawing) return;
    isDrawing = false;
    
    if (currentBox) {
        // Ensure box has minimum size
        const minSize = 10;
        if (Math.abs(currentBox.x2 - currentBox.x1) < minSize || 
            Math.abs(currentBox.y2 - currentBox.y1) < minSize) {
            // Box too small, don't add
            currentBox = null;
            redrawCanvas();
            return;
        }
        
        // Convert to original image coordinates
        const originalBox = convertToOriginalCoordinates(currentBox);
        boxes.push(originalBox);
        updateBoxCount();
        previewBtn.disabled = boxes.length === 0;
        // Clear previous preview
        previewData = null;
        previewSection.style.display = 'none';
        currentBox = null;
    }
    
    redrawCanvas();
}

// Convert canvas coordinates to original image coordinates
function convertToOriginalCoordinates(canvasBox) {
    const canvasWidth = imageCanvasEl.width;
    const canvasHeight = imageCanvasEl.height;
    const originalWidth = imageElement.width;
    const originalHeight = imageElement.height;
    
    return {
        x1: Math.round((canvasBox.x1 / canvasWidth) * originalWidth),
        y1: Math.round((canvasBox.y1 / canvasHeight) * originalHeight),
        x2: Math.round((canvasBox.x2 / canvasWidth) * originalWidth),
        y2: Math.round((canvasBox.y2 / canvasHeight) * originalHeight)
    };
}

// Draw single box
function drawBox(x1, y1, x2, y2, color = '#00ff00', isTemporary = false) {
    canvasContext.strokeStyle = color;
    canvasContext.lineWidth = isTemporary ? 2 : 3;
    canvasContext.setLineDash(isTemporary ? [5, 5] : []);
    canvasContext.strokeRect(x1, y1, x2 - x1, y2 - y1);
    
    if (!isTemporary) {
        // Draw box label
        const boxIndex = boxes.length;
        canvasContext.fillStyle = color;
        canvasContext.font = 'bold 14px Arial';
        canvasContext.fillText(`Exemplar ${boxIndex + 1}`, x1 + 5, y1 + 18);
    }
}

// Draw all boxes
function drawAllBoxes() {
    const canvasWidth = imageCanvasEl.width;
    const canvasHeight = imageCanvasEl.height;
    const originalWidth = imageElement.width;
    const originalHeight = imageElement.height;
    
    boxes.forEach((box, index) => {
        // Convert original coordinates to canvas coordinates
        const x1 = (box.x1 / originalWidth) * canvasWidth;
        const y1 = (box.y1 / originalHeight) * canvasHeight;
        const x2 = (box.x2 / originalWidth) * canvasWidth;
        const y2 = (box.y2 / originalHeight) * canvasHeight;
        
        drawBox(x1, y1, x2, y2, '#00ff00', false);
    });
}

// Redraw canvas
function redrawCanvas() {
    canvasContext.clearRect(0, 0, imageCanvasEl.width, imageCanvasEl.height);
    canvasContext.drawImage(imageElement, 0, 0, imageCanvasEl.width, imageCanvasEl.height);
    drawAllBoxes();
}

// Clear all boxes
function clearBoxes() {
    boxes = [];
    updateBoxCount();
    redrawCanvas();
    previewBtn.disabled = true;
    previewData = null;
    previewSection.style.display = 'none';
}

// Update box count
function updateBoxCount() {
    boxCountEl.textContent = boxes.length;
}

// Preview exemplar boxes
async function previewBoxes() {
    if (!selectedFile) {
        showError('Please select an image file first');
        return;
    }
    
    if (boxes.length === 0) {
        showError('Please draw at least 1 exemplar box! Exemplar boxes tell the model what objects to count.');
        return;
    }
    
    // Show loading indicator
    loadingIndicator.style.display = 'block';
    previewSection.style.display = 'none';
    previewBtn.disabled = true;
    hideError();
    
    try {
        // Create FormData
        const formData = new FormData();
        formData.append('image', selectedFile);
        
        // Add exemplar boxes (format: [[x1, y1, x2, y2], ...])
        const exemplarBoxes = boxes.map(box => [box.x1, box.y1, box.x2, box.y2]);
        formData.append('exemplar_boxes', JSON.stringify(exemplarBoxes));
        
        // Send preview request
        const response = await fetch('/api/preview', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Error during preview');
        }
        
        // Save preview data
        previewData = data;
        
        // Display preview
        displayPreview(data);
        
    } catch (error) {
        showError(error.message || 'Error during preview, please try again');
        console.error('Error:', error);
    } finally {
        loadingIndicator.style.display = 'none';
        previewBtn.disabled = boxes.length > 0;
    }
}

// Display preview
function displayPreview(data) {
    if (!data.success || !data.previews || data.previews.length === 0) {
        showError('Invalid preview data');
        return;
    }
    
    // Clear preview grid
    previewGrid.innerHTML = '';
    
    // Add each preview item
    data.previews.forEach(preview => {
        const previewItem = document.createElement('div');
        previewItem.className = 'preview-item';
        
        const label = document.createElement('div');
        label.className = 'preview-item-label';
        label.textContent = `Exemplar Box ${preview.index}`;
        
        const img = document.createElement('img');
        img.src = preview.image;
        img.alt = `Exemplar Box ${preview.index}`;
        img.className = 'zoomable-image';
        img.setAttribute('data-title', `Exemplar Box ${preview.index}`);
        
        previewItem.appendChild(label);
        previewItem.appendChild(img);
        previewGrid.appendChild(previewItem);
    });
    
    // Show preview area
    previewSection.style.display = 'block';
    previewSection.scrollIntoView({ behavior: 'smooth' });
}

// Cancel preview, return to edit
function cancelPreview() {
    previewSection.style.display = 'none';
    previewData = null;
    // Scroll to canvas area
    imageCanvasContainer.scrollIntoView({ behavior: 'smooth' });
}

// Confirm and process image
async function confirmAndProcess() {
    if (!selectedFile) {
        showError('Please select an image file first');
        return;
    }
    
    if (boxes.length === 0) {
        showError('Please draw at least 1 exemplar box!');
        return;
    }
    
    // Hide preview area, show loading indicator
    previewSection.style.display = 'none';
    loadingIndicator.style.display = 'block';
    resultsSection.style.display = 'none';
    confirmBtn.disabled = true;
    hideError();
    
    try {
        // Create FormData
        const formData = new FormData();
        formData.append('image', selectedFile);
        
        // Add exemplar boxes (format: [[x1, y1, x2, y2], ...])
        const exemplarBoxes = boxes.map(box => [box.x1, box.y1, box.x2, box.y2]);
        formData.append('exemplar_boxes', JSON.stringify(exemplarBoxes));
        
        // Send processing request
        const response = await fetch('/api/process', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Error processing image');
        }
        
        // Display results
        displayResults(data);
        
    } catch (error) {
        showError(error.message || 'Error processing image, please try again');
        console.error('Error:', error);
    } finally {
        loadingIndicator.style.display = 'none';
        confirmBtn.disabled = false;
    }
}

// Display results
function displayResults(data) {
    if (!data.success) {
        showError('Processing failed');
        return;
    }
    
    // Update count
    document.getElementById('predCount').textContent = data.pred_count;
    
    // Update images and ensure zoom functionality is added
    const correlationMapImg = document.getElementById('correlationMapImg');
    const dynamicWeightsImg = document.getElementById('dynamicWeightsImg');
    const densityMapImg = document.getElementById('densityMapImg');
    
    // Set image sources and ensure class names and attributes are correct
    correlationMapImg.src = data.images.correlation_map;
    correlationMapImg.className = 'zoomable-image';
    correlationMapImg.setAttribute('data-title', 'Correlation Map');
    
    dynamicWeightsImg.src = data.images.dynamic_weights;
    dynamicWeightsImg.className = 'zoomable-image';
    dynamicWeightsImg.setAttribute('data-title', 'Dynamic Weights');
    
    densityMapImg.src = data.images.density_map;
    densityMapImg.className = 'zoomable-image';
    densityMapImg.setAttribute('data-title', 'Density Map');
    
    // Show results area
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Clear all
function clearAll() {
    selectedFile = null;
    previewImage = null;
    imageElement = null;
    boxes = [];
    previewData = null;
    imageInput.value = '';
    uploadPlaceholder.style.display = 'block';
    imageCanvasContainer.style.display = 'none';
    previewBtn.disabled = true;
    previewSection.style.display = 'none';
    resultsSection.style.display = 'none';
    loadingIndicator.style.display = 'none';
    hideError();
    updateBoxCount();
}

// Error handling
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    errorMessage.scrollIntoView({ behavior: 'smooth' });
}

function hideError() {
    errorMessage.style.display = 'none';
}

// Image zoom modal functionality
const imageModal = document.getElementById('imageModal');
const modalImage = document.getElementById('modalImage');
const modalCaption = document.getElementById('modalCaption');
const modalClose = document.querySelector('.modal-close');

// Add click events for all zoomable images (using event delegation, supports dynamically added images)
document.addEventListener('click', function(e) {
    // Check if clicked element is zoomable-image class or its child element
    let target = e.target;
    
    // If clicked element is img tag and contains zoomable-image class
    if (target.tagName === 'IMG' && target.classList.contains('zoomable-image') && target.src) {
        openModal(target);
    }
    // If clicked element is inside zoomable-image, search up for img tag
    else {
        let imgElement = target.closest('.zoomable-image');
        if (imgElement && imgElement.tagName === 'IMG' && imgElement.src) {
            openModal(imgElement);
        }
    }
});

// Open modal
function openModal(img) {
    imageModal.style.display = 'block';
    modalImage.src = img.src;
    modalCaption.textContent = img.getAttribute('data-title') || img.alt;
    // Prevent background scrolling
    document.body.style.overflow = 'hidden';
}

// Close modal
function closeModal() {
    imageModal.style.display = 'none';
    document.body.style.overflow = 'auto';
}

// Click close button to close modal
if (modalClose) {
    modalClose.addEventListener('click', closeModal);
}

// Click modal background to close
imageModal.addEventListener('click', function(e) {
    if (e.target === imageModal) {
        closeModal();
    }
});

// Press ESC key to close modal
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && imageModal.style.display === 'block') {
        closeModal();
    }
});

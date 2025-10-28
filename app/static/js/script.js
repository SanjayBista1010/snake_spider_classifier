// In the checkModelStatus function, update the success message:
function checkModelStatus() {
    fetch('/health')
        .then(response => response.json())
        .then(data => {
            const statusBadge = document.getElementById('statusBadge');
            const statusDetails = document.getElementById('statusDetails');
            const modelStatusCard = document.getElementById('modelStatusCard');
            const predictBtn = document.getElementById('predictBtn');
            const batchPredictBtn = document.getElementById('batchPredictBtn');
            
            if (data.model_loaded) {
                statusBadge.className = 'badge bg-success';
                statusBadge.textContent = '✅ Loaded (98.68% Accuracy)';
                statusDetails.textContent = `Device: ${data.device} | Classes: ${data.class_names.join(', ')}`;
                modelStatusCard.className = 'card mb-4 border-success';
                
                // Enable buttons
                predictBtn.disabled = false;
                batchPredictBtn.disabled = false;
            } else {
                statusBadge.className = 'badge bg-danger';
                statusBadge.textContent = '❌ Not Loaded';
                statusDetails.textContent = 'Model failed to load. Check console for errors.';
                modelStatusCard.className = 'card mb-4 border-danger';
                
                // Keep buttons disabled
                predictBtn.disabled = true;
                batchPredictBtn.disabled = true;
            }
        })
        .catch(error => {
            const statusBadge = document.getElementById('statusBadge');
            const statusDetails = document.getElementById('statusDetails');
            
            statusBadge.className = 'badge bg-danger';
            statusBadge.textContent = '❌ Error';
            statusDetails.textContent = 'Failed to check model status';
        });
}
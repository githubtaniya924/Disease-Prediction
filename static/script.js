// Wait for the DOM content to be fully loaded
document.addEventListener('DOMContentLoaded', () => {
    const symptomSearchInput = document.getElementById('symptom-search');
    const symptomResultsContainer = document.getElementById('symptom-results');
    const selectedSymptomsContainer = document.getElementById('selected-symptoms-container');
    const modelButtons = document.querySelectorAll('.model-button');
    const symptomCountMessage = document.getElementById('symptom-count-message');
    
    // Elements to hold prediction results
    const dtDisease = document.getElementById('dt-disease');
    const dtConfidence = document.getElementById('dt-confidence');
    const knnDisease = document.getElementById('knn-disease');
    const knnConfidence = document.getElementById('knn-confidence');
    const associationSymptoms = document.getElementById('association-symptoms');

    let allSymptoms = [];
    let selectedSymptoms = [];

    // Fetch the full list of symptoms from the backend
    fetch('/symptoms')
        .then(response => response.json())
        .then(data => {
            allSymptoms = data.all_symptoms;
        });

    // Event listener for the search bar input
    symptomSearchInput.addEventListener('input', () => {
        const searchTerm = symptomSearchInput.value.toLowerCase();
        symptomResultsContainer.innerHTML = '';
        if (searchTerm.length > 1) {
            const filteredSymptoms = allSymptoms.filter(symptom =>
                symptom.toLowerCase().includes(searchTerm)
            );
            renderSymptoms(filteredSymptoms);
        }
    });

    function renderSymptoms(symptoms) {
        symptoms.forEach(symptom => {
            if (!selectedSymptoms.includes(symptom)) {
                const symptomDiv = document.createElement('div');
                symptomDiv.className = 'symptom-item';
                symptomDiv.textContent = symptom;
                symptomDiv.addEventListener('click', () => addSymptom(symptom));
                symptomResultsContainer.appendChild(symptomDiv);
            }
        });
    }

    function addSymptom(symptom) {
        selectedSymptoms.push(symptom);
        symptomSearchInput.value = '';
        symptomResultsContainer.innerHTML = '';
        renderSelectedSymptoms();
        updateButtonState();
    }

    function removeSymptom(symptomToRemove) {
        selectedSymptoms = selectedSymptoms.filter(symptom => symptom !== symptomToRemove);
        renderSelectedSymptoms();
        updateButtonState();
    }

    function renderSelectedSymptoms() {
        selectedSymptomsContainer.innerHTML = '';
        selectedSymptoms.forEach(symptom => {
            const symptomChip = document.createElement('span');
            symptomChip.className = 'symptom-chip';
            symptomChip.innerHTML = `${symptom} &times;`;
            symptomChip.addEventListener('click', () => removeSymptom(symptom));
            selectedSymptomsContainer.appendChild(symptomChip);
        });
    }

    function updateButtonState() {
        if (selectedSymptoms.length >= 2) {
            modelButtons.forEach(button => button.disabled = false);
            symptomCountMessage.textContent = '';
        } else {
            modelButtons.forEach(button => button.disabled = true);
            symptomCountMessage.textContent = 'Please select a minimum of 2 symptoms to enable prediction.';
        }
    }

    window.predict = function(model) {
        // Clear previous results from other models
        clearResults();

        fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                'symptoms': selectedSymptoms,
                'model': model
            })
        })
        .then(response => response.json())
        .then(data => {
            if (model === 'decision_tree') {
                dtDisease.textContent = data.disease;
                dtConfidence.textContent = data.confidence;
            } else if (model === 'knn') {
                knnDisease.textContent = data.disease;
                knnConfidence.textContent = data.confidence;
            } else if (model === 'association') {
                associationSymptoms.textContent = data.rules;
            }
        })
        .catch(error => console.error('Error:', error));
    };
    
    // Function to clear all prediction results
    function clearResults() {
        dtDisease.textContent = '';
        dtConfidence.textContent = '';
        knnDisease.textContent = '';
        knnConfidence.textContent = '';
        associationSymptoms.textContent = '';
    }
});

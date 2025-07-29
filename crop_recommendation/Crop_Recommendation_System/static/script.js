document.getElementById("predictionForm").addEventListener("submit", function(event) {
            event.preventDefault();
            
            // Show loading indicator
            const button = this.querySelector('button');
            const buttonText = button.querySelector('.button-text');
            const loader = button.querySelector('.loader');
            const resultBox = document.getElementById("resultBox");
            const resultElement = document.getElementById("result");
            const resultIcon = document.getElementById("resultIcon");
            const errorMessage = document.getElementById("errorMessage");
            
            buttonText.style.display = 'none';
            loader.style.display = 'block';
            errorMessage.textContent = '';
            
            // Get form data
            const formData = new FormData(this);
            
            // Make the actual API request to Flask backend
            fetch("/predict", {
                method: "POST",
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                // Hide loader, show button text
                buttonText.style.display = 'block';
                loader.style.display = 'none';
                
                if (data.crop) {
                    // Set appropriate icon based on prediction
                    if (data.crop === "Rice") {
                        resultIcon.innerHTML = '<i class="fas fa-seedling"></i>';
                    } else if (data.crop === "Maize") {
                        resultIcon.innerHTML = '<i class="fas fa-wheat-awn"></i>';
                    }
                    
                    resultElement.textContent = data.crop;
                    resultBox.classList.add("active");
                } else if (data.error) {
                    resultElement.textContent = "Prediction Error";
                    errorMessage.textContent = data.error;
                    resultIcon.innerHTML = '<i class="fas fa-exclamation-triangle" style="color: var(--error-color);"></i>';
                    resultBox.classList.add("active");
                }
            })
            .catch(error => {
                console.error("Error:", error);
                
                // Show error message
                buttonText.style.display = 'block';
                loader.style.display = 'none';
                resultElement.textContent = "Connection Error";
                errorMessage.textContent = "Could not connect to the prediction service. Please try again later.";
                resultIcon.innerHTML = '<i class="fas fa-exclamation-triangle" style="color: var(--error-color);"></i>';
                resultBox.classList.add("active");
            });
        });
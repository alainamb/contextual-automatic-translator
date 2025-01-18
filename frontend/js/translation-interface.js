translation-interface.js

// State management for translation direction
let isEnglishToSpanish = true;

// DOM Elements
const reverseBtn = document.getElementById('reverseBtn');
const translateBtn = document.getElementById('translateBtn');
const sourceDisplay = document.getElementById('sourceLanguageDisplay');
const targetDisplay = document.getElementById('targetLanguageDisplay');
const translationOutput = document.getElementById('translationOutput');

// Event Handlers
function handleDirectionChange() {
    isEnglishToSpanish = !isEnglishToSpanish;
    
    if (isEnglishToSpanish) {
        sourceDisplay.textContent = 'English (US)';
        targetDisplay.textContent = 'Spanish (LATAM)';
    } else {
        sourceDisplay.textContent = 'Spanish (LATAM)';
        targetDisplay.textContent = 'English (US)';
    }
}

async function handleTranslate() {
    const sourceLanguage = isEnglishToSpanish ? 'en-US' : 'es-419';
    const targetLanguage = isEnglishToSpanish ? 'es-419' : 'en-US';
    
    try {
        // Show loading state
        translateBtn.disabled = true;
        translateBtn.textContent = 'Translating...';
        
        // Your translation API call will go here
        // const response = await translateText(sourceLanguage, targetLanguage, text);
        // translationOutput.textContent = response.translatedText;
        
    } catch (error) {
        console.error('Translation error:', error);
        translationOutput.textContent = 'Error during translation. Please try again.';
    } finally {
        // Reset button state
        translateBtn.disabled = false;
        translateBtn.textContent = 'Translate';
    }
}

// Event Listeners
reverseBtn.addEventListener('click', handleDirectionChange);
translateBtn.addEventListener('click', handleTranslate);

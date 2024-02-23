function toggleTheme() {
    const body = document.body;
    body.classList.toggle('dark-mode');

    // Check if dark mode is enabled and save the preference to localStorage
    const isDarkModeEnabled = body.classList.contains('dark-mode');
    localStorage.setItem('darkMode', isDarkModeEnabled);
}

// Check the saved theme preference on page load
document.addEventListener('DOMContentLoaded', () => {
    const isDarkModeEnabled = localStorage.getItem('darkMode') === 'true';

    // Apply the saved theme preference
    const body = document.body;
    if (isDarkModeEnabled) {
        body.classList.add('dark-mode');
    } else {
        body.classList.remove('dark-mode');
    }
});

// Function to toggle content visibility
function toggleContent(element) {
    var score = element.nextElementSibling;
    score.classList.toggle('show');
}
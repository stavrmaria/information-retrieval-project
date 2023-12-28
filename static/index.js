/*=================== Autocomplete ===================*/
let availableKeywords = [
    'Information Retrieval',
    'Boolean model',
    'Greek Parliament',
];

const resultsBox = document.querySelector(".results__box");
const inputBox = document.querySelector("#input-box");

inputBox.onkeyup = function() {
    let result = [];
    let inputValue = inputBox.value;

    if (inputValue.length) {
        result = availableKeywords.filter((keyword) => {
            return keyword.toLowerCase().includes(inputValue.toLowerCase());
        });

        console.log(result);
    }

    displayResult(result);

    if (!result.length) {
        resultsBox.innerHTML = '';
    }
}

function displayResult(result) {
    const content = result.map((list) => {
        return "<li onclick=selectInput(this)>" + list + "</li>";
    });

    resultsBox.innerHTML = "<ul>" + content.join('') + "</ul>";
}

function selectInput(list) {
    inputBox.value = list.innerHTML;
    resultsBox.innerHTML = '';
}

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

function redirectToTopKeywordsMemberPlot() {
    // Redirect to the /top_keywords_member_plot endpoint
    window.location.href = '/top_keywords_member_plot';
}

function redirectToTopKeywordsPartyPlot() {
    // Redirect to the /top_keywords_party_plot endpoint
    window.location.href = '/top_keywords_party_plot';
}
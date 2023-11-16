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
}
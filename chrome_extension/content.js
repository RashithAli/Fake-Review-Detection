let selectedText = "";

// Create floating Analyze button
function createAnalyzeButton(x, y) {
    removeAnalyzeButton();

    const button = document.createElement("button");
    button.id = "analyze-review-btn";
    button.innerText = "Analyze Review";

    button.style.position = "absolute";
    button.style.left = x + "px";
    button.style.top = y + "px";
    button.style.padding = "6px 10px";
    button.style.background = "#111";
    button.style.color = "#fff";
    button.style.border = "none";
    button.style.borderRadius = "6px";
    button.style.cursor = "pointer";
    button.style.zIndex = "999999";

    button.onclick = () => {
        analyzeReview(selectedText);
        removeAnalyzeButton();
    };

    document.body.appendChild(button);
}

function removeAnalyzeButton() {
    const oldBtn = document.getElementById("analyze-review-btn");
    if (oldBtn) oldBtn.remove();
}

// Create popup box
function createResultBox() {
    const existingBox = document.getElementById("review-ai-box");
    if (existingBox) existingBox.remove();

    const box = document.createElement("div");
    box.id = "review-ai-box";

    box.style.position = "fixed";
    box.style.bottom = "20px";
    box.style.right = "20px";
    box.style.width = "350px";
    box.style.background = "#fff";
    box.style.borderRadius = "10px";
    box.style.boxShadow = "0 5px 20px rgba(0,0,0,0.3)";
    box.style.padding = "15px";
    box.style.zIndex = "999999";
    box.style.fontFamily = "Arial";

    box.innerHTML = `
        <div id="ai-content">
            <h3>Analyzing...</h3>
        </div>
        <button id="ai-close" style="
            position:absolute;
            top:8px;
            right:10px;
            background:red;
            color:white;
            border:none;
            border-radius:50%;
            width:25px;
            height:25px;
            cursor:pointer;
        ">X</button>
    `;

    document.body.appendChild(box);

    document.getElementById("ai-close").onclick = () => {
        box.remove();
    };
}

// Call backend
async function analyzeReview(text) {
    createResultBox();

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();

        const contentDiv = document.getElementById("ai-content");

        if (!data || !data.label) {
            contentDiv.innerHTML = "<p style='color:red;'>Invalid response</p>";
            return;
        }

        contentDiv.innerHTML = `
            <h2>${data.label}</h2>
            <p><strong>Confidence:</strong> ${data.confidence}%</p>
            <hr>
            <p>${data.explanation || "No explanation available."}</p>
        `;

        if (data.label.includes("Fake")) {
            contentDiv.style.color = "red";
        } else {
            contentDiv.style.color = "green";
        }

    } catch (error) {
        document.getElementById("ai-content").innerHTML =
            "<p style='color:red;'>Server not reachable</p>";
        console.error(error);
    }
}

// Detect proper text selection
document.addEventListener("mouseup", function (e) {
    setTimeout(() => {
        const text = window.getSelection().toString().trim();

        if (text.length > 20) {
            selectedText = text;
            createAnalyzeButton(e.pageX, e.pageY);
        } else {
            removeAnalyzeButton();
        }
    }, 200);
});
async function analyze() {
    const text = document.getElementById("reviewText").value.trim();

    if (!text) {
        alert("Enter a review");
        return;
    }

    const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ review_text: text })
    });

    const data = await response.json();

    document.getElementById("label").innerText =
        data.label === "Fake" ? "❌ Fake Review" : "✅ Genuine Review";

    document.getElementById("confidence").innerText =
        `Fake: ${(data.fake_probability * 100).toFixed(2)}%
Genuine: ${(data.genuine_probability * 100).toFixed(2)}%`;

    document.getElementById("result").classList.remove("hidden");
}
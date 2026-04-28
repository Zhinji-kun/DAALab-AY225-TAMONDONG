let chart;

// Sample HAR-like data
const RAW_DATA = [
    [0.1, 0.2, 0.3],
    [0.5, 0.6, 0.7],
    [0.01, 0.02, 0.03],
    [0.9, 0.8, 0.7]
];

async function predict(features) {
    const res = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features })
    });

    const data = await res.json();
    return data.activity;
}

async function loadData() {
    const tbody = document.getElementById("tableBody");
    tbody.innerHTML = "";

    let counts = {};

    for (let i = 0; i < RAW_DATA.length; i++) {
        const features = RAW_DATA[i];
        const activity = await predict(features);

        counts[activity] = (counts[activity] || 0) + 1;

        const row = `
            <tr>
                <td>${i + 1}</td>
                <td>${features[0]}</td>
                <td>${features[1]}</td>
                <td>${features[2]}</td>
                <td>${activity}</td>
            </tr>
        `;

        tbody.innerHTML += row;
    }

    renderChart(counts);
}

function renderChart(counts) {
    const ctx = document.getElementById("chart");

    if (chart) chart.destroy();

    chart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: Object.keys(counts),
            datasets: [{
                label: "Activity Frequency",
                data: Object.values(counts)
            }]
        }
    });
}
function renderChart(counts) {
    const ctx = document.getElementById("chart");

    if (chart) chart.destroy();

    chart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: Object.keys(counts),
            datasets: [{
                label: "Activity Frequency",
                data: Object.values(counts)
            }]
        }
    });
}

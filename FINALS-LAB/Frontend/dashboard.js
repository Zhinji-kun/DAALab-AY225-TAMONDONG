const API = "http://127.0.0.1:5000";

async function predict() {
  const f1 = parseFloat(document.getElementById("f1").value);
  const f2 = parseFloat(document.getElementById("f2").value);
  const f3 = parseFloat(document.getElementById("f3").value);

  const res = await fetch(`${API}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      features: [f1, f2, f3]
    })
  });

  const data = await res.json();

  document.getElementById("result").innerText =
    data.activity ? "Predicted: " + data.activity : "Error: " + data.error;
}

async function loadSummary() {
  const res = await fetch(`${API}/summary`);
  const data = await res.json();

  document.getElementById("summary").textContent =
    JSON.stringify(data, null, 2);
}                <td>${features[0]}</td>
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

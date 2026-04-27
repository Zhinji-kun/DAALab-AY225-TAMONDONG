const activityEl = document.getElementById("activity");
const xEl = document.getElementById("x");
const yEl = document.getElementById("y");
const zEl = document.getElementById("z");
const startBtn = document.getElementById("startBtn");

function classifyActivity(x, y, z) {
  const magnitude = Math.sqrt(x*x + y*y + z*z);

  if (magnitude < 1.2) return "Idle / Sitting";
  if (magnitude < 2) return "Walking";
  return "Running";
}

startBtn.addEventListener("click", () => {
  if (typeof DeviceMotionEvent !== "undefined" &&
      typeof DeviceMotionEvent.requestPermission === "function") {

    DeviceMotionEvent.requestPermission().then(response => {
      if (response === "granted") {
        startTracking();
      } else {
        alert("Permission denied");
      }
    });

  } else {
    startTracking();
  }
});

function startTracking() {
  window.addEventListener("devicemotion", (event) => {
    const x = event.accelerationIncludingGravity.x || 0;
    const y = event.accelerationIncludingGravity.y || 0;
    const z = event.accelerationIncludingGravity.z || 0;

    xEl.textContent = x.toFixed(2);
    yEl.textContent = y.toFixed(2);
    zEl.textContent = z.toFixed(2);

    const activity = classifyActivity(x, y, z);
    activityEl.textContent = activity;
  });
}
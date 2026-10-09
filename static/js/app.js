// Upload -> poll job status -> redirect to result page.
const drop = document.getElementById("drop");
const fileInput = document.getElementById("file");
const progress = document.getElementById("progress");
const fill = document.getElementById("fill");
const pct = document.getElementById("pct");
const toast = document.getElementById("toast");

function showToast(msg) {
  toast.textContent = msg;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 3500);
}

drop.onclick = () => fileInput.click();

drop.ondragover = (e) => { e.preventDefault(); drop.style.background = "#e0e7ff"; };
drop.ondragleave = () => { drop.style.background = ""; };
drop.ondrop = (e) => {
  e.preventDefault(); drop.style.background = "";
  if (e.dataTransfer.files.length) upload(e.dataTransfer.files[0]);
};
fileInput.onchange = () => { if (fileInput.files.length) upload(fileInput.files[0]); };

async function upload(file) {
  const form = new FormData();
  form.append("video", file);
  drop.classList.add("hidden");
  progress.classList.remove("hidden");

  let res;
  try {
    res = await fetch("/api/upload", { method: "POST", body: form });
  } catch {
    return showToast("Server se connect nahi ho saka.");
  }
  const data = await res.json();
  if (!res.ok) return showToast(data.error || "Upload fail.");

  // Poll progress har 1 second
  const timer = setInterval(async () => {
    const r = await fetch(`/api/job/${data.job_id}`);
    const j = await r.json();
    fill.style.width = (j.progress || 0) + "%";
    pct.textContent = (j.progress || 0) + "%";
    if (j.status === "done") {
      clearInterval(timer);
      window.location = `/result/${data.job_id}`;
    } else if (j.status === "error") {
      clearInterval(timer);
      showToast("Processing fail: " + (j.error || ""));
    }
  }, 1000);
}

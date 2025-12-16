// Netlify will replace this at build time
const API_BASE = "https://resumeragbot.onrender.com";


const qInput = document.getElementById("q");
const askBtn = document.getElementById("askBtn");
const thread = document.getElementById("thread");
const typing = document.getElementById("typing");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");

function setStatus(ok, text) {
  statusDot.classList.toggle("warn", !ok);
  statusText.textContent = text;
}

function addMessage(role, text) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "U" : "R";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const title = document.createElement("div");
  title.className = "title";
  title.textContent = role === "user" ? "You" : "Answer";

  const body = document.createElement("div");
  body.className = "text";
  body.textContent = text;

  bubble.appendChild(title);
  bubble.appendChild(body);

  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  thread.appendChild(wrap);

  const chatBox = document.querySelector(".chat");
  chatBox.scrollTop = chatBox.scrollHeight;
}

async function pingHealth() {
  try {
    const r = await fetch(`${API_BASE}/health`);
    if (!r.ok) throw new Error();
    setStatus(true, "Connected");
  } catch {
    setStatus(false, "Backend not reachable");
  }
}

async function ask() {
  const question = (qInput.value || "").trim();
  if (!question) return;

  addMessage("user", question);
  qInput.value = "";
  askBtn.disabled = true;
  typing.classList.remove("hidden");

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });

    if (!res.ok) throw new Error();
    const data = await res.json();

    addMessage(
      "bot",
      data.answer || "I don’t see that information in the resume."
    );
    setStatus(true, "Connected");
  } catch {
    addMessage(
      "bot",
      "Sorry — the chatbot is temporarily unavailable. Please try again."
    );
    setStatus(false, "Temporary issue");
  } finally {
    typing.classList.add("hidden");
    askBtn.disabled = false;
    qInput.focus();
  }
}

askBtn.addEventListener("click", ask);
qInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") ask();
});

pingHealth();

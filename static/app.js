const API_URL = "http://127.0.0.1:8000/chat"; // ← adapte le port si besoin

const SESSION_ID = (() => {
  let id = localStorage.getItem("vodacom_session_id");
  if (!id) {
    id = "web-" + Math.random().toString(36).slice(2, 10) + "-" + Date.now();
    localStorage.setItem("vodacom_session_id", id);
  }
  return id;
})();

const CUSTOMER_ID = "CUST_001"; // ← change si tu veux tester CUST_002

const messagesEl = document.getElementById("messages");
const form = document.getElementById("chat-form");
const input = document.getElementById("user-input");
const sendBtn = form.querySelector("button[type='submit']");

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

/**
 * Ajoute un message dans le DOM.
 * @param {"user"|"assistant"} role
 * @param {string} text
 * @returns {HTMLElement} la bulle
 */
function addMessage(role, text = "") {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  if (role === "assistant") {
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = "V";
    wrapper.appendChild(avatar);
  }

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  wrapper.appendChild(bubble);
  messagesEl.appendChild(wrapper);
  scrollToBottom();

  return bubble;
}

function setLoading(loading) {
  input.disabled = loading;
  sendBtn.disabled = loading;
  if (!loading) input.focus();
}

// ============================================================
//  Envoi du message
// ============================================================
form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const text = input.value.trim();
  if (!text) return;

  addMessage("user", text);
  input.value = "";
  setLoading(true);

  const bubble = addMessage("assistant", "…");

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: SESSION_ID,
        customer_id: CUSTOMER_ID,
        message: text,
      }),
    });

    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        const errData = await res.json();
        if (errData.detail) detail = errData.detail;
      } catch {}
      throw new Error(detail);
    }

    // Réponse FastAPI : ChatResponse
    // { session_id, intent, scope, response, state, offers }
    const data = await res.json();

    // Remplace le placeholder par la vraie réponse
    bubble.textContent = data.response ?? "(réponse vide)";

    // Logs utiles pour débugger selon le scope
    console.log("Intent :", data.intent);
    console.log("Scope  :", data.scope);
    console.log("State  :", data.state);
    console.log("Offers :", data.offers);

    // Petit rendu enrichi selon le scope (optionnel)
    if (data.scope === "recommendation" && Array.isArray(data.offers) && data.offers.length > 0) {
      const list = document.createElement("div");
      list.style.marginTop = "8px";
      list.style.fontSize = "13px";
      list.style.color = "#555";
      data.offers.forEach((offer) => {
        const line = document.createElement("div");
        line.textContent = `• ${offer.name} — ${offer.price_monthly}$ / mois, ${offer.data_gb} Go`;
        list.appendChild(line);
      });
      bubble.appendChild(list);
    }

    if (data.scope === "tobi" && data.state) {
      // Le state TOBi contient intent, entities, missing_slots, slots_complete, action_result
      if (data.state.missing_slots && data.state.missing_slots.length > 0) {
        const hint = document.createElement("div");
        hint.style.marginTop = "8px";
        hint.style.fontSize = "12px";
        hint.style.color = "#8f8f8f";
        hint.textContent = `Slots manquants : ${data.state.missing_slots.join(", ")}`;
        bubble.appendChild(hint);
      }
    }
  } catch (err) {
    bubble.textContent = "❌ " + err.message;
    bubble.style.color = "#b91c1c";
  } finally {
    setLoading(false);
    scrollToBottom();
  }
});

// Focus automatique au chargement
input.focus();
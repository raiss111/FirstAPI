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
 * @returns {HTMLElement} la bulle (pour la mettre à jour si besoin)
 */
function addMessage(role, text = "") {
    const wrapper = document.createElement("div");
    wrapper.className = `message ${role}`;

    // Avatar uniquement côté assistant (comme dans ton CSS)
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
form.addEventListener("submit", async(event) => {
    event.preventDefault();

    const text = input.value.trim();
    if (!text) return;

    // 1. Affiche le message de l'utilisateur
    addMessage("user", text);
    input.value = "";
    setLoading(true);

    // 2. Placeholder en attendant la réponse
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
            // FastAPI renvoie souvent {"detail": "..."} en cas d'erreur
            let detail = `HTTP ${res.status}`;
            try {
                const errData = await res.json();
                if (errData.detail) detail = errData.detail;
            } catch {}
            throw new Error(detail);
        }

        const data = await res.json();

        // 3. Remplace le placeholder par la vraie réponse
        //    Ton API renvoie le texte dans "response"
        bubble.textContent = data.response || "(réponse vide)";

        // 4. (Optionnel) log des infos internes pour débugger
        console.log("Intent :", data.intent);
        console.log("Scope  :", data.scope);
        console.log("State  :", data.state);
        console.log("Offers :", data.offers);
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
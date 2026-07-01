(function () {
  if (window.TridotsChatbotWidget) {
    return;
  }

  var API_URL = "/api/method/tridots_chatbot.api.chat";

  var DEFAULT_STARTERS = [
    "What ERPNext services do you offer?",
    "Tell me about Frappe HR",
    "How do I get started with ERP implementation?"
  ];

  function createElement(tag, className, text) {
    var node = document.createElement(tag);
    if (className) { node.className = className; }
    if (typeof text === "string") { node.textContent = text; }
    return node;
  }

  function formatLatency(ms) {
    if (!ms && ms !== 0) { return ""; }
    return (ms / 1000).toFixed(1) + "s";
  }

  function Widget() {
    this.history = [];
    this.hasOpened = false;
    this.isOpen = false;
    this.isPending = false;
    this.lastQuestion = "";
    this.mount();
  }

  Widget.prototype.mount = function () {
    var root = createElement("div", "tridots-chatbot-root");
    root.setAttribute("data-open", "false");
    document.body.appendChild(root);

    var launcher = createElement("div", "tridots-chatbot-launcher");
    launcher.innerHTML = '<svg width="26" height="26" viewBox="0 0 24 24" fill="none"><path d="M7 10.5h10M7 14h6m-8 6 1.6-3.2A8 8 0 1 1 20 14a8 8 0 0 1-8 8 7.9 7.9 0 0 1-3.7-.9L5 20Z" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    root.appendChild(launcher);
    this.launcher = launcher;

    var panel = createElement("div", "tridots-chatbot-panel");
    root.appendChild(panel);
    this.panel = panel;

    var header = createElement("div", "tridots-chatbot-header");
    header.innerHTML = '<div><div class="tridots-chatbot-title">Tridots Tech Assistant</div><div class="tridots-chatbot-subtitle">Ask about ERPNext, Frappe, and implementation services.</div></div><button class="tridots-chatbot-close" type="button">&times;</button>';
    panel.appendChild(header);
    this.closeButton = header.querySelector(".tridots-chatbot-close");

    var body = createElement("div", "tridots-chatbot-body");
    panel.appendChild(body);
    this.body = body;

    var footer = createElement("div", "tridots-chatbot-footer");
    footer.innerHTML = '<form class="tridots-chatbot-form"><textarea class="tridots-chatbot-input" rows="1" placeholder="Ask a question..."></textarea><button class="tridots-chatbot-send" type="submit">Send</button></form><div class="tridots-chatbot-hint">Answers are grounded in Tridots Tech website content.</div>';
    panel.appendChild(footer);

    this.form = footer.querySelector("form");
    this.input = footer.querySelector("textarea");
    this.sendButton = footer.querySelector(".tridots-chatbot-send");

    this.renderWelcome();
    this.bindEvents();
  };

  Widget.prototype.bindEvents = function () {
    var self = this;
    this.launcher.addEventListener("click", function () { self.open(); });
    this.closeButton.addEventListener("click", function () { self.close(); });
    this.form.addEventListener("submit", function (e) {
      e.preventDefault();
      self.submit(self.input.value);
    });
    this.input.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); self.submit(self.input.value); }
    });
    this.input.addEventListener("input", function () {
      self.input.style.height = "auto";
      self.input.style.height = Math.min(self.input.scrollHeight, 120) + "px";
    });
  };

  Widget.prototype.renderWelcome = function () {
    this.body.innerHTML = '<div class="tridots-chatbot-empty">Ask about services, ERPNext modules, Frappe products, or how to start a project with Tridots Tech.</div><div class="tridots-chatbot-starters"></div>';
    var starters = this.body.querySelector(".tridots-chatbot-starters");
    var self = this;
    DEFAULT_STARTERS.forEach(function (q) {
      var chip = createElement("button", "tridots-chatbot-chip", q);
      chip.type = "button";
      chip.addEventListener("click", function () { self.submit(q); });
      starters.appendChild(chip);
    });
  };

  Widget.prototype.open = function () {
    this.isOpen = true;
    this.root.setAttribute("data-open", "true");
    if (!this.hasOpened) { this.hasOpened = true; }
    this.input.focus();
  };

  Widget.prototype.close = function () {
    this.isOpen = false;
    this.root.setAttribute("data-open", "false");
  };

  Widget.prototype.scrollToBottom = function () {
    var self = this;
    requestAnimationFrame(function () { self.body.scrollTop = self.body.scrollHeight; });
  };

  Widget.prototype.appendMessage = function (role, content) {
    var wrapper = createElement("div", "tridots-chatbot-message");
    wrapper.setAttribute("data-role", role);
    var bubble = createElement("div", "tridots-chatbot-bubble");
    bubble.textContent = content || "";
    wrapper.appendChild(bubble);
    this.body.appendChild(wrapper);
    this.scrollToBottom();
    return { wrapper: wrapper, bubble: bubble, content: content || "" };
  };

  Widget.prototype.setBusy = function (busy) {
    this.isPending = busy;
    this.sendButton.disabled = busy;
    this.sendButton.textContent = busy ? "Thinking..." : "Send";
    this.input.disabled = busy;
  };

  Widget.prototype.submit = function (question) {
    var trimmed = String(question || "").trim();
    if (!trimmed || this.isPending) { return; }
    if (!this.isOpen) { this.open(); }

    if (!this.history.length) { this.body.innerHTML = ""; }

    this.lastQuestion = trimmed;
    this.input.value = "";
    this.input.style.height = "44px";

    this.appendMessage("user", trimmed);
    this.history.push({ role: "user", content: trimmed });

    var pending = this.appendMessage("assistant", "");
    pending.bubble.textContent = "...";
    this.setBusy(true);

    var self = this;
    var payload = {
      message: trimmed,
      conversation_history: this.history.slice(-6)
    };

    frappe.call({
      method: API_URL,
      args: payload,
      callback: function (r) {
        if (r.message && r.message.answer) {
          var data = r.message;
          pending.bubble.textContent = data.answer;
          pending.content = data.answer;

          var meta = createElement("div", "tridots-chatbot-meta");
          if (data.latency_ms && data.latency_ms.total) {
            meta.textContent = "Responded in " + formatLatency(data.latency_ms.total);
          }
          pending.wrapper.appendChild(meta);

          if (data.sources && data.sources.length) {
            var toggle = createElement("button", "tridots-chatbot-source-toggle", "Sources (" + data.sources.length + ")");
            toggle.type = "button";
            var sources = createElement("div", "tridots-chatbot-sources");
            data.sources.forEach(function (s) {
              var item = createElement("div", "tridots-chatbot-source-item");
              var url = s.url || "#";
              var title = s.title || url;
              item.innerHTML = '<a href="' + url + '" target="_blank" rel="noopener">' + title + "</a>";
              sources.appendChild(item);
            });
            toggle.addEventListener("click", function () {
              var open = sources.getAttribute("data-open") === "true";
              sources.setAttribute("data-open", open ? "false" : "true");
            });
            pending.wrapper.appendChild(toggle);
            pending.wrapper.appendChild(sources);
          }

          self.history.push({ role: "assistant", content: data.answer });
          self.scrollToBottom();
        } else {
          pending.bubble.textContent = "I couldn't generate a response. Please try again.";
        }
        self.setBusy(false);
      },
      error: function () {
        pending.bubble.textContent = "Something went wrong. Please try again.";
        var retry = createElement("button", "tridots-chatbot-retry", "Retry");
        retry.type = "button";
        retry.addEventListener("click", function () {
          pending.wrapper.parentNode.removeChild(pending.wrapper);
          self.submit(self.lastQuestion);
        });
        pending.wrapper.appendChild(retry);
        self.setBusy(false);
        self.scrollToBottom();
      }
    });
  };

  var style = document.createElement("style");
  style.textContent = [
    ".tridots-chatbot-root { position: fixed; right: 20px; bottom: 20px; z-index: 2147483000; font-family: system-ui, -apple-system, sans-serif; }",
    ".tridots-chatbot-launcher { width: 56px; height: 56px; border-radius: 999px; background: #007ee5; color: #fff; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 32px rgba(0,0,0,0.18); cursor: pointer; transition: transform 160ms ease, opacity 160ms ease; }",
    ".tridots-chatbot-launcher:hover { transform: translateY(-2px); }",
    ".tridots-chatbot-panel { position: absolute; right: 0; bottom: 72px; width: 380px; height: 560px; display: flex; flex-direction: column; background: #fff; border-radius: 16px; box-shadow: 0 8px 32px rgba(0,0,0,0.18); overflow: hidden; opacity: 0; pointer-events: none; transform: translateY(18px) scale(0.98); transition: transform 180ms ease, opacity 180ms ease; }",
    ".tridots-chatbot-root[data-open='true'] .tridots-chatbot-panel { opacity: 1; pointer-events: auto; transform: translateY(0) scale(1); }",
    ".tridots-chatbot-root[data-open='true'] .tridots-chatbot-launcher { opacity: 0; pointer-events: none; }",
    ".tridots-chatbot-header { background: #007ee5; color: #fff; padding: 16px 18px; display: flex; align-items: flex-start; justify-content: space-between; }",
    ".tridots-chatbot-title { font-size: 16px; font-weight: 700; }",
    ".tridots-chatbot-subtitle { font-size: 12px; opacity: 0.92; margin-top: 4px; }",
    ".tridots-chatbot-close { background: transparent; border: 0; color: inherit; font-size: 24px; line-height: 1; cursor: pointer; }",
    ".tridots-chatbot-body { flex: 1; overflow-y: auto; padding: 16px; background: linear-gradient(180deg, #f8fbff, #fff 30%); }",
    ".tridots-chatbot-empty { margin-bottom: 16px; color: #4a6178; font-size: 14px; line-height: 1.5; }",
    ".tridots-chatbot-starters { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }",
    ".tridots-chatbot-chip { border: 1px solid #c9dff5; background: #fff; color: #005aa5; border-radius: 999px; padding: 10px 12px; font: inherit; font-size: 13px; cursor: pointer; }",
    ".tridots-chatbot-message { display: flex; margin-bottom: 14px; }",
    ".tridots-chatbot-message[data-role='user'] { justify-content: flex-end; }",
    ".tridots-chatbot-message[data-role='assistant'] { justify-content: flex-start; }",
    ".tridots-chatbot-bubble { max-width: 85%; padding: 12px 14px; border-radius: 16px; font-size: 14px; line-height: 1.5; white-space: pre-wrap; word-break: break-word; }",
    ".tridots-chatbot-message[data-role='user'] .tridots-chatbot-bubble { background: #007ee5; color: #fff; border-bottom-right-radius: 4px; }",
    ".tridots-chatbot-message[data-role='assistant'] .tridots-chatbot-bubble { background: #f0f4f8; color: #1a1a1a; border-bottom-left-radius: 4px; }",
    ".tridots-chatbot-meta { font-size: 12px; color: #62758a; margin-top: 8px; }",
    ".tridots-chatbot-source-toggle { margin-top: 10px; background: transparent; border: 0; padding: 0; color: #005aa5; font: inherit; font-size: 13px; font-weight: 600; cursor: pointer; }",
    ".tridots-chatbot-sources { margin-top: 10px; border-top: 1px solid #dbe8f4; padding-top: 10px; display: none; }",
    ".tridots-chatbot-sources[data-open='true'] { display: block; }",
    ".tridots-chatbot-sources a { color: #005aa5; text-decoration: none; display: block; margin-top: 6px; }",
    ".tridots-chatbot-retry { margin-top: 10px; border: 0; border-radius: 10px; background: #007ee5; color: #fff; padding: 9px 12px; font: inherit; font-size: 13px; cursor: pointer; }",
    ".tridots-chatbot-footer { border-top: 1px solid #e4edf5; padding: 12px; background: #fff; }",
    ".tridots-chatbot-form { display: flex; gap: 10px; align-items: flex-end; }",
    ".tridots-chatbot-input { flex: 1; resize: none; min-height: 44px; max-height: 120px; border: 1px solid #cbd8e6; border-radius: 12px; padding: 11px 12px; font: inherit; font-size: 14px; color: #1a1a1a; }",
    ".tridots-chatbot-input:focus { outline: none; border-color: #007ee5; box-shadow: 0 0 0 3px rgba(0,126,229,0.14); }",
    ".tridots-chatbot-send { min-width: 92px; height: 44px; border: 0; border-radius: 12px; background: #007ee5; color: #fff; font: inherit; font-weight: 600; cursor: pointer; }",
    ".tridots-chatbot-send[disabled] { opacity: 0.6; cursor: wait; }",
    ".tridots-chatbot-hint { margin-top: 8px; font-size: 12px; color: #74879a; }",
    "@media (max-width: 479px) { .tridots-chatbot-root { inset: 0; right: 0; bottom: 0; } .tridots-chatbot-panel { position: fixed; inset: 0; width: 100vw; height: 100vh; border-radius: 0; transform: translateY(100%); } .tridots-chatbot-root[data-open='true'] .tridots-chatbot-panel { transform: translateY(0); } }"
  ].join("");
  document.head.appendChild(style);

  window.TridotsChatbotWidget = new Widget();
})();

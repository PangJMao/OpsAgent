const state = {
  user: null,
  conversations: [],
  activeConversationId: null,
  loading: false,
  sidebarCollapsed: localStorage.getItem("ops-agent-sidebar-collapsed") === "true",
};

const roleRank = { user: 1, admin: 2, root: 3 };
const viewMeta = {
  "chat-view": ["Knowledge RAG", "OpsAgent"],
  "users-view": ["Access Control", "用户管理"],
  "knowledge-view": ["Vector Knowledge", "知识管理"],
};

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const isForm = options.body instanceof FormData;
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: isForm ? {} : { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `请求失败：${response.status}`);
  }
  return payload;
}

function setRuntimeStatus(text, tone = "ok") {
  const node = $("runtime-status");
  if (!node) return;
  node.textContent = text;
  node.className = `runtime-pill ${tone}`;
}

function setLoginMessage(text) {
  $("login-error").textContent = text || "";
}

function isPrivileged() {
  return ["admin", "root"].includes(state.user?.role);
}

function formatRole(role) {
  return { root: "Root", admin: "Admin", user: "User" }[role] || role;
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("ops-agent-theme", theme);
  $("theme-toggle").textContent = theme === "light" ? "深色" : "浅色";
}

function applySidebarState() {
  document.body.classList.toggle("sidebar-collapsed", state.sidebarCollapsed);
  localStorage.setItem("ops-agent-sidebar-collapsed", String(state.sidebarCollapsed));
  $("sidebar-toggle").title = state.sidebarCollapsed ? "展开侧边栏" : "收起侧边栏";
}

function toggleSidebar() {
  if (window.matchMedia("(max-width: 860px)").matches) {
    document.body.classList.toggle("sidebar-open");
    return;
  }
  state.sidebarCollapsed = !state.sidebarCollapsed;
  applySidebarState();
}

function showLogin() {
  state.user = null;
  state.conversations = [];
  state.activeConversationId = null;
  $("workspace-view").classList.add("hidden");
  $("login-view").classList.remove("hidden");
}

function showWorkspace(user) {
  state.user = user;
  $("login-view").classList.add("hidden");
  $("workspace-view").classList.remove("hidden");
  $("current-user").textContent = `${user.username} · ${formatRole(user.role)}`;
  $("nav-users").classList.toggle("hidden", !isPrivileged());
  $("nav-knowledge").classList.toggle("hidden", !isPrivileged());
  showView("chat-view");
}

function showView(viewId) {
  document.querySelectorAll(".view-panel").forEach((node) => node.classList.add("hidden"));
  document.querySelectorAll(".nav-item").forEach((node) => node.classList.remove("active"));

  $(viewId).classList.remove("hidden");
  document.querySelector(`[data-view="${viewId}"]`)?.classList.add("active");
  $("view-eyebrow").textContent = viewMeta[viewId][0];
  $("view-title").textContent = viewMeta[viewId][1];
  document.body.classList.remove("sidebar-open");

  if (viewId === "chat-view") loadConversations();
  if (viewId === "users-view") loadUsers();
}

function renderEmptyState(title = "今天需要查询什么？", text = "面向企业制度、客户流程、内部知识和文档依据进行问答。") {
  const empty = document.createElement("div");
  empty.className = "empty-state";
  empty.innerHTML = `
    <div class="empty-logo">OA</div>
    <h3></h3>
    <p></p>
  `;
  empty.querySelector("h3").textContent = title;
  empty.querySelector("p").textContent = text;
  $("messages").replaceChildren(empty);
}

function addMessage(kind, text, citations = []) {
  const message = document.createElement("article");
  message.className = `message ${kind}`;

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = String(text ?? "");
  message.appendChild(bubble);

  if (citations.length) {
    message.appendChild(createCitationList(citations));
  }

  $("messages").appendChild(message);
  $("messages").scrollTop = $("messages").scrollHeight;
}

function createCitationList(citations) {
  const citationList = document.createElement("div");
  citationList.className = "citations";
  citations.forEach((item, index) => {
    const citation = document.createElement("span");
    citation.className = "citation";
    const title = item.title || item.document_id || item.source || `来源 ${index + 1}`;
    const chunk = item.chunk_id ? ` · ${item.chunk_id}` : "";
    citation.textContent = `${index + 1}. ${title}${chunk}`;
    citationList.appendChild(citation);
  });
  return citationList;
}

function createStreamingMessage() {
  const message = document.createElement("article");
  message.className = "message agent streaming";
  message.innerHTML = `
    <section class="thought-panel">
      <div class="message-label">思考摘要</div>
      <div class="thought-text"></div>
    </section>
    <div class="message-bubble answer-text"></div>
    <div class="citations"></div>
  `;
  $("messages").appendChild(message);
  $("messages").scrollTop = $("messages").scrollHeight;
  return message;
}

function appendStreamText(message, selector, delta) {
  const target = message.querySelector(selector);
  target.textContent += delta;
  $("messages").scrollTop = $("messages").scrollHeight;
}

function renderStreamCitations(message, citations = []) {
  const citationList = message.querySelector(".citations");
  citationList.replaceChildren(...Array.from(createCitationList(citations).children));
}

function renderMessages(messages) {
  $("messages").replaceChildren();
  if (!messages?.length) {
    renderEmptyState();
    return;
  }
  messages.forEach((message) => {
    addMessage(message.role === "user" ? "user" : "agent", message.content, message.citations || []);
  });
}

async function createConversation(title = "新对话") {
  const payload = await api("/conversations", {
    method: "POST",
    body: JSON.stringify({ title }),
  });
  state.activeConversationId = payload.conversation.conversation_id;
  return payload.conversation;
}

async function loadConversations() {
  if (!state.user) return;
  try {
    const payload = await api("/conversations");
    state.conversations = payload.conversations || [];
    if (!state.conversations.length) {
      const conversation = await createConversation();
      state.conversations = [conversation];
    }
    const stillExists = state.conversations.some(
      (item) => item.conversation_id === state.activeConversationId,
    );
    if (!state.activeConversationId || !stillExists) {
      state.activeConversationId = state.conversations[0]?.conversation_id || null;
    }
    renderConversations();
    if (state.activeConversationId) {
      await loadConversationMessages(state.activeConversationId);
    } else {
      renderEmptyState("暂无对话", "创建新对话后开始提问。");
    }
  } catch (error) {
    setRuntimeStatus(error.message, "error");
    renderEmptyState("对话不可用", error.message);
  }
}

function renderConversations() {
  const rows = state.conversations.map((conversation) => {
    const row = document.createElement("div");
    row.className = "conversation-row";
    row.classList.toggle("active", conversation.conversation_id === state.activeConversationId);

    const openButton = document.createElement("button");
    openButton.type = "button";
    openButton.className = "conversation-open";
    openButton.textContent = conversation.title || "新对话";
    openButton.addEventListener("click", async () => {
      state.activeConversationId = conversation.conversation_id;
      renderConversations();
      await loadConversationMessages(conversation.conversation_id);
      document.body.classList.remove("sidebar-open");
    });

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "conversation-delete";
    deleteButton.title = "删除对话";
    deleteButton.textContent = "×";
    deleteButton.addEventListener("click", async () => {
      await api(`/conversations/${conversation.conversation_id}`, { method: "DELETE" });
      if (state.activeConversationId === conversation.conversation_id) {
        state.activeConversationId = null;
      }
      await loadConversations();
    });

    row.append(openButton, deleteButton);
    return row;
  });
  $("conversation-list").replaceChildren(...rows);
}

async function loadConversationMessages(conversationId) {
  try {
    const payload = await api(`/conversations/${conversationId}/messages`);
    renderMessages(payload.messages || []);
  } catch (error) {
    renderEmptyState("对话加载失败", error.message);
  }
}

async function loadUsers() {
  if (!isPrivileged()) return;
  try {
    const payload = await api("/users");
    const creatableRoles = state.user.role === "root" ? ["user", "admin"] : ["user"];
    $("new-role").replaceChildren(
      ...creatableRoles.map((role) => {
        const option = document.createElement("option");
        option.value = role;
        option.textContent = role === "admin" ? "管理员" : "普通用户";
        return option;
      }),
    );

    const users = (payload.users || []).map((user) => {
      const row = document.createElement("div");
      row.className = "user-row";
      const canManage = roleRank[state.user.role] > roleRank[user.role];
      const canPromote = state.user.role === "root" && canManage;

      const meta = document.createElement("div");
      meta.className = "user-meta";
      meta.innerHTML = `<strong></strong><span></span>`;
      meta.querySelector("strong").textContent = user.username;
      meta.querySelector("span").textContent = `${formatRole(user.role)} · ${user.user_id}`;

      const roleButton = document.createElement("button");
      roleButton.type = "button";
      roleButton.className = "chip-button";
      roleButton.textContent = user.role === "admin" ? "降为普通用户" : "设为管理员";
      roleButton.disabled = !canPromote || user.role === "root";
      roleButton.addEventListener("click", async () => {
        await api(`/users/${user.user_id}/role`, {
          method: "PATCH",
          body: JSON.stringify({ role: user.role === "admin" ? "user" : "admin" }),
        });
        await loadUsers();
      });

      const deleteButton = document.createElement("button");
      deleteButton.type = "button";
      deleteButton.className = "chip-button danger";
      deleteButton.textContent = "删除";
      deleteButton.disabled = !canManage;
      deleteButton.addEventListener("click", async () => {
        await api(`/users/${user.user_id}`, { method: "DELETE" });
        await loadUsers();
      });

      row.append(meta, roleButton, deleteButton);
      return row;
    });

    $("users").replaceChildren(...users);
  } catch (error) {
    $("users").replaceChildren(statusRow(error.message));
  }
}

function statusRow(text) {
  const row = document.createElement("div");
  row.className = "upload-row";
  row.innerHTML = `<strong></strong><span></span>`;
  row.querySelector("strong").textContent = text;
  row.querySelector("span").textContent = "System";
  return row;
}

function renderUploadRow(file, status, tone = "") {
  const row = document.createElement("div");
  row.className = "upload-row";
  row.innerHTML = `<strong></strong><span></span>`;
  row.querySelector("strong").textContent = file.name;
  row.querySelector("span").textContent = status;
  if (tone) row.querySelector("span").classList.add(tone);
  $("document-status").appendChild(row);
  return row;
}

async function inspectRuntime() {
  try {
    const health = await api("/health");
    const errors = health.runtime?.startup_errors || [];
    if (health.status === "ok" && !errors.length) {
      setRuntimeStatus("System ready", "ok");
      setLoginMessage("");
      return;
    }
    setRuntimeStatus(errors[0] || "System degraded", "warn");
    setLoginMessage(errors.join("；") || "系统处于降级状态。");
  } catch {
    setRuntimeStatus("Service unavailable", "error");
    setLoginMessage("后端服务不可用，请确认系统已启动。");
  }
}

function bindEvents() {
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.addEventListener("click", () => showView(button.dataset.view));
  });

  $("theme-toggle").addEventListener("click", () => {
    const theme = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    applyTheme(theme);
  });

  $("sidebar-toggle").addEventListener("click", toggleSidebar);
  $("main-sidebar-toggle").addEventListener("click", toggleSidebar);

  $("login-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    setLoginMessage("");
    try {
      const payload = await api("/auth/login", {
        method: "POST",
        body: JSON.stringify({
          username: $("login-username").value.trim(),
          password: $("login-password").value,
        }),
      });
      showWorkspace(payload.user);
    } catch (error) {
      setLoginMessage(error.message);
    }
  });

  $("logout-button").addEventListener("click", async () => {
    await api("/auth/logout", { method: "POST" }).catch(() => {});
    showLogin();
  });

  $("new-conversation").addEventListener("click", async () => {
    try {
      await createConversation();
      await loadConversations();
      document.body.classList.remove("sidebar-open");
    } catch (error) {
      setRuntimeStatus(error.message, "error");
    }
  });

  $("chat-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = $("question").value.trim();
    if (!question || state.loading) return;
    if (!state.activeConversationId) {
      await createConversation();
      await loadConversations();
    }

    state.loading = true;
    $("send-button").disabled = true;
    if ($("messages").querySelector(".empty-state")) $("messages").replaceChildren();
    addMessage("user", question);
    const streamMessage = createStreamingMessage();
    $("question").value = "";
    autosizeQuestion();

    try {
      const payload = await streamAnswer(question, streamMessage);
      if (payload?.messages) {
        renderMessages(payload.messages);
      }
      await loadConversations();
    } catch (error) {
      streamMessage.remove();
      if (error.allowFallback) {
        await sendWithFallback(question);
      } else {
        addMessage("agent", error.message);
      }
      setRuntimeStatus(error.message, "error");
    } finally {
      state.loading = false;
      $("send-button").disabled = false;
    }
  });

  $("question").addEventListener("input", autosizeQuestion);
  $("question").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("chat-form").requestSubmit();
    }
  });

  $("user-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      await api("/users", {
        method: "POST",
        body: JSON.stringify({
          username: $("new-username").value.trim(),
          password: $("new-password").value,
          role: $("new-role").value,
        }),
      });
      $("user-form").reset();
      await loadUsers();
      setRuntimeStatus("User created", "ok");
    } catch (error) {
      setRuntimeStatus(error.message, "error");
    }
  });

  $("document-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const files = Array.from($("document-file").files || []);
    if (!files.length) return;
    $("document-status").replaceChildren();
    for (const file of files) {
      const row = renderUploadRow(file, "上传中");
      try {
        const bytes = await file.arrayBuffer();
        const payload = await api(`/rag/documents?filename=${encodeURIComponent(file.name)}`, {
          method: "POST",
          headers: { "Content-Type": "application/octet-stream" },
          body: bytes,
        });
        row.querySelector("span").textContent = `已入库 · ${payload.chunks || payload.chunk_count || 0} chunks`;
        row.querySelector("span").classList.add("success");
      } catch (error) {
        row.querySelector("span").textContent = error.message;
        row.querySelector("span").classList.add("danger");
      }
    }
    $("document-file").value = "";
  });
}

async function streamAnswer(question, message) {
  const response = await fetch(`/conversations/${state.activeConversationId}/messages/stream`, {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!response.ok || !response.body) {
    const error = new Error(`请求失败：${response.status}`);
    error.allowFallback = true;
    throw error;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let donePayload = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const part of parts) {
      const event = parseSseEvent(part);
      if (!event) continue;
      if (event.type === "thought") appendStreamText(message, ".thought-text", event.data.delta || "");
      if (event.type === "answer") appendStreamText(message, ".answer-text", event.data.delta || "");
      if (event.type === "citations") renderStreamCitations(message, event.data.citations || []);
      if (event.type === "done") donePayload = event.data;
      if (event.type === "error") throw new Error(event.data.detail || "流式生成失败");
    }
  }

  message.classList.remove("streaming");
  return donePayload;
}

function parseSseEvent(raw) {
  const lines = raw.split("\n");
  const typeLine = lines.find((line) => line.startsWith("event:"));
  const dataLine = lines.find((line) => line.startsWith("data:"));
  if (!typeLine || !dataLine) return null;
  return {
    type: typeLine.slice(6).trim(),
    data: JSON.parse(dataLine.slice(5).trim()),
  };
}

async function sendWithFallback(question) {
  const payload = await api(`/conversations/${state.activeConversationId}/messages`, {
    method: "POST",
    body: JSON.stringify({ question }),
  });
  renderMessages(payload.messages || []);
}

function autosizeQuestion() {
  const input = $("question");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 190)}px`;
}

async function bootstrap() {
  applyTheme(localStorage.getItem("ops-agent-theme") || "dark");
  applySidebarState();
  bindEvents();
  renderEmptyState();
  await inspectRuntime();
  try {
    const payload = await api("/auth/me");
    showWorkspace(payload.user);
  } catch {
    showLogin();
  }
}

bootstrap();

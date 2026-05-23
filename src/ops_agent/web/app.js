const state = { user: null };

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: options.body instanceof FormData ? {} : { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `请求失败：${response.status}`);
  }
  return payload;
}

function showWorkspace(user) {
  state.user = user;
  $("login-view").classList.add("hidden");
  $("workspace-view").classList.remove("hidden");
  $("current-user").textContent = `${user.username} · ${user.role}`;
  $("admin-panel").classList.toggle("hidden", !["admin", "root"].includes(user.role));
  if (["admin", "root"].includes(user.role)) {
    loadUsers();
  }
}

function showLogin() {
  state.user = null;
  $("workspace-view").classList.add("hidden");
  $("login-view").classList.remove("hidden");
}

async function renderSystemStatus() {
  try {
    const health = await api("/health");
    if (health.status === "degraded") {
      $("login-error").textContent = `系统降级运行：${(health.runtime?.startup_errors || []).join("；")}`;
    }
  } catch {
    $("login-error").textContent = "系统状态不可用，请确认后端服务是否启动。";
  }
}

function addMessage(kind, text, citations = []) {
  const node = document.createElement("div");
  node.className = `message ${kind}`;
  node.textContent = text;
  if (citations.length) {
    const citationNode = document.createElement("div");
    citationNode.className = "citations";
    citationNode.textContent = `引用来源：${citations
      .map((item) => `${item.title || item.document_id} / ${item.chunk_id}`)
      .join("；")}`;
    node.appendChild(citationNode);
  }
  $("messages").appendChild(node);
  $("messages").scrollTop = $("messages").scrollHeight;
}

async function loadUsers() {
  const payload = await api("/users");
  $("users").replaceChildren(
    ...payload.users.map((user) => {
      const row = document.createElement("div");
      row.className = "user-row";
      const label = document.createElement("span");
      label.textContent = `${user.username} · ${user.role}`;
      const roleButton = document.createElement("button");
      roleButton.className = "secondary";
      roleButton.textContent = user.role === "admin" ? "降为普通" : "设为管理员";
      roleButton.disabled = state.user.role !== "root" || user.role === "root";
      roleButton.onclick = async () => {
        await api(`/users/${user.user_id}/role`, {
          method: "PATCH",
          body: JSON.stringify({ role: user.role === "admin" ? "user" : "admin" }),
        });
        await loadUsers();
      };
      const deleteButton = document.createElement("button");
      deleteButton.className = "secondary";
      deleteButton.textContent = "删除";
      deleteButton.disabled = user.role === "root";
      deleteButton.onclick = async () => {
        await api(`/users/${user.user_id}`, { method: "DELETE" });
        await loadUsers();
      };
      row.append(label, roleButton, deleteButton);
      return row;
    }),
  );
}

$("login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("login-error").textContent = "";
  try {
    const payload = await api("/auth/login", {
      method: "POST",
      body: JSON.stringify({
        username: $("login-username").value,
        password: $("login-password").value,
      }),
    });
    showWorkspace(payload.user);
  } catch (error) {
    $("login-error").textContent = error.message;
  }
});

$("logout-button").addEventListener("click", async () => {
  await api("/auth/logout", { method: "POST" });
  showLogin();
});

$("chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = $("question").value.trim();
  if (!question) return;
  $("question").value = "";
  addMessage("user", question);
  try {
    const answer = await api("/rag/ask", { method: "POST", body: JSON.stringify({ question }) });
    addMessage("agent", answer.answer, answer.citations || []);
  } catch (error) {
    addMessage("agent", error.message);
  }
});

$("user-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  await api("/users", {
    method: "POST",
    body: JSON.stringify({
      username: $("new-username").value,
      password: $("new-password").value,
      role: $("new-role").value,
    }),
  });
  $("user-form").reset();
  await loadUsers();
});

$("document-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = $("document-file").files[0];
  if (!file) return;
  $("document-status").textContent = "正在上传并写入知识库...";
  try {
    const payload = await fetch(`/rag/documents?filename=${encodeURIComponent(file.name)}`, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/octet-stream" },
      body: await file.arrayBuffer(),
    }).then(async (response) => {
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || `请求失败：${response.status}`);
      return data;
    });
    $("document-status").textContent = `已入库：${payload.title}，片段数 ${payload.chunk_count}`;
  } catch (error) {
    $("document-status").textContent = error.message;
  }
});

api("/auth/me")
  .then((payload) => showWorkspace(payload.user))
  .catch(() => {
    showLogin();
    renderSystemStatus();
  });

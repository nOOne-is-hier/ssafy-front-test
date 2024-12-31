import "regenerator-runtime/runtime"; // if needed for async/await in older browsers

const chatContainer = document.getElementById("chat-container");
const messageForm = document.getElementById("message-form");
const userInput = document.getElementById("user-input");
const apiSelector = document.getElementById("api-selector");
const newChatBtn = document.getElementById("new-chat-btn");

const BASE_URL = process.env.API_ENDPOINT;

let db;

async function initDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("myChatDB", 1);
    request.onupgradeneeded = function (e) {
      db = e.target.result;
      if (!db.objectStoreNames.contains("chats")) {
        db.createObjectStore("chats", { keyPath: "id", autoIncrement: true });
      }
      if (!db.objectStoreNames.contains("metadata")) {
        db.createObjectStore("metadata", { keyPath: "key" });
      }
    };
    request.onsuccess = function (e) {
      db = e.target.result;
      resolve();
    };
    request.onerror = function (e) {
      reject(e);
    };
  });
}

async function saveMessage(role, content) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction("chats", "readwrite");
    const store = tx.objectStore("chats");
    store.add({ role, content });
    tx.oncomplete = () => resolve();
    tx.onerror = (e) => reject(e);
  });
}

async function getAllMessages() {
  return new Promise((resolve, reject) => {
    const tx = db.transaction("chats", "readonly");
    const store = tx.objectStore("chats");
    const req = store.getAll();
    req.onsuccess = () => resolve(req.result);
    req.onerror = (e) => reject(e);
  });
}

async function saveMetadata(key, value) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction("metadata", "readwrite");
    const store = tx.objectStore("metadata");
    store.put({ key, value });
    tx.oncomplete = () => resolve();
    tx.onerror = (e) => reject(e);
  });
}

async function getMetadata(key) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction("metadata", "readonly");
    const store = tx.objectStore("metadata");
    const req = store.get(key);
    req.onsuccess = () => resolve(req.result ? req.result.value : null);
    req.onerror = (e) => reject(e);
  });
}

async function clearAllData() {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(["chats", "metadata"], "readwrite");
    tx.objectStore("chats").clear();
    tx.objectStore("metadata").clear();
    tx.oncomplete = () => resolve();
    tx.onerror = (e) => reject(e);
  });
}

function createMessageBubble(content, sender = "user") {
  const wrapper = document.createElement("div");
  wrapper.classList.add("mb-6", "flex", "items-start");
  
  // 메시지 보낸 주체에 따라 정렬을 다르게
  if (sender === "assistant") {
    // 좌측 배치 (기본 정렬)
    wrapper.classList.add("flex-row", "space-x-3");
  } else {
    // 우측 배치 (정렬 반전)
    wrapper.classList.add("flex-row-reverse", "space-x-3", "space-x-reverse");
  }

  const avatar = document.createElement("div");
  avatar.classList.add(
    "w-10",
    "h-10",
    "rounded-full",
    "flex-shrink-0",
    "flex",
    "items-center",
    "justify-center",
    "font-bold",
    "text-white"
  );

  if (sender === "assistant") {
    avatar.classList.add("bg-gradient-to-br", "from-green-400", "to-green-600");
    avatar.textContent = "봇";
  } else {
    avatar.classList.add("bg-gradient-to-br", "from-blue-500", "to-blue-700");
    avatar.textContent = "나";
  }

  const bubble = document.createElement("div");
  bubble.classList.add(
    "max-w-full",
    "md:max-w-2xl",
    "p-3",
    "rounded-lg",
    "whitespace-pre-wrap",
    "leading-relaxed",
    "shadow-sm"
  );

  if (sender === "assistant") {
    bubble.classList.add("bg-gray-200", "text-gray-900");
  } else {
    bubble.classList.add("bg-blue-600", "text-white");
  }

  bubble.textContent = content;

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  return wrapper;
}


function scrollToBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

/**
 * 서버에 메시지를 보내고, 응답을 받아오는 함수
 */
async function getAssistantResponse(userMessage) {
  const mode = apiSelector.value;  // "assistant" 또는 "naive"
  let url;
  let payload;

  const thread_id = await getMetadata("thread_id"); // thread_id를 항상 읽어옴
  // if (!thread_id && mode === "assistant") {
  //     console.error("Missing thread_id in assistant mode.");
  //     return "Error: Thread ID is missing.";
  // }

  if (mode === "assistant") {
      payload = { message: userMessage };
      if (thread_id) {
          payload.thread_id = thread_id;
      }
      url = `${BASE_URL}/assistant`;
  } else {
      payload = { message: userMessage };
      url = `${BASE_URL}/chat`;
  }

  const response = await fetch(url, {
      method: "POST",
      headers: {
          "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
  });

  if (!response.ok) {
      throw new Error("Network response was not ok");
  }

  const data = await response.json();

  // thread_id를 업데이트
  if (mode === "assistant" && data.thread_id) {
      await saveMetadata("thread_id", data.thread_id);
  }

  return data.reply;
}


messageForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = userInput.value.trim();
  if (!message) return;

  // 사용자 메시지 화면에 추가 + DB 저장
  chatContainer.appendChild(createMessageBubble(message, "user"));
  await saveMessage("user", message);

  userInput.value = "";
  scrollToBottom();

  try {

    showLoading();
    // 서버에 메시지를 보내고 응답 받기
    const response = await getAssistantResponse(message);

    // 서버에서 받은 assistant 메시지 화면 + DB 저장
    chatContainer.appendChild(createMessageBubble(response, "assistant"));
    await saveMessage("assistant", response);
    scrollToBottom();
  } catch (error) {
    console.error("Error fetching assistant response:", error);
    const errMsg = "Error fetching response. Check console.";
    chatContainer.appendChild(createMessageBubble(errMsg, "assistant"));
    await saveMessage("assistant", errMsg);
    scrollToBottom();
  } finally {
    hideLoading();
  } 
});

async function loadExistingMessages() {
  const allMsgs = await getAllMessages();
  for (const msg of allMsgs) {
    chatContainer.appendChild(createMessageBubble(msg.content, msg.role));
  }
  scrollToBottom();
}

newChatBtn.addEventListener("click", async () => {
  // DB 및 채팅창 clear
  await clearAllData();
  await saveMetadata("thread_id", null); // thread_id 초기화
  chatContainer.innerHTML = "";
  // 이제 새 채팅을 시작할 수 있음
});

// 초기화 후 기존 메시지 로드
initDB().then(loadExistingMessages);

console.log(BASE_URL);

// 로딩 이미지 엘리먼트 가져오기
const loadingImg = document.querySelector(".loading");

// 로딩 표시 함수
function showLoading() {
  loadingImg.style.display = "flex";
}

// 로딩 숨김 함수
function hideLoading() {
  loadingImg.style.display = "none";
}


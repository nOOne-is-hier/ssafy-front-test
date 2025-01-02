// app.js

import "regenerator-runtime/runtime"; // if needed for async/await in older browsers

const chatContainer = document.getElementById("chat-container");
const messageForm = document.getElementById("message-form");
const userInput = document.getElementById("user-input");
const newChatBtn = document.getElementById("new-chat-btn");
const modelSelector = document.getElementById("api-selector");
const manualBtn = document.getElementById("manual-btn");
const modal = document.getElementById("manual-modal");
const closeBtn = document.getElementById("close-btn");

const BASE_URL = process.env.API_ENDPOINT;

let messages = []; // 대화 히스토리 배열
const MAX_MESSAGES = 5; // 유지할 메시지 수

// Create a chat message bubble element
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
    avatar.textContent = modelSelector.value;
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

// Scroll the chat container to the bottom
function scrollToBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

/**
 * 서버에 메시지를 보내고, 응답을 받아오는 함수
 */
async function getAssistantResponse() {
  const modelId = parseInt(modelSelector.value, 10); // Get selected model ID
  const payload = { messages, model_id: modelId };

  const response = await fetch(`${BASE_URL}/chat`, {
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

  return data.reply;
}

/**
 * 매뉴얼 버튼 눌렀을 때->매뉴얼 모달창 나오게
 */
manualBtn.addEventListener("click", () => {
  modal.classList.remove("hidden"); // hidden 클래스 제거
});
// 매뉴얼 모달창 닫기
closeBtn.addEventListener("click", () => {
  modal.classList.add("hidden"); // hidden 클래스 추가
});

// 모달창 외부 클릭 시 닫기
modal.addEventListener("click", (event) => {
  if (event.target === modal) {
    modal.classList.add("hidden");
  }
});



// Handle message form submission
messageForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = userInput.value.trim();
  if (!message) return;

  // 사용자 메시지 화면에 추가
  chatContainer.appendChild(createMessageBubble(message, "user"));
  scrollToBottom();

  // 메시지를 배열에 추가
  messages.push({ role: "user", content: message });

  // 최근 MAX_MESSAGES개의 메시지만 유지
  if (messages.length > MAX_MESSAGES) {
    messages = messages.slice(-MAX_MESSAGES);
  }

  // 메시지를 로컬 스토리지에 저장
  saveMessagesToLocalStorage();

  userInput.value = "";

  try {
    showLoading();
    // 서버에 메시지를 보내고 응답 받기
    const response = await getAssistantResponse();

    // 서버에서 받은 assistant 메시지 화면에 추가
    chatContainer.appendChild(createMessageBubble(response, "assistant"));
    scrollToBottom();

    // 메시지를 배열에 추가
    messages.push({ role: "assistant", content: response });

    // 최근 MAX_MESSAGES개의 메시지만 유지
    if (messages.length > MAX_MESSAGES) {
      messages = messages.slice(-MAX_MESSAGES);
    }

    // 메시지를 로컬 스토리지에 저장
    saveMessagesToLocalStorage();
  } catch (error) {
    console.error("Error fetching assistant response:", error);
    const errMsg = "응답을 가져오는 중 오류가 발생했습니다. 콘솔을 확인해주세요.";
    chatContainer.appendChild(createMessageBubble(errMsg, "assistant"));
    scrollToBottom();

    // 에러 메시지를 배열에 추가
    messages.push({ role: "assistant", content: errMsg });

    // 최근 MAX_MESSAGES개의 메시지만 유지
    if (messages.length > MAX_MESSAGES) {
      messages = messages.slice(-MAX_MESSAGES);
    }

    // 메시지를 로컬 스토리지에 저장
    saveMessagesToLocalStorage();
  } finally {
    hideLoading();
  }
});

// Clear chat and reset messages for a new conversation
newChatBtn.addEventListener("click", () => {
  messages = [];
  chatContainer.innerHTML = "";
  localStorage.removeItem("chatMessages");
});

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

// 메시지 배열을 로컬 스토리지에 저장
function saveMessagesToLocalStorage() {
  localStorage.setItem("chatMessages", JSON.stringify(messages));
}

// 로컬 스토리지에서 메시지 불러오기
function loadMessagesFromLocalStorage() {
  const storedMessages = localStorage.getItem("chatMessages");
  if (storedMessages) {
    messages = JSON.parse(storedMessages);
    chatContainer.innerHTML = "";
    messages.forEach(msg => {
      chatContainer.appendChild(createMessageBubble(msg.content, msg.role));
    });
    scrollToBottom();
  }
}

// 페이지 로드 시 메시지 불러오기
window.onload = () => {
  loadMessagesFromLocalStorage();
};

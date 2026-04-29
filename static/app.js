/**
 * MEOW AI Tutor — Frontend Logic (Secure Backend Architecture)
 * Chat engine, IndexedDB history, session management.
 * All API calling happens safely on the backend.
 */

// ──────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────
const API_URL = '/api/chat';
const DB_NAME = 'MeowDB';
const DB_VERSION = 1;
const STORE_NAME = 'sessions';

// ──────────────────────────────────────────────
// State
// ──────────────────────────────────────────────
let currentSessionId = generateId();
let currentMessages = [];
let db = null;
let isWaiting = false;

// ──────────────────────────────────────────────
// DOM Elements
// ──────────────────────────────────────────────
const scrollContainer = document.getElementById('scroll-container');
const heroSection = document.getElementById('hero-section');
const chatArea = document.getElementById('chat-area');
const messagesContainer = document.getElementById('messages-container');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const newChatSidebarBtn = document.getElementById('new-chat-sidebar-btn');
const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebar-toggle');
const sidebarClose = document.getElementById('sidebar-close');
const sidebarOverlay = document.getElementById('sidebar-overlay');
const historyList = document.getElementById('history-list');

// ──────────────────────────────────────────────
// Initialization
// ──────────────────────────────────────────────
async function init() {
    await openDB();
    await cleanupOldSessions();
    await refreshHistorySidebar();
    
    // Auto-load most recent session if available
    const sessions = await getAllSessions();
    if (sessions.length > 0 && currentMessages.length === 0) {
        await loadSession(sessions[0].id);
    }
    
    userInput.focus();
}

// ──────────────────────────────────────────────
// IndexedDB for Chat History
// ──────────────────────────────────────────────
function openDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, DB_VERSION);
        request.onupgradeneeded = (e) => {
            const db = e.target.result;
            if (!db.objectStoreNames.contains(STORE_NAME)) {
                const store = db.createObjectStore(STORE_NAME, { keyPath: 'id' });
                store.createIndex('updatedAt', 'updatedAt');
            }
        };
        request.onsuccess = (e) => {
            db = e.target.result;
            resolve(db);
        };
        request.onerror = (e) => reject(e.target.error);
    });
}

async function saveSession() {
    if (!db || currentMessages.length === 0) return;
    const firstUserMsg = currentMessages.find(m => m.role === 'user');
    const title = firstUserMsg
        ? firstUserMsg.content.substring(0, 45) + (firstUserMsg.content.length > 45 ? '...' : '')
        : 'New Chat';

    const session = {
        id: currentSessionId,
        title,
        messages: currentMessages,
        createdAt: currentMessages[0]?.timestamp || Date.now(),
        updatedAt: Date.now(),
    };

    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        tx.objectStore(STORE_NAME).put(session);
        tx.oncomplete = () => resolve();
        tx.onerror = (e) => reject(e.target.error);
    });
}

async function getAllSessions() {
    if (!db) return [];
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readonly');
        const store = tx.objectStore(STORE_NAME);
        const request = store.getAll();
        request.onsuccess = () => {
            const sessions = request.result.sort((a, b) => b.updatedAt - a.updatedAt);
            resolve(sessions);
        };
        request.onerror = (e) => reject(e.target.error);
    });
}

async function getSession(id) {
    if (!db) return null;
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readonly');
        const request = tx.objectStore(STORE_NAME).get(id);
        request.onsuccess = () => resolve(request.result);
        request.onerror = (e) => reject(e.target.error);
    });
}

async function cleanupOldSessions() {
    if (!db) return;
    const oneMonthAgo = Date.now() - (30 * 24 * 60 * 60 * 1000);
    const sessions = await getAllSessions();
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    let deletedCount = 0;
    
    sessions.forEach(session => {
        if (session.updatedAt < oneMonthAgo) {
            store.delete(session.id);
            deletedCount++;
        }
    });
    
    if (deletedCount > 0) {
        console.log(`Deleted ${deletedCount} old session(s)`);
    }
}

// ──────────────────────────────────────────────
// Chat UI Layer
// ──────────────────────────────────────────────
function showChatMode() {
    heroSection.classList.add('hidden');
    chatArea.classList.add('visible');
}

function showHeroMode() {
    heroSection.classList.remove('hidden');
    chatArea.classList.remove('visible');
}

function scrollToBottom() {
    scrollContainer.scrollTo({
        top: scrollContainer.scrollHeight,
        behavior: 'smooth'
    });
}

function addMessageToUI(role, content) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? 'U' : 'M';
    msgDiv.appendChild(avatar);

    const wrapper = document.createElement('div');
    wrapper.className = 'message-content-wrapper';
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    if (role === 'ai') {
        bubble.innerHTML = renderMarkdown(content);
        // Models used footer removed for branding reasons
    } else {
        bubble.textContent = content;
    }

    wrapper.appendChild(bubble);
    msgDiv.appendChild(wrapper);
    messagesContainer.appendChild(msgDiv);
    
    setTimeout(scrollToBottom, 50);
}

function showTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator message ai';
    indicator.id = 'typing-indicator';
    
    indicator.innerHTML = `
        <div class="message-avatar">M</div>
        <div class="message-content-wrapper">
            <div class="typing-dots">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;
    
    messagesContainer.appendChild(indicator);
    setTimeout(scrollToBottom, 50);
}

function removeTypingIndicator() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
}

// Use Marked.js for rendering
function renderMarkdown(text) {
    if (!text) return '';
    try {
        if (typeof marked !== 'undefined') {
            return marked.parse(text, { breaks: true, gfm: true });
        }
        return `<p>${escapeHtml(text)}</p>`;
    } catch (e) {
        console.error("Markdown parsing error", e);
        return `<p>${escapeHtml(text)}</p>`;
    }
}

// ──────────────────────────────────────────────
// Chat Logic & API Calls
// ──────────────────────────────────────────────
async function sendMessage() {
    const text = userInput.value.trim();
    if (!text || isWaiting) return;

    isWaiting = true;
    sendBtn.disabled = true;
    userInput.value = '';
    userInput.style.height = 'auto'; // Reset height
    updateSendButtonState();

    showChatMode();

    // Add user message
    currentMessages.push({
        role: 'user',
        content: text,
        timestamp: Date.now(),
    });
    addMessageToUI('user', text);
    showTypingIndicator();

    try {
        const historyMessages = currentMessages
            .filter(msg => msg.role !== 'system')
            .map(msg => ({
                role: msg.role === 'ai' ? 'assistant' : 'user',
                content: msg.content
            }));
            
        const resp = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: text,
                session_id: currentSessionId,
                history: historyMessages
            }),
        });

        if (!resp.ok) {
            throw new Error(`Server error: ${resp.status}`);
        }

        const data = await resp.json();

        removeTypingIndicator();
        addMessageToUI('ai', data.response);

        currentMessages.push({
            role: 'ai',
            content: data.response,
            timestamp: Date.now(),
        });

        await saveSession();
        await refreshHistorySidebar();
    } catch (err) {
        removeTypingIndicator();
        const errorMsg = `⚠️ **Connection Error**\n\nCould not reach the MEOW backend.\n\n*Error: ${err.message}*`;
        addMessageToUI('ai', errorMsg);
        currentMessages.push({
            role: 'ai',
            content: errorMsg,
            timestamp: Date.now(),
        });
    }

    isWaiting = false;
    userInput.focus();
    updateSendButtonState();
}

// ──────────────────────────────────────────────
// Session Management & Sidebar Handling
// ──────────────────────────────────────────────
function startNewChat() {
    currentSessionId = generateId();
    currentMessages = [];
    messagesContainer.innerHTML = '';
    showHeroMode();
    closeSidebar();
    userInput.value = '';
    userInput.style.height = 'auto';
    updateSendButtonState();
    userInput.focus();
    refreshHistorySidebar();
}

async function loadSession(sessionId) {
    const session = await getSession(sessionId);
    if (!session) return;

    currentSessionId = session.id;
    currentMessages = session.messages;
    messagesContainer.innerHTML = '';
    
    showChatMode();

    for (const msg of session.messages) {
        addMessageToUI(msg.role, msg.content);
    }

    closeSidebar();
    setTimeout(scrollToBottom, 100);
}

function openSidebar() {
    sidebar.classList.add('open');
    sidebarOverlay.classList.add('visible');
    refreshHistorySidebar();
}

function closeSidebar() {
    sidebar.classList.remove('open');
    sidebarOverlay.classList.remove('visible');
}

async function refreshHistorySidebar() {
    const sessions = await getAllSessions();

    if (sessions.length === 0) {
        historyList.innerHTML = `
            <div class="history-empty">
                No past chats
            </div>
        `;
        return;
    }

    historyList.innerHTML = sessions.map(s => `
        <div class="history-item ${s.id === currentSessionId ? 'active' : ''}"
             onclick="loadSession('${s.id}')"
             title="${escapeHtml(s.title)}">
            ${escapeHtml(s.title)}
        </div>
    `).join('');
}

// ──────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────
function generateId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function updateSendButtonState() {
    if (userInput.value.trim().length > 0 && !isWaiting) {
        sendBtn.disabled = false;
    } else {
        sendBtn.disabled = true;
    }
}

// ──────────────────────────────────────────────
// Event Listeners
// ──────────────────────────────────────────────
sendBtn.addEventListener('click', sendMessage);

userInput.addEventListener('input', () => {
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 200) + 'px';
    updateSendButtonState();
});

userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

newChatSidebarBtn.addEventListener('click', startNewChat);
sidebarToggle.addEventListener('click', openSidebar);
sidebarClose.addEventListener('click', closeSidebar);
sidebarOverlay.addEventListener('click', closeSidebar);

// Boot
init();

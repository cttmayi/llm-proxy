// API Monitor JavaScript

const API_BASE_URL = window.location.origin;
let autoRefreshInterval = null;
let currentCalls = [];
let filterModel = '';
let filterStatus = '';

// Initialize the monitor
document.addEventListener('DOMContentLoaded', function() {
    loadInitialData();
    setupEventListeners();
    startAutoRefresh();
});

// Setup event listeners
function setupEventListeners() {
    document.getElementById('filterModel').addEventListener('change', handleFilterChange);
    document.getElementById('filterStatus').addEventListener('change', handleFilterChange);
    document.getElementById('autoScroll').addEventListener('change', toggleAutoScroll);
}

// Load initial data
async function loadInitialData() {
    await Promise.all([
        loadAPICalls(),
        loadStats(),
        loadModels()
    ]);
}

// Load API calls with deep comparison to prevent unnecessary updates
async function loadAPICalls() {
    try {
        const response = await fetch(`${API_BASE_URL}/web/api/calls?limit=50`);
        const data = await response.json();
        
        if (response.ok) {
            // Quick check: compare lengths first
            if (data.calls.length === currentCalls.length) {
                // Deep comparison of key fields only
                const hasChanges = data.calls.some((newCall, index) => {
                    const oldCall = currentCalls[index];
                    if (!oldCall) return true;
                    
                    // Compare essential fields
                    return newCall.id !== oldCall.id ||
                           newCall.timestamp !== oldCall.timestamp ||
                           newCall.status_code !== oldCall.status_code ||
                           newCall.duration_ms !== oldCall.duration_ms ||
                           newCall.model !== oldCall.model;
                });
                
                if (!hasChanges) {
                    return; // No changes, skip update
                }
            }
            
            currentCalls = data.calls;
            renderAPICalls();
        }
    } catch (error) {
        console.error('Error loading API calls:', error);
    }
}

// Load statistics
async function loadStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/web/api/stats`);
        const data = await response.json();
        
        if (response.ok) {
            updateStats(data);
        }
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Load available models
async function loadModels() {
    try {
        const response = await fetch(`${API_BASE_URL}/v1/models`);
        const data = await response.json();
        
        if (response.ok && data.data) {
            populateModelFilter(data.data);
        }
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

// Populate model filter dropdown
function populateModelFilter(models) {
    const select = document.getElementById('filterModel');
    const modelNames = [...new Set(models.map(m => m.id))].sort();
    
    select.innerHTML = '<option value="">所有模型</option>';
    modelNames.forEach(model => {
        const option = document.createElement('option');
        option.value = model;
        option.textContent = model;
        select.appendChild(option);
    });
}

// Update statistics display
function updateStats(stats) {
    document.getElementById('totalCalls').textContent = stats.total_calls || 0;
    document.getElementById('errorRate').textContent = 
        ((stats.error_rate || 0) * 100).toFixed(1) + '%';
    document.getElementById('avgDuration').textContent = 
        Math.round(stats.avg_duration || 0) + 'ms';
    document.getElementById('activeModels').textContent = 
        Object.keys(stats.model_stats || {}).length;
}

// Render API calls with content comparison to prevent flickering
let lastRenderedHTML = '';

function renderAPICalls() {
    const container = document.getElementById('apiCalls');
    
    if (currentCalls.length === 0) {
        const emptyHTML = '<div class="text-center py-8 text-gray-500">暂无API调用数据</div>';
        if (lastRenderedHTML !== emptyHTML) {
            container.innerHTML = emptyHTML;
            lastRenderedHTML = emptyHTML;
        }
        return;
    }
    
    const filteredCalls = filterCalls();
    const newHTML = filteredCalls.map(call => renderAPICall(call)).join('');
    
    // Only update if content has actually changed
    if (newHTML !== lastRenderedHTML) {
        container.innerHTML = newHTML;
        lastRenderedHTML = newHTML;
        
        // Scroll to bottom if auto-scroll is enabled
        if (document.getElementById('autoScroll').checked) {
            container.scrollTop = container.scrollHeight;
        }
    }
}

// Filter calls based on selected filters
function filterCalls() {
    return currentCalls.filter(call => {
        const modelMatch = !filterModel || call.model === filterModel;
        const statusMatch = !filterStatus || 
            (filterStatus === 'success' && call.status_code < 400) ||
            (filterStatus === 'error' && call.status_code >= 400);
        return modelMatch && statusMatch;
    });
}

// Render individual API call
function renderAPICall(call) {
    const isSuccess = call.status_code < 400;
    const methodClass = `method-${call.method}`;
    const statusClass = isSuccess ? 'status-success' : 'status-error';
    
    const time = new Date(call.timestamp * 1000).toLocaleTimeString('zh-CN');
    
// Extract all messages from request and response
    let messagesTable = '';
    
    let messages = [];
    
    // Extract request messages
    if (call.request_body && call.request_body.messages && call.request_body.messages.length > 0) {
        messages = call.request_body.messages.map(msg => ({
            role: msg.role || 'user',
            content: msg.content || ''
        }));
    }
    
    // Extract response messages
    if (call.response_body && call.response_body.choices && call.response_body.choices.length > 0) {
        const responseMessages = call.response_body.choices.map(choice => ({
            role: choice.message?.role || 'assistant',
            content: choice.message?.content || choice.text || ''
        }));
        messages = messages.concat(responseMessages);
    }
    
    if (messages.length > 0) {
        messagesTable = `
            <div class="messages-table mt-2">
                <table class="min-w-full divide-y divide-gray-200 text-sm">
                    <thead class="bg-gray-50">
                        <tr>
                            <th class="px-3 py-2 text-left font-medium text-gray-700">Role</th>
                            <th class="px-3 py-2 text-left font-medium text-gray-700">Content</th>
                        </tr>
                    </thead>
                    <tbody class="bg-white divide-y divide-gray-200">
                        ${messages.map(msg => `
                            <tr>
                                <td class="px-3 py-2 whitespace-nowrap text-xs font-medium text-gray-900">${escapeHtml(msg.role)}</td>
                                <td class="px-3 py-2 text-xs text-gray-700">
                                    <div class="max-w-md truncate" title="${escapeHtml(msg.content).replace(/\n\n/g, '\n')}">
                                        ${escapeHtml(msg.content)}
                                    </div>
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
    }
    
    return `
        <div class="api-call-item ${statusClass} p-4" data-id="${call.id}">
            <div class="flex items-center justify-between mb-2">
                <div class="flex items-center space-x-3">
                    <span class="method-badge ${methodClass}">${call.method}</span>
                    <span class="text-sm font-medium text-gray-900">${call.path}</span>
                    ${call.model ? `<span class="provider-badge provider-openai text-xs">${call.model}</span>` : ''}
                </div>
                <div class="flex items-center space-x-3">
                    <span class="text-sm text-gray-500 timestamp">${time}</span>
                    <span class="text-sm font-medium ${isSuccess ? 'text-green-600' : 'text-red-600'}">
                        ${call.status_code}
                    </span>
                    <span class="text-sm text-gray-600 duration">${call.duration_ms}ms</span>
                    <button onclick="showDetail('${call.id}')" class="text-blue-600 hover:text-blue-800 text-sm">
                        详情
                    </button>
                </div>
            </div>
            
            ${messagesTable}
            
            ${call.error ? `<div class="text-sm text-red-600 mb-2">${call.error}</div>` : ''}
        </div>
    `;
}

// Handle filter changes
function handleFilterChange() {
    filterModel = document.getElementById('filterModel').value;
    filterStatus = document.getElementById('filterStatus').value;
    renderAPICalls();
}

// Show call detail modal
async function showDetail(callId) {
    const call = currentCalls.find(c => c.id === callId);
    if (!call) return;
    
    const modal = document.getElementById('detailModal');
    const content = document.getElementById('modalContent');
    
    const time = new Date(call.timestamp * 1000).toLocaleString('zh-CN');
    
    // Extract full messages
    let inputMessages = [];
    let outputMessages = [];
    
    if (call.request_body && call.request_body.messages) {
        inputMessages = call.request_body.messages.map(msg => ({
            role: msg.role || 'user',
            content: msg.content || ''
        }));
    }
    
    if (call.response_body && call.response_body.choices) {
        outputMessages = call.response_body.choices.map(choice => ({
            content: choice.message?.content || choice.text || '',
            role: choice.message?.role || 'assistant'
        }));
    }
    
    content.innerHTML = `
        <div class="space-y-6">
            <div class="grid grid-cols-2 gap-4">
                <div>
                    <h4 class="font-semibold text-gray-700 mb-2">基本信息</h4>
                    <table class="text-sm">
                        <tr><td class="font-medium text-gray-600">时间:</td><td>${time}</td></tr>
                        <tr><td class="font-medium text-gray-600">方法:</td><td>${call.method}</td></tr>
                        <tr><td class="font-medium text-gray-600">路径:</td><td>${call.path}</td></tr>
                        <tr><td class="font-medium text-gray-600">状态:</td><td class="${call.status_code < 400 ? 'text-green-600' : 'text-red-600'}">${call.status_code}</td></tr>
                        <tr><td class="font-medium text-gray-600">耗时:</td><td>${call.duration_ms}ms</td></tr>
                        ${call.model ? `<tr><td class="font-medium text-gray-600">模型:</td><td>${call.model}</td></tr>` : ''}
                    </table>
                </div>
                <div>
                    <h4 class="font-semibold text-gray-700 mb-2">请求头</h4>
                    <div class="json-viewer text-xs">${formatJson(call.headers)}</div>
                </div>
            </div>
            
            ${inputMessages.length > 0 || outputMessages.length > 0 ? `
                <div>
                    <h4 class="font-semibold text-gray-700 mb-2">对话内容</h4>
                    <div class="bg-white border rounded">
                <table class="min-w-full divide-y divide-gray-200">
                    <thead class="bg-gray-50">
                        <tr>
                            <th class="px-4 py-2 text-left text-sm font-medium text-gray-700">Role</th>
                            <th class="px-4 py-2 text-left text-sm font-medium text-gray-700">Content</th>
                        </tr>
                    </thead>
                    <tbody class="bg-white divide-y divide-gray-200">
                        ${inputMessages.map(msg => `
                            <tr>
                                <td class="px-4 py-2 text-sm font-medium text-gray-900">${escapeHtml(msg.role)}</td>
                                <td class="px-4 py-2 text-sm text-gray-700">
                                    <div class="max-w-xl" style="white-space: pre-line; max-height: 200px; overflow-y: auto;">
                                        ${escapeHtml(msg.content).replace(/\n\n/g, '\n').replace(/\n/g, ' ')}
                                    </div>
                                </td>
                            </tr>
                        `).join('')}
                        ${outputMessages.map(msg => `
                            <tr>
                                <td class="px-4 py-2 text-sm font-medium text-gray-900">${escapeHtml(msg.role)}</td>
                                <td class="px-4 py-2 text-sm text-gray-700">
                                    <div class="max-w-xl" style="white-space: pre-line; max-height: 200px; overflow-y: auto;">
                                        ${escapeHtml(msg.content).replace(/\n\n/g, '\n').replace(/\n/g, ' ')}
                                    </div>
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
                </div>
            ` : ''}
            
            ${call.request_body ? `
                <div>
                    <h4 class="font-semibold text-gray-700 mb-2">完整请求体</h4>
                    <div class="json-viewer text-xs">${formatJson(call.request_body)}</div>
                </div>
            ` : ''}
            
            ${call.response_body ? `
                <div>
                    <h4 class="font-semibold text-gray-700 mb-2">完整响应体</h4>
                    <div class="json-viewer text-xs">${formatJson(call.response_body)}</div>
                </div>
            ` : ''}
            
            ${call.error ? `
                <div>
                    <h4 class="font-semibold text-gray-700 mb-2">错误信息</h4>
                    <div class="bg-red-50 border border-red-200 rounded p-3 text-red-700">
                        ${call.error}
                    </div>
                </div>
            ` : ''}
        </div>
    `;
    
    modal.classList.remove('hidden');
}

// Close modal
function closeModal() {
    document.getElementById('detailModal').classList.add('hidden');
}

// Format JSON for display
function formatJson(obj) {
    if (obj === null || obj === undefined) return '';
    try {
        return JSON.stringify(obj, null, 2);
    } catch (e) {
        return String(obj);
    }
}

// Escape HTML to prevent XSS and handle newlines
function escapeHtml(text) {
    if (!text) return '';
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

// Get provider from model name
function getProviderFromModel(model) {
    if (!model) return 'unknown';
    const modelLower = model.toLowerCase();
    if (modelLower.includes('claude')) return 'claude';
    if (modelLower.includes('gpt') || modelLower.includes('o1') || modelLower.includes('o3')) return 'openai';
    if (modelLower.includes('azure')) return 'azure';
    return 'unknown';
}

// Auto-refresh functions with debouncing
let lastUpdateTime = 0;
let updateInProgress = false;

function startAutoRefresh() {
    if (autoRefreshInterval) return;
    
    document.getElementById('startBtn').classList.add('hidden');
    document.getElementById('stopBtn').classList.remove('hidden');
    
    autoRefreshInterval = setInterval(async () => {
        if (updateInProgress) return;
        
        const now = Date.now();
        if (now - lastUpdateTime >= 2000) {
            updateInProgress = true;
            try {
                await Promise.all([
                    loadAPICalls(),
                    loadStats()
                ]);
                lastUpdateTime = now;
            } finally {
                updateInProgress = false;
            }
        }
    }, 2000);
}

function stopAutoRefresh() {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
    }
    
    document.getElementById('startBtn').classList.remove('hidden');
    document.getElementById('stopBtn').classList.add('hidden');
}

function toggleAutoScroll() {
    // Auto scroll is handled in renderAPICalls
}

// Clear logs
async function clearLogs() {
    try {
        await fetch(`${API_BASE_URL}/web/api/calls`, { method: 'DELETE' });
        currentCalls = [];
        renderAPICalls();
    } catch (error) {
        console.error('Error clearing logs:', error);
    }
}

// Close modal when clicking outside
window.addEventListener('click', function(event) {
    const modal = document.getElementById('detailModal');
    if (event.target === modal) {
        closeModal();
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        closeModal();
    }
});
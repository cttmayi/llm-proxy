// Web interface JavaScript for LLM Proxy

const API_BASE_URL = window.location.origin;

// Global state
let currentChatModel = 'gpt-4o';
let currentEmbedModel = 'text-embedding-ada-002';

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    loadAvailableModels();
    
    // Set up event listeners
    document.getElementById('chatModel').addEventListener('change', function() {
        currentChatModel = this.value;
    });
    
    document.getElementById('embedModel').addEventListener('change', function() {
        currentEmbedModel = this.value;
    });
    
    // Set up keyboard shortcuts
    document.getElementById('chatMessage').addEventListener('keydown', handleChatKeydown);
    document.getElementById('embedText').addEventListener('keydown', handleEmbedKeydown);
});

// Handle keyboard shortcuts
function handleChatKeydown(event) {
    if (event.ctrlKey && event.key === 'Enter') {
        event.preventDefault();
        sendChatMessage();
    }
}

function handleEmbedKeydown(event) {
    if (event.ctrlKey && event.key === 'Enter') {
        event.preventDefault();
        createEmbeddings();
    }
}

// Load available models from API
async function loadAvailableModels() {
    try {
        const response = await fetch(`${API_BASE_URL}/v1/models`);
        const data = await response.json();
        
        if (response.ok && data.data) {
            displayModels(data.data);
            updateModelSelects(data.data);
        } else {
            throw new Error('Failed to load models');
        }
    } catch (error) {
        console.error('Error loading models:', error);
        displayModelsError('Failed to load available models. Please check the server.');
    }
}

// Display available models
function displayModels(models) {
    const modelsList = document.getElementById('modelsList');
    modelsList.innerHTML = '';
    
    if (models.length === 0) {
        modelsList.innerHTML = '<div class="text-center text-gray-500 py-4">No models available</div>';
        return;
    }
    
    models.forEach(model => {
        const modelCard = createModelCard(model);
        modelsList.appendChild(modelCard);
    });
}

// Create a model card element
function createModelCard(model) {
    const card = document.createElement('div');
    card.className = 'model-card bg-gray-50 p-4 rounded-lg border border-gray-200';
    
    const provider = getProviderFromModel(model.id);
    const providerColor = getProviderColor(provider);
    
    card.innerHTML = `
        <div class="flex justify-between items-start">
            <div>
                <h3 class="font-semibold text-gray-800">${model.id}</h3>
                <p class="text-sm text-gray-600">${model.object}</p>
            </div>
            <span class="inline-block px-2 py-1 text-xs font-semibold rounded-full ${providerColor}">
                ${provider}
            </span>
        </div>
    `;
    
    return card;
}

// Update model selects with available models
function updateModelSelects(models) {
    const chatSelect = document.getElementById('chatModel');
    const embedSelect = document.getElementById('embedModel');
    
    // Clear existing options
    chatSelect.innerHTML = '';
    embedSelect.innerHTML = '';
    
    // Add chat models
    const chatModels = models.filter(m => 
        m.id.includes('gpt') || m.id.includes('claude') || m.id.includes('o1')
    );
    
    chatModels.forEach(model => {
        const option = new Option(model.id, model.id);
        chatSelect.add(option);
    });
    
    // Add embedding models
    const embedModels = models.filter(m => 
        m.id.includes('embedding')
    );
    
    embedModels.forEach(model => {
        const option = new Option(model.id, model.id);
        embedSelect.add(option);
    });
}

// Send chat message
async function sendChatMessage() {
    const message = document.getElementById('chatMessage').value.trim();
    const model = document.getElementById('chatModel').value;
    const enableStreaming = document.getElementById('chatStream').checked;
    
    if (!message) {
        showChatError('Please enter a message');
        return;
    }
    
    showChatLoading();
    
    try {
        const startTime = Date.now();
        
        if (enableStreaming) {
            await sendStreamingChat(message, model, startTime);
        } else {
            await sendRegularChat(message, model, startTime);
        }
    } catch (error) {
        console.error('Error sending chat message:', error);
        showChatError(`Error: ${error.message}`);
    }
}

// Send regular chat message
async function sendRegularChat(message, model, startTime) {
    const response = await fetch(`${API_BASE_URL}/v1/chat/completions`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            model: model,
            messages: [{ role: 'user', content: message }]
        })
    });
    
    const data = await response.json();
    const endTime = Date.now();
    
    if (!response.ok) {
        throw new Error(data.detail || data.error || 'Failed to get response');
    }
    
    displayChatResponse(data.choices[0].message.content, data.usage, endTime - startTime);
}

// Send streaming chat message
async function sendStreamingChat(message, model, startTime) {
    const response = await fetch(`${API_BASE_URL}/v1/chat/completions`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            model: model,
            messages: [{ role: 'user', content: message }],
            stream: true
        })
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || error.error || 'Failed to get response');
    }
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let content = '';
    
    displayChatStreaming(content);
    
    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (data === '[DONE]') continue;
                    
                    try {
                        const parsed = JSON.parse(data);
                        const delta = parsed.choices[0]?.delta?.content || '';
                        content += delta;
                        updateChatStreaming(content);
                    } catch (e) {
                        console.warn('Failed to parse streaming chunk:', e);
                    }
                }
            }
        }
        
        const endTime = Date.now();
        finalizeChatStreaming(content, endTime - startTime);
    } catch (error) {
        console.error('Streaming error:', error);
        showChatError(`Streaming error: ${error.message}`);
    }
}

// Create embeddings
async function createEmbeddings() {
    const text = document.getElementById('embedText').value.trim();
    const model = document.getElementById('embedModel').value;
    
    if (!text) {
        showEmbedError('Please enter text to embed');
        return;
    }
    
    showEmbedLoading();
    
    try {
        const startTime = Date.now();
        
        const response = await fetch(`${API_BASE_URL}/v1/embeddings`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: model,
                input: text
            })
        });
        
        const data = await response.json();
        const endTime = Date.now();
        
        if (!response.ok) {
            throw new Error(data.detail || data.error || 'Failed to create embeddings');
        }
        
        displayEmbedResponse(data, endTime - startTime);
    } catch (error) {
        console.error('Error creating embeddings:', error);
        showEmbedError(`Error: ${error.message}`);
    }
}

// Display chat response
function displayChatResponse(content, usage, responseTime) {
    const responseDiv = document.getElementById('chatResponse');
    const outputDiv = document.getElementById('chatOutput');
    
    responseDiv.classList.remove('hidden');
    responseDiv.classList.add('streaming-content');
    
    let html = `<div>${escapeHtml(content)}</div>`;
    
    if (usage) {
        html += `
            <div class="token-info">
                Tokens: ${usage.prompt_tokens} prompt + ${usage.completion_tokens || 0} completion = ${usage.total_tokens} total
            </div>
        `;
    }
    
    html += `<div class="response-time">Response time: ${responseTime}ms</div>`;
    
    outputDiv.innerHTML = html;
    hideChatLoading();
}

// Display streaming chat response
function displayChatStreaming(content) {
    const responseDiv = document.getElementById('chatResponse');
    const outputDiv = document.getElementById('chatOutput');
    
    responseDiv.classList.remove('hidden');
    responseDiv.classList.add('streaming-content');
    
    outputDiv.innerHTML = `<div>${escapeHtml(content)}<span class="animate-pulse">▊</span></div>`;
}

// Update streaming chat response
function updateChatStreaming(content) {
    const outputDiv = document.getElementById('chatOutput');
    outputDiv.innerHTML = `<div>${escapeHtml(content)}<span class="animate-pulse">▊</span></div>`;
}

// Finalize streaming chat response
function finalizeChatStreaming(content, responseTime) {
    const outputDiv = document.getElementById('chatOutput');
    outputDiv.innerHTML = `<div>${escapeHtml(content)}</div><div class="response-time">Response time: ${responseTime}ms</div>`;
}

// Display embeddings response
function displayEmbedResponse(data, responseTime) {
    const responseDiv = document.getElementById('embedResponse');
    const outputDiv = document.getElementById('embedOutput');
    
    responseDiv.classList.remove('hidden');
    
    const embeddings = data.data;
    let html = '';
    
    if (embeddings.length === 1) {
        const embedding = embeddings[0];
        html = `
            <div class="code-block">
                [${embedding.embedding.slice(0, 10).join(', ')}, ...]
            </div>
            <div class="token-info">
                ${embedding.embedding.length} dimensions, ${data.usage.total_tokens} tokens
            </div>
        `;
    } else {
        html = `
            <div>Generated ${embeddings.length} embeddings:</div>
            ${embeddings.map((emb, index) => `
                <div class="mt-2">
                    <strong>Item ${index + 1}:</strong>
                    <div class="code-block">
                        [${emb.embedding.slice(0, 5).join(', ')}, ...]
                    </div>
                </div>
            `).join('')}
            <div class="token-info">
                ${embeddings[0]?.embedding?.length || 0} dimensions each, ${data.usage.total_tokens} tokens total
            </div>
        `;
    }
    
    html += `<div class="response-time">Response time: ${responseTime}ms</div>`;
    
    outputDiv.innerHTML = html;
    hideEmbedLoading();
}

// Loading states
function showChatLoading() {
    const button = document.querySelector('#chatResponse').parentElement.querySelector('button');
    button.innerHTML = '<div class="loading"></div> Sending...';
    button.disabled = true;
}

function hideChatLoading() {
    const button = document.querySelector('#chatResponse').parentElement.querySelector('button');
    button.innerHTML = 'Send Message';
    button.disabled = false;
}

function showEmbedLoading() {
    const button = document.querySelector('#embedResponse').parentElement.querySelector('button');
    button.innerHTML = '<div class="loading"></div> Creating...';
    button.disabled = true;
}

function hideEmbedLoading() {
    const button = document.querySelector('#embedResponse').parentElement.querySelector('button');
    button.innerHTML = 'Create Embeddings';
    button.disabled = false;
}

// Error states
function showChatError(message) {
    const responseDiv = document.getElementById('chatResponse');
    const outputDiv = document.getElementById('chatOutput');
    
    responseDiv.classList.remove('hidden');
    responseDiv.classList.add('error-message');
    responseDiv.classList.remove('streaming-content');
    
    outputDiv.innerHTML = `<div>${escapeHtml(message)}</div>`;
    hideChatLoading();
}

function showEmbedError(message) {
    const responseDiv = document.getElementById('embedResponse');
    const outputDiv = document.getElementById('embedOutput');
    
    responseDiv.classList.remove('hidden');
    responseDiv.classList.add('error-message');
    
    outputDiv.innerHTML = `<div>${escapeHtml(message)}</div>`;
    hideEmbedLoading();
}

function displayModelsError(message) {
    const modelsList = document.getElementById('modelsList');
    modelsList.innerHTML = `<div class="text-center text-red-500 py-4">${escapeHtml(message)}</div>`;
}

// Utility functions
function getProviderFromModel(modelName) {
    if (modelName.includes('claude')) return 'Claude';
    if (modelName.includes('gpt') || modelName.includes('o1')) return 'OpenAI';
    if (modelName.includes('azure')) return 'Azure';
    return 'Unknown';
}

function getProviderColor(provider) {
    const colors = {
        'Claude': 'bg-purple-100 text-purple-800',
        'OpenAI': 'bg-green-100 text-green-800',
        'Azure': 'bg-blue-100 text-blue-800',
        'Unknown': 'bg-gray-100 text-gray-800'
    };
    return colors[provider] || colors['Unknown'];
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Clear responses
function clearChatResponse() {
    document.getElementById('chatResponse').classList.add('hidden');
    document.getElementById('chatOutput').innerHTML = '';
}

function clearEmbedResponse() {
    document.getElementById('embedResponse').classList.add('hidden');
    document.getElementById('embedOutput').innerHTML = '';
}

// Auto-refresh models every 30 seconds
setInterval(loadAvailableModels, 30000);
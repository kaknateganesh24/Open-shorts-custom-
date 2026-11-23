import React, { useState, useEffect } from 'react';
import { Key, Eye, EyeOff, Check, Cpu, Sparkles, RefreshCw, AlertCircle } from 'lucide-react';
import { getApiUrl } from '../config';

export default function KeyInput({ onKeySet, savedKey, onAiConfigChange }) {
    const [provider, setProvider] = useState(() => localStorage.getItem('ai_provider') || 'ollama');
    const [key, setKey] = useState(savedKey || '');
    const [isVisible, setIsVisible] = useState(false);
    const [ollamaBaseUrl, setOllamaBaseUrl] = useState(() => localStorage.getItem('ollama_base_url') || 'http://host.docker.internal:11434/v1');
    const [ollamaModel, setOllamaModel] = useState(() => localStorage.getItem('ollama_model') || 'llama3:8b');
    const [availableModels, setAvailableModels] = useState(['llama3:8b', 'mistral:7b', 'gemma:7b']);
    const [ollamaDetected, setOllamaDetected] = useState(false);

    const [testLoading, setTestLoading] = useState(false);
    const [testResult, setTestResult] = useState(null); // { valid: bool, message: string }

    useEffect(() => {
        if (savedKey) setKey(savedKey);
    }, [savedKey]);

    useEffect(() => {
        // Query /api/config to discover detected Ollama models
        fetch(getApiUrl('/api/config'))
            .then(res => res.json())
            .then(data => {
                if (data.ollamaDetected) {
                    setOllamaDetected(true);
                    if (data.ollamaModels && data.ollamaModels.length > 0) {
                        setAvailableModels(data.ollamaModels);
                        if (!localStorage.getItem('ollama_model')) {
                            const preferred = data.ollamaModels.find(m => m.includes('llama3')) || data.ollamaModels[0];
                            setOllamaModel(preferred);
                            localStorage.setItem('ollama_model', preferred);
                        }
                    }
                }
            })
            .catch(() => {});
    }, []);

    const notifyParent = (newProv, newKey, newUrl, newModel) => {
        if (onAiConfigChange) {
            onAiConfigChange({
                provider: newProv,
                apiKey: newKey,
                baseUrl: newUrl,
                model: newModel
            });
        }
    };

    const handleProviderChange = (newProv) => {
        setProvider(newProv);
        localStorage.setItem('ai_provider', newProv);
        setTestResult(null);
        notifyParent(newProv, key, ollamaBaseUrl, ollamaModel);
    };

    const handleSaveGeminiKey = (newKey) => {
        setKey(newKey);
        localStorage.setItem('gemini_key', newKey);
        if (onKeySet) onKeySet(newKey);
        notifyParent(provider, newKey, ollamaBaseUrl, ollamaModel);
    };

    const handleModelChange = (newModel) => {
        setOllamaModel(newModel);
        localStorage.setItem('ollama_model', newModel);
        notifyParent(provider, key, ollamaBaseUrl, newModel);
    };

    const handleTestConnection = async () => {
        setTestLoading(true);
        setTestResult(null);
        try {
            const body = provider === 'ollama' 
                ? { provider: 'ollama', base_url: ollamaBaseUrl, model: ollamaModel }
                : { provider: 'gemini', api_key: key };

            const res = await fetch(getApiUrl('/api/ai/validate'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            const data = await res.json();
            if (data.valid) {
                setTestResult({ valid: true, message: data.message || 'Connection verified successfully!' });
            } else {
                setTestResult({ valid: false, message: data.error || 'Validation failed.' });
            }
        } catch (e) {
            setTestResult({ valid: false, message: `Could not reach server: ${e.message}` });
        } finally {
            setTestLoading(false);
        }
    };

    return (
        <div className="card p-4 sm:p-6 mb-8 animate-fade">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-paper3 rounded-input text-brass">
                        {provider === 'ollama' ? <Cpu size={18} /> : <Key size={18} />}
                    </div>
                    <div>
                        <h2 className="font-display lowercase text-lg text-ink">AI Moment Detection Provider</h2>
                        <p className="text-xs text-muted">Select how OpenShorts detects viral moments from video transcripts</p>
                    </div>
                </div>

                {/* Provider switcher tabs */}
                <div className="flex bg-paper3 p-1 rounded-input border border-rule gap-1">
                    <button
                        type="button"
                        onClick={() => handleProviderChange('ollama')}
                        className={`px-3 py-1 text-xs rounded font-medium transition-colors flex items-center gap-1.5 ${
                            provider === 'ollama' ? 'bg-paper text-ink shadow-sm' : 'text-muted hover:text-ink'
                        }`}
                    >
                        <Cpu size={13} />
                        Local AI (Ollama)
                        {ollamaDetected && <span className="w-2 h-2 rounded-full bg-ok" title="Ollama running locally"></span>}
                    </button>
                    <button
                        type="button"
                        onClick={() => handleProviderChange('gemini')}
                        className={`px-3 py-1 text-xs rounded font-medium transition-colors flex items-center gap-1.5 ${
                            provider === 'gemini' ? 'bg-paper text-ink shadow-sm' : 'text-muted hover:text-ink'
                        }`}
                    >
                        <Sparkles size={13} />
                        Google Gemini
                    </button>
                </div>
            </div>

            {provider === 'ollama' ? (
                <div className="space-y-4">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                        <div>
                            <label className="block text-xs font-medium text-muted mb-1">Ollama Model</label>
                            <select
                                value={ollamaModel}
                                onChange={(e) => handleModelChange(e.target.value)}
                                className="input-field w-full font-mono text-sm"
                            >
                                {availableModels.map(m => (
                                    <option key={m} value={m}>{m} (Installed)</option>
                                ))}
                            </select>
                        </div>
                        <div>
                            <label className="block text-xs font-medium text-muted mb-1">Ollama Endpoint URL</label>
                            <input
                                type="text"
                                value={ollamaBaseUrl}
                                onChange={(e) => {
                                    setOllamaBaseUrl(e.target.value);
                                    localStorage.setItem('ollama_base_url', e.target.value);
                                    notifyParent(provider, key, e.target.value, ollamaModel);
                                }}
                                placeholder="http://host.docker.internal:11434/v1"
                                className="input-field w-full font-mono text-sm"
                            />
                        </div>
                    </div>

                    <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
                        <button
                            type="button"
                            onClick={handleTestConnection}
                            disabled={testLoading}
                            className="btn-quiet px-3 py-1.5 text-xs flex items-center gap-1.5"
                        >
                            <RefreshCw size={12} className={testLoading ? 'animate-spin' : ''} />
                            {testLoading ? 'Testing...' : 'Test Ollama Connection'}
                        </button>
                        <span className="text-xs text-muted">
                            {ollamaDetected ? '✓ Ollama service detected on host machine.' : 'Ensure Ollama is running on your machine.'}
                        </span>
                    </div>
                </div>
            ) : (
                <div className="space-y-4">
                    <div>
                        <label className="block text-xs font-medium text-muted mb-1">Gemini API Key</label>
                        <div className="flex flex-col sm:flex-row gap-2">
                            <div className="relative flex-1">
                                <input
                                    type={isVisible ? "text" : "password"}
                                    value={key}
                                    onChange={(e) => handleSaveGeminiKey(e.target.value)}
                                    placeholder="AIzaSy..."
                                    className="input-field w-full pr-10 font-mono text-sm"
                                />
                                <button
                                    type="button"
                                    onClick={() => setIsVisible(!isVisible)}
                                    className="absolute right-3 top-1/2 -translate-y-1/2 text-muted hover:text-ink transition-colors"
                                >
                                    {isVisible ? <EyeOff size={16} /> : <Eye size={16} />}
                                </button>
                            </div>
                            <button
                                type="button"
                                onClick={handleTestConnection}
                                disabled={!key || testLoading}
                                className="btn-quiet px-3 py-1.5 text-xs flex items-center gap-1.5"
                            >
                                <RefreshCw size={12} className={testLoading ? 'animate-spin' : ''} />
                                {testLoading ? 'Verifying...' : 'Verify Key & Free Tier'}
                            </button>
                        </div>
                    </div>
                    <p className="text-xs text-muted">
                        Note: Google restricts the Gemini free tier on some Google Cloud projects. If you get a 403 error, switch to <strong>Local AI (Ollama)</strong> above for 100% free unlimited processing.
                        <br />
                        <a
                            href="https://aistudio.google.com/app/apikey"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-brass hover:underline mt-1 inline-block"
                        >
                            Get your Gemini API Key here →
                        </a>
                    </p>
                </div>
            )}

            {testResult && (
                <div className={`mt-3 p-3 rounded-input text-xs flex items-start gap-2 ${
                    testResult.valid ? 'bg-ok/10 text-ok border border-ok/20' : 'bg-warn/10 text-warn border border-warn/20'
                }`}>
                    {testResult.valid ? <Check size={14} className="shrink-0 mt-0.5" /> : <AlertCircle size={14} className="shrink-0 mt-0.5" />}
                    <span>{testResult.message}</span>
                </div>
            )}
        </div>
    );
}

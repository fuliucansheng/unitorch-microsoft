import { useState, useRef, useEffect, type KeyboardEvent, type ChangeEvent } from 'react';
import { useStore } from '../../store/useStore';
import { api } from '../../lib/api';
import { Send, Command, Database, Paperclip, X, FileIcon, TerminalSquare, Tag, FileText, Brain, Wrench, ChevronDown, Check } from 'lucide-react';
import { cn } from '../../lib/utils';

export function ChatInput() {
  const [input, setInput] = useState('');
  const [showCommands, setShowCommands] = useState(false);
  const [showEntities, setShowEntities] = useState(false);
  const [showModels, setShowModels] = useState(false);
  const [history, setHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [files, setFiles] = useState<File[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  
  const [mentionQuery, setMentionQuery] = useState('');
  const [commandQuery, setCommandQuery] = useState('');
  
  // API dynamically loaded states
  const [commands, setCommands] = useState<any[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  
  const { 
    addMessage, entities, setTyping,
    availableModels, selectedModel, agentMode, setModel, setAgentMode, activeSessionId,
    initData
  } = useStore();

  useEffect(() => {
    // Load commands dynamically
    api.chat.getCommands().then(setCommands).catch(console.error);
  }, []);

  const allEntities = entities.map((e: any) => {
    let icon = Database;
    let color = 'text-blue-500';
    if (e.type === 'job') { icon = TerminalSquare; color = 'text-orange-500'; }
    if (e.type === 'label') { icon = Tag; color = 'text-purple-500'; }
    if (e.type === 'report') { icon = FileText; color = 'text-green-500'; }
    return { ...e, icon, color };
  });

  const filteredEntities = allEntities.filter(e => 
    e.name.toLowerCase().includes(mentionQuery) || 
    e.id.toLowerCase().includes(mentionQuery)
  );

  const filteredCommands = commands.filter(c => 
    c.name.toLowerCase().includes(commandQuery)
  );

  const handleSend = async () => {
    if (!input.trim() && files.length === 0) return;

    const userMessage = input.trim();
    
    // Create UI attachments
    const storeAttachments = files.map(f => ({
      id: Math.random().toString(36).substring(2, 9),
      name: f.name,
      size: f.size,
      type: f.type
    }));

    // Extract entities from mentions (e.g., @data1)
    const entityMatches = userMessage.match(/@([a-zA-Z0-9_-]+)/g) || [];
    const requestEntities = entityMatches.map(match => {
      const id = match.slice(1);
      const entity = allEntities.find(e => e.id === id);
      return entity ? { type: entity.type, id: entity.id } : null;
    }).filter(Boolean);

    // Show the user's message immediately
    addMessage({ role: 'user', content: userMessage, attachments: storeAttachments });
    
    if (userMessage) {
      setHistory(prev => [...prev, userMessage]);
    }
    setHistoryIndex(-1);
    setInput('');
    setShowCommands(false);
    setShowEntities(false);
    setTyping(true);
    
    const filesToSend = [...files];
    setFiles([]);
    
    try {
      let appendedContent = '';
      
      // Upload files sequentially or in parallel, then append paths to the prompt
      if (filesToSend.length > 0) {
        const uploadPromises = filesToSend.map(f => api.utils.upload(f));
        const uploadedResults = await Promise.all(uploadPromises);
        
        appendedContent = '\n\n[System Info: The user has uploaded the following files. Please use the paths below to access them:]\n';
        uploadedResults.forEach(res => {
          appendedContent += `- ${res.filename} (Server Path: ${res.path})\n`;
        });
      }

      // Final api prompt includes the user's original message plus the appended file paths context
      const finalApiMessageContent = userMessage + appendedContent;

      // Find the selected model details
      const currentModelObj = availableModels.find(m => m.id === selectedModel);

      const response = await fetch(api.chat.getCompletionsUrl(), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({
          session_id: activeSessionId,
          message: {
            role: 'user',
            content: finalApiMessageContent
          },
          mode: agentMode,
          model: selectedModel,
          model_id: currentModelObj?.model_id,
          provider_id: currentModelObj?.provider_id,
          entities: requestEntities,
          stream: false // Disabled streaming
        })
      });

      if (!response.ok) throw new Error('Network response was not ok');

      const data = await response.json();
      
      // Extract content from response (handling various common API response structures)
      let finalContent = '';
      if (typeof data === 'string') {
        finalContent = data;
      } else if (data.content) {
        finalContent = data.content;
      } else if (data.message?.content) {
        finalContent = data.message.content;
      } else if (data.choices?.[0]?.message?.content) {
        finalContent = data.choices[0].message.content;
      } else {
        finalContent = JSON.stringify(data, null, 2);
      }

      addMessage({ role: 'assistant', content: finalContent });
      
      // Refresh background data (datasets, jobs, labels, reports) after AI response
      initData().catch(console.error);

    } catch (error) {
      console.error('Error in completions:', error);
      addMessage({ role: 'assistant', content: 'Sorry, an error occurred while processing your request.' });
    } finally {
      setTyping(false);
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Handle dropdown navigation
    if (showCommands && filteredCommands.length > 0) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex(prev => Math.min(prev + 1, filteredCommands.length - 1));
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex(prev => Math.max(prev - 1, 0));
        return;
      }
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        insertText(`/${filteredCommands[selectedIndex].name}`);
        return;
      }
      if (e.key === 'Escape') {
        setShowCommands(false);
        return;
      }
    }

    if (showEntities && filteredEntities.length > 0) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex(prev => Math.min(prev + 1, filteredEntities.length - 1));
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex(prev => Math.max(prev - 1, 0));
        return;
      }
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        insertText(`@${filteredEntities[selectedIndex].id}`);
        return;
      }
      if (e.key === 'Escape') {
        setShowEntities(false);
        return;
      }
    }

    // 按下 Shift+Enter 时发送消息
    if (e.key === 'Enter' && e.shiftKey) {
      e.preventDefault();
      handleSend();
      return;
    }
    
    if (input === '') {
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        if (history.length > 0) {
          const nextIndex = historyIndex < history.length - 1 ? historyIndex + 1 : historyIndex;
          setHistoryIndex(nextIndex);
          setInput(history[history.length - 1 - nextIndex]);
        }
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        if (historyIndex > 0) {
          const nextIndex = historyIndex - 1;
          setHistoryIndex(nextIndex);
          setInput(history[history.length - 1 - nextIndex]);
        } else if (historyIndex === 0) {
          setHistoryIndex(-1);
          setInput('');
        }
      }
    }
  };

  const handleChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    const val = e.target.value;
    setInput(val);
    
    const cursor = e.target.selectionStart;
    const beforeCursor = val.slice(0, cursor);
    
    // Match the last '@' or '/' and everything after it until a space or end of string
    const match = beforeCursor.match(/([@/])([^@/\s]*)$/);
    
    if (match) {
      const trigger = match[1];
      const query = match[2].toLowerCase();
      
      if (trigger === '@') {
        setShowEntities(true);
        setShowCommands(false);
        setMentionQuery(query);
      } else if (trigger === '/') {
        setShowCommands(true);
        setShowEntities(false);
        setCommandQuery(query);
      }
      setSelectedIndex(0);
    } else {
      setShowEntities(false);
      setShowCommands(false);
      setMentionQuery('');
      setCommandQuery('');
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const newFiles = Array.from(e.target.files);
      setFiles(prev => [...prev, ...newFiles]);
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const insertText = (text: string) => {
    const cursor = textareaRef.current?.selectionStart || input.length;
    const beforeCursor = input.slice(0, cursor);
    const afterCursor = input.slice(cursor);
    
    const match = beforeCursor.match(/([@/])([^@/\s]*)$/);
    let newValue = input;
    
    if (match) {
      const triggerIndex = beforeCursor.length - match[0].length;
      newValue = beforeCursor.slice(0, triggerIndex) + text + ' ' + afterCursor;
      setInput(newValue);
    }
    
    setShowCommands(false);
    setShowEntities(false);
    
    // Refocus the textarea and move cursor to the very end
    setTimeout(() => {
      if (textareaRef.current) {
        textareaRef.current.focus();
        const endPos = newValue.length;
        textareaRef.current.setSelectionRange(endPos, endPos);
      }
    }, 0);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    else return (bytes / 1048576).toFixed(1) + ' MB';
  };

  return (
    <div className="p-4 bg-background/80 backdrop-blur-md border-t border-border flex flex-col items-center">
      <div className="w-full max-w-3xl relative flex flex-col gap-2">
        
        {/* Controls Row: Model Selector & Agent Mode */}
        <div className="w-full flex items-center justify-between mb-1 px-1">
          {/* Left: Model Selector */}
          <div className="relative">
            <button 
              type="button"
              onClick={() => setShowModels(!showModels)}
              className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium bg-secondary/30 hover:bg-secondary rounded-lg border border-border text-muted-foreground hover:text-foreground transition-colors"
            >
              {availableModels.find(m => m.id === selectedModel)?.name || 'Select Model'} <ChevronDown size={12} />
            </button>
            
            {showModels && (
              <>
                <div 
                  className="fixed inset-0 z-10" 
                  onClick={() => setShowModels(false)}
                />
                <div className="absolute bottom-full left-0 mb-2 w-48 bg-card border border-border rounded-xl shadow-xl overflow-hidden z-20 animate-in fade-in slide-in-from-bottom-1">
                  <div className="p-1 max-h-48 overflow-y-auto">
                    {availableModels.map((m, idx) => (
                      <button
                        type="button"
                        key={`${m.id}-${idx}`}
                        onClick={() => { setModel(m.id); setShowModels(false); }}
                        className="w-full text-left px-3 py-2 text-xs hover:bg-secondary rounded-lg flex items-center justify-between transition-colors"
                      >
                        {m.name}
                        {selectedModel === m.id && <Check size={12} className="text-primary" />}
                      </button>
                    ))}
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Right: Mode Toggle */}
          <div className="flex items-center bg-secondary/30 p-0.5 rounded-lg border border-border">
            <button
              type="button"
              onClick={() => setAgentMode('plan')}
              className={cn(
                "flex items-center gap-1.5 px-3 py-1 text-xs font-medium rounded-md transition-all",
                agentMode === 'plan' 
                  ? "bg-background shadow-sm text-foreground" 
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              <Brain size={12} /> Plan
            </button>
            <button
              type="button"
              onClick={() => setAgentMode('build')}
              className={cn(
                "flex items-center gap-1.5 px-3 py-1 text-xs font-medium rounded-md transition-all",
                agentMode === 'build' 
                  ? "bg-background shadow-sm text-foreground" 
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              <Wrench size={12} /> Build
            </button>
          </div>
        </div>

        {/* Autocomplete menus */}
        {showCommands && (
          <div className="absolute bottom-full left-0 mb-2 w-64 bg-card border border-border rounded-xl shadow-xl overflow-hidden z-10 animate-in fade-in slide-in-from-bottom-2">
            <div className="px-3 py-2 text-xs font-medium text-muted-foreground bg-secondary/50 flex justify-between">
              <span>Commands</span>
              <span className="text-[10px]">Enter to select</span>
            </div>
            <div className="p-1 max-h-48 overflow-y-auto">
              {filteredCommands.map((cmd, idx) => (
                <button
                  type="button"
                  key={`${cmd.name}-${idx}`}
                  onClick={() => insertText(`/${cmd.name}`)}
                  onMouseEnter={() => setSelectedIndex(idx)}
                  className={cn(
                    "w-full text-left px-3 py-2 text-sm rounded-lg flex flex-col gap-1 transition-colors",
                    selectedIndex === idx ? "bg-secondary" : "hover:bg-secondary/50"
                  )}
                >
                  <span className="flex items-center gap-2"><Command size={14} className="text-primary" /> /{cmd.name}</span>
                  {cmd.description && <span className="text-[10px] text-muted-foreground ml-5">{cmd.description}</span>}
                </button>
              ))}
            </div>
          </div>
        )}

        {showEntities && (
          <div className="absolute bottom-full left-0 mb-2 w-64 bg-card border border-border rounded-xl shadow-xl overflow-hidden z-10 animate-in fade-in slide-in-from-bottom-2">
            <div className="px-3 py-2 text-xs font-medium text-muted-foreground bg-secondary/50 flex justify-between">
              <span>Mentions</span>
              {mentionQuery && <span className="text-[10px]">Searching "{mentionQuery}"</span>}
            </div>
            <div className="p-1 max-h-48 overflow-y-auto">
              {filteredEntities.length > 0 ? (
                filteredEntities.map((entity, idx) => (
                  <button
                    type="button"
                    key={`${entity.type}-${entity.id}-${idx}`}
                    onClick={() => insertText(`@${entity.id}`)}
                    onMouseEnter={() => setSelectedIndex(idx)}
                    className={cn(
                      "w-full text-left px-3 py-2 text-sm rounded-lg flex items-center gap-2 transition-colors",
                      selectedIndex === idx ? "bg-secondary" : "hover:bg-secondary/50"
                    )}
                  >
                    <entity.icon size={14} className={entity.color} shrink-0 /> 
                    <span className="truncate">{entity.name}</span>
                  </button>
                ))
              ) : (
                <div className="px-3 py-4 text-center text-sm text-muted-foreground">
                  No matches found
                </div>
              )}
            </div>
          </div>
        )}

        {/* File Previews */}
        {files.length > 0 && (
          <div className="flex flex-wrap gap-2 px-1 mb-1">
            {files.map((file, idx) => (
               <div key={idx} className="flex items-center gap-2 bg-card border border-border shadow-sm rounded-lg px-3 py-2 text-xs group relative pr-8 animate-in zoom-in-95">
                <div className="p-1.5 bg-blue-500/10 rounded-md text-blue-500">
                  <FileIcon size={14} />
                </div>
                <div className="flex flex-col">
                  <span className="truncate max-w-[150px] font-medium">{file.name}</span>
                  <span className="text-muted-foreground text-[10px]">{formatFileSize(file.size)}</span>
                </div>
                <button 
                  type="button"
                  onClick={() => removeFile(idx)}
                  className="absolute right-2 p-1 rounded-full hover:bg-destructive hover:text-destructive-foreground text-muted-foreground transition-colors"
                >
                  <X size={12} />
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Input Box */}
        <div className="relative flex items-end gap-2 bg-card border border-border rounded-2xl p-2 shadow-sm focus-within:ring-1 focus-within:ring-primary focus-within:border-primary transition-all">
          <input 
            type="file" 
            multiple 
            ref={fileInputRef} 
            style={{ display: 'none' }}
            onChange={handleFileChange} 
          />
          
          <button 
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="p-2 text-muted-foreground hover:text-foreground hover:bg-secondary rounded-xl transition-colors shrink-0 mb-0.5"
            title="Attach files"
          >
            <Paperclip size={20} />
          </button>
          
          <textarea
            ref={textareaRef}
            rows={1}
            value={input}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            placeholder="Ask Ads Studio anything... (Shift+Enter to send)"
            className="w-full bg-transparent border-none py-2.5 px-1 text-sm focus:outline-none resize-none max-h-[200px] min-h-[44px]"
            style={{ fieldSizing: 'content' } as React.CSSProperties}
          />
          
          <button 
            type="button"
            onClick={handleSend}
            disabled={!input.trim() && files.length === 0}
            title="Send message"
            className={`p-2 rounded-xl transition-all shrink-0 mb-0.5 flex items-center justify-center ${
              (input.trim() || files.length > 0) 
                ? 'bg-primary text-primary-foreground shadow-md hover:bg-primary/90 hover:scale-105' 
                : 'bg-secondary text-muted-foreground opacity-50'
            }`}
          >
            <Send size={18} className={(input.trim() || files.length > 0) ? 'ml-0.5' : ''} />
          </button>
        </div>
        <div className="text-center mt-1 text-[10px] text-muted-foreground">
          Ads Studio can make mistakes. Check important info.
        </div>
      </div>
    </div>
  );
}
import { useState, useEffect, useCallback, useRef } from 'react';
import { useStore } from '../../store/useStore';
import { Tag, Search, Crosshair, MousePointer2, ZoomIn, BarChart2, Users, CheckCircle2, Share2, Check, Table, Trash2, Loader2, Info, Settings2, Save } from 'lucide-react';
import { api } from '../../lib/api';
import { cn } from '../../lib/utils';

export function LabelView() {
  const { labelTasks, selectedLabelId, setView, initData, userProfile } = useStore();
  const [search, setSearch] = useState('');
  
  const [activeTab, setActiveTab] = useState<'labeling' | 'stats' | 'settings'>('labeling');
  const [copied, setCopied] = useState(false);
  
  const [taskDetails, setTaskDetails] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  // Labeling states
  const [currentSample, setCurrentSample] = useState<any>(null);
  const [loadingSample, setLoadingSample] = useState(false);
  const [formValues, setFormValues] = useState<Record<string, any>>({});
  const [submitting, setSubmitting] = useState(false);
  const [autoSubmit, setAutoSubmit] = useState(false);

  // Stats / Export states
  const [exportedData, setExportedData] = useState<any[]>([]);
  const [loadingExport, setLoadingExport] = useState(false);

  // BBox Interactive States
  const [bbox, setBbox] = useState({ x: 0, y: 0, w: 0, h: 0 }); // Hidden by default
  const [activeTool, setActiveTool] = useState<'crosshair' | 'pointer' | 'zoom'>('pointer');
  const [zoom, setZoom] = useState(1);
  const containerRef = useRef<HTMLDivElement>(null);
  
  const [dragInfo, setDragInfo] = useState<{
    type: 'move' | 'resize' | 'draw';
    startX: number;
    startY: number;
    initBox: typeof bbox;
    handle?: string;
  } | null>(null);

  // Settings State
  const [settingsForm, setSettingsForm] = useState({
    name: '',
    description: '',
    ui_html: '',
    display_fields_json: '',
    label_fields_json: '',
    extra_json: ''
  });
  const [savingSettings, setSavingSettings] = useState(false);
  const [settingsError, setSettingsError] = useState('');

  const fetchNextSample = useCallback(async () => {
    if (!selectedLabelId) return;
    setLoadingSample(true);
    setCurrentSample(null);
    try {
      const sample = await api.labels.getRandomSample(selectedLabelId);
      setCurrentSample(sample);
      
      // Reset form values
      if (taskDetails) {
        const initialForm: Record<string, any> = {};
        taskDetails.label_fields?.forEach((f: any) => {
          initialForm[f.key] = f.default ?? (f.type === 'multiselect' ? [] : (f.type === 'bool' ? null : ''));
        });
        setFormValues(initialForm);
      }
    } catch (err) {
      console.error('Failed to load next sample:', err);
    } finally {
      setLoadingSample(false);
      setBbox({ x: 0, y: 0, w: 0, h: 0 }); // Clear bbox on next sample
      setZoom(1);
    }
  }, [selectedLabelId, taskDetails]);

  useEffect(() => {
    if (!selectedLabelId) return;
    let active = true;
    setLoading(true);
    
    api.labels.getDetails(selectedLabelId)
      .then(data => {
        if (!active) return;
        setTaskDetails(data);
        
        // Initialize settings form
        setSettingsForm({
          name: data.name || '',
          description: data.description || '',
          ui_html: data.ui_html || '',
          display_fields_json: data.display_fields ? JSON.stringify(data.display_fields, null, 2) : '[]',
          label_fields_json: data.label_fields ? JSON.stringify(data.label_fields, null, 2) : '[]',
          extra_json: data.extra ? JSON.stringify(data.extra, null, 2) : '{}'
        });

        // Initialize default form values if we don't have a sample yet
        const initialForm: Record<string, any> = {};
        data.label_fields?.forEach((f: any) => {
          initialForm[f.key] = f.default ?? (f.type === 'multiselect' ? [] : (f.type === 'bool' ? null : ''));
        });
        setFormValues(initialForm);
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to load label details', err);
        if (active) setLoading(false);
      });

    return () => { active = false; };
  }, [selectedLabelId]);

  // Fetch sample when active tab becomes labeling
  useEffect(() => {
    if (activeTab === 'labeling' && taskDetails && !currentSample && !loadingSample) {
      fetchNextSample();
    }
  }, [activeTab, taskDetails, currentSample, loadingSample, fetchNextSample]);

  // Fetch export data when active tab becomes stats
  useEffect(() => {
    if (activeTab === 'stats' && selectedLabelId) {
      setLoadingExport(true);
      api.labels.export({ task_id: selectedLabelId, labeler_ids: null, include_unfinished: false })
        .then(data => {
          if (data && typeof data === 'object') {
            const arr = Object.keys(data).map(sampleId => ({
              sample_id: sampleId,
              ...data[sampleId]
            }));
            setExportedData(arr);
          } else {
            setExportedData([]);
          }
        })
        .catch(err => {
          console.error("Failed to fetch export data:", err);
          setExportedData([]);
        })
        .finally(() => {
          setLoadingExport(false);
        });
    }
  }, [activeTab, selectedLabelId]);

  useEffect(() => {
    setActiveTab('labeling');
    setCurrentSample(null);
  }, [selectedLabelId]);

  // Handle global drag, resize, and draw events for bounding boxes
  useEffect(() => {
    if (!dragInfo || !containerRef.current || activeTab !== 'labeling') return;

    const handleMouseMove = (e: MouseEvent) => {
      const rect = containerRef.current!.getBoundingClientRect();
      
      if (dragInfo.type === 'draw') {
        const currentX_pct = ((e.clientX - rect.left) / rect.width) * 100;
        const currentY_pct = ((e.clientY - rect.top) / rect.height) * 100;
        const startX_pct = dragInfo.initBox.x;
        const startY_pct = dragInfo.initBox.y;

        const x = Math.min(startX_pct, currentX_pct);
        const y = Math.min(startY_pct, currentY_pct);
        const w = Math.abs(currentX_pct - startX_pct);
        const h = Math.abs(currentY_pct - startY_pct);

        setBbox({
          x: Math.max(0, Math.min(x, 100)),
          y: Math.max(0, Math.min(y, 100)),
          w: Math.max(1, Math.min(w, 100 - x)),
          h: Math.max(1, Math.min(h, 100 - y))
        });
        return;
      }

      const dx = ((e.clientX - dragInfo.startX) / rect.width) * 100;
      const dy = ((e.clientY - dragInfo.startY) / rect.height) * 100;

      let { x, y, w, h } = dragInfo.initBox;

      if (dragInfo.type === 'move') {
        x += dx;
        y += dy;
      } else if (dragInfo.type === 'resize' && dragInfo.handle) {
        if (dragInfo.handle.includes('e')) w += dx;
        if (dragInfo.handle.includes('s')) h += dy;
        if (dragInfo.handle.includes('w')) { x += dx; w -= dx; }
        if (dragInfo.handle.includes('n')) { y += dy; h -= dy; }
      }

      const minSize = 2;
      x = Math.max(0, Math.min(x, 100 - minSize));
      y = Math.max(0, Math.min(y, 100 - minSize));
      w = Math.max(minSize, Math.min(w, 100 - x));
      h = Math.max(minSize, Math.min(h, 100 - y));

      setBbox({ x, y, w, h });
    };

    const handleMouseUp = () => setDragInfo(null);

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [dragInfo, activeTab]);

  const handleSubmit = async (overrideValues?: Record<string, any>) => {
    if (!selectedLabelId || !currentSample || submitting) return;
    
    const valuesToSubmit = overrideValues || formValues;

    // Basic validation check
    if (taskDetails?.label_fields) {
      for (const field of taskDetails.label_fields) {
        if (field.required) {
          const val = valuesToSubmit[field.key];
          if (val === undefined || val === null || val === '' || (Array.isArray(val) && val.length === 0)) {
            alert(`Please fill in required field: ${field.label || field.key}`);
            return;
          }
        }
      }
    }

    setSubmitting(true);
    try {
      const labelData = { ...valuesToSubmit };
      const comment = labelData.comment || "";
      delete labelData.comment; // Optional: separate comment if backend treats it specifically

      // If bbox is drawn, include it in the label payload
      if (bbox.w > 0 && bbox.h > 0) {
        labelData.bbox = bbox;
      }

      const requestData = {
        task_id: selectedLabelId,
        labeler_id: userProfile.username || 'user',
        sample_id: currentSample.sample_id,
        label: labelData,
        comment: comment
      };

      console.log('Submitting label with request data:', requestData);

      await api.labels.submitSample(requestData);

      // Update local progress counter
      setTaskDetails((prev: any) => ({ ...prev, completed_samples: (prev.completed_samples || 0) + 1 }));
      await fetchNextSample();
    } catch (err) {
      console.error('Failed to submit label:', err);
      alert('Failed to submit label');
    } finally {
      setSubmitting(false);
    }
  };

  const handleFieldChange = (key: string, value: any, type: string) => {
    let newValues = { ...formValues };
    
    if (type === 'multiselect') {
      const current = Array.isArray(newValues[key]) ? newValues[key] : [];
      newValues[key] = current.includes(value) 
        ? current.filter((v: any) => v !== value) 
        : [...current, value];
    } else {
      newValues[key] = value;
    }
    
    setFormValues(newValues);

    // Auto submit if enabled and it's a direct selection
    if (autoSubmit && (type === 'select' || type === 'bool')) {
      handleSubmit(newValues);
    }
  };

  // Compute option shortcuts globally (1-9)
  const optionShortcuts: { key: string, value: any, type: string, shortcut: string }[] = [];
  let shortcutIndex = 1;
  if (taskDetails?.label_fields) {
    taskDetails.label_fields.forEach((field: any) => {
      if (field.type === 'select' || field.type === 'multiselect') {
        field.options?.forEach((opt: string) => {
          if (shortcutIndex <= 9) {
            optionShortcuts.push({ key: field.key, value: opt, type: field.type, shortcut: shortcutIndex.toString() });
            shortcutIndex++;
          }
        });
      } else if (field.type === 'bool') {
        if (shortcutIndex <= 8) {
          optionShortcuts.push({ key: field.key, value: true, type: 'bool', shortcut: shortcutIndex.toString() });
          shortcutIndex++;
          optionShortcuts.push({ key: field.key, value: false, type: 'bool', shortcut: shortcutIndex.toString() });
          shortcutIndex++;
        }
      }
    });
  }

  const getShortcut = (fieldKey: string, value: any) => {
    const match = optionShortcuts.find(s => s.key === fieldKey && s.value === value);
    return match ? match.shortcut : null;
  };

  // Keyboard shortcut for Submit & Options
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (activeTab !== 'labeling' || !currentSample || submitting) return;
      
      const activeTag = document.activeElement?.tagName;
      const isTextarea = activeTag === 'TEXTAREA' || activeTag === 'INPUT';
      
      if (e.key === 'Enter') {
        // If focusing a textarea, only submit if Ctrl or Meta (Cmd) is pressed
        if (isTextarea) {
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            handleSubmit();
          }
          return;
        }

        // If focusing a button, let the browser handle its click natively
        if (activeTag === 'BUTTON') {
          return;
        }

        e.preventDefault();
        handleSubmit();
        return;
      }

      // Handle Option Shortcuts (1-9)
      if (!isTextarea && /^[1-9]$/.test(e.key)) {
        const shortcutMatch = optionShortcuts.find(s => s.shortcut === e.key);
        if (shortcutMatch) {
          e.preventDefault();
          handleFieldChange(shortcutMatch.key, shortcutMatch.value, shortcutMatch.type);
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }); // Run on every render so it closures over the latest state

  const handleSaveSettings = async () => {
    if (!selectedLabelId) return;
    setSettingsError('');
    setSavingSettings(true);

    let parsedDisplayFields = [];
    let parsedLabelFields = [];
    let parsedExtra = {};

    try {
      if (settingsForm.display_fields_json.trim()) parsedDisplayFields = JSON.parse(settingsForm.display_fields_json);
      if (settingsForm.label_fields_json.trim()) parsedLabelFields = JSON.parse(settingsForm.label_fields_json);
      if (settingsForm.extra_json.trim()) parsedExtra = JSON.parse(settingsForm.extra_json);
      
      if (!Array.isArray(parsedDisplayFields)) throw new Error("Display fields must be an array");
      if (!Array.isArray(parsedLabelFields)) throw new Error("Label fields must be an array");
    } catch (e: any) {
      setSettingsError("Invalid JSON configuration: " + e.message);
      setSavingSettings(false);
      return;
    }

    try {
      const updated = await api.labels.update({
        id: selectedLabelId,
        name: settingsForm.name,
        description: settingsForm.description,
        ui_html: settingsForm.ui_html,
        display_fields: parsedDisplayFields,
        label_fields: parsedLabelFields,
        extra: parsedExtra
      });
      setTaskDetails(updated);
      alert('Settings saved successfully!');
    } catch (err: any) {
      console.error('Failed to update settings', err);
      setSettingsError('Failed to save settings: ' + err.message);
    } finally {
      setSavingSettings(false);
    }
  };

  const handleShare = () => {
    navigator.clipboard.writeText(`${window.location.origin}/tasks/${selectedLabelId}`);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDelete = async () => {
    if (!selectedLabelId) return;
    if (!confirm('Are you sure you want to delete this label task? This action cannot be undone.')) return;
    
    try {
      await api.labels.delete(selectedLabelId);
      await initData();
      setView('label', null);
    } catch (err) {
      console.error('Failed to delete label task:', err);
      alert('Failed to delete label task');
    }
  };

  const getSampleData = (source: string, key: string) => {
    if (!currentSample) return null;
    if (source === 'meta') return currentSample.meta?.[key];
    return currentSample.data?.[key];
  };

  const renderPreviewContent = (dataObj: any) => {
    if (!dataObj) return '-';
    // Match image URL
    const imgKey = Object.keys(dataObj).find(k => typeof dataObj[k] === 'string' && dataObj[k].match(/\.(jpeg|jpg|gif|png)$/i));
    if (imgKey) {
      return (
        <div className="flex items-center gap-3">
          <div className="w-12 h-10 rounded border border-border bg-secondary overflow-hidden shrink-0 flex items-center justify-center">
             <img src={dataObj[imgKey]} alt="preview" className="w-full h-full object-cover" />
          </div>
        </div>
      );
    }
    // Stringify or show first text field
    const textKey = Object.keys(dataObj).find(k => typeof dataObj[k] === 'string');
    if (textKey) return <div className="truncate max-w-[200px] text-xs" title={dataObj[textKey]}>{dataObj[textKey]}</div>;
    return <div className="truncate max-w-[200px] text-xs">{JSON.stringify(dataObj)}</div>;
  };

  if (!selectedLabelId) {
    const filtered = labelTasks.filter(t => t.name.toLowerCase().includes(search.toLowerCase()));
    return (
      <div className="h-full flex flex-col bg-background p-8">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold flex items-center gap-3">
            <Tag className="text-purple-400" /> Label Tasks Overview
          </h2>
          <div className="relative w-64">
            <Search className="absolute left-3 top-2.5 text-muted-foreground" size={16} />
            <input 
              type="text" 
              placeholder="Search label tasks..." 
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="w-full pl-9 pr-4 py-2 bg-secondary/30 border border-border rounded-lg text-sm focus:outline-none focus:ring-1 focus:ring-primary"
            />
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 overflow-y-auto">
          {filtered.map(t => (
            <div 
              key={t.id} 
              onClick={() => setView('label', t.id)}
              className="p-5 bg-card border border-border rounded-xl cursor-pointer hover:border-purple-500/50 transition-colors group"
            >
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="font-semibold group-hover:text-purple-400 transition-colors truncate">{t.name}</h3>
                  <span className="text-[10px] uppercase text-muted-foreground bg-secondary px-2 py-0.5 rounded mt-1 inline-block">
                    {t.type === 'image_bbox' ? 'Image BBox' : 'Text / Configurable'}
                  </span>
                </div>
                <Tag size={16} className="text-muted-foreground" />
              </div>
              <div className="flex justify-between text-xs mb-1 text-muted-foreground">
                <span>Progress: {t.progress} / {t.total}</span>
                <span>{t.total > 0 ? Math.round((t.progress / t.total) * 100) : 0}%</span>
              </div>
              <div className="h-1.5 w-full bg-secondary rounded-full overflow-hidden">
                <div 
                  className="h-full bg-purple-500 transition-all" 
                  style={{ width: `${t.total > 0 ? (t.progress / t.total) * 100 : 0}%` }} 
                />
              </div>
            </div>
          ))}
          {filtered.length === 0 && (
            <div className="col-span-full py-12 text-center text-muted-foreground">No tasks found matching "{search}"</div>
          )}
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center text-muted-foreground">
        <Loader2 className="animate-spin mr-2" size={24} /> Loading task details...
      </div>
    );
  }

  if (!taskDetails) return <div className="p-8 text-muted-foreground">Task not found</div>;

  const total = taskDetails.total_samples || 0;
  const progressDisplay = taskDetails.completed_samples || 0;
  const isImageTask = taskDetails.display_fields?.some((f: any) => f.type === 'image') || false;

  return (
    <div className="h-full flex flex-col bg-background p-8 overflow-hidden">
      {/* Header Area */}
      <div className="flex justify-between items-start gap-4 mb-6 shrink-0">
        <div className="flex-1 min-w-0">
          <h2 className="text-2xl font-bold flex items-center gap-3 mb-1">
            <Tag className="text-purple-400 shrink-0" /> {taskDetails.name}
          </h2>
          <p className="text-sm text-muted-foreground mb-3 truncate">{taskDetails.description}</p>
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <span className="font-medium">Progress: {progressDisplay} / {total}</span>
            <div className="w-32 md:w-48 h-2 bg-secondary rounded-full overflow-hidden">
              <div className="h-full bg-purple-500 transition-all duration-300" style={{ width: `${total > 0 ? (progressDisplay / total) * 100 : 0}%` }} />
            </div>
            <span className="px-2 py-0.5 bg-secondary text-xs rounded-md uppercase tracking-wider">{taskDetails.status}</span>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {/* Tabs */}
          <div className="flex bg-secondary/50 p-1 rounded-lg shrink-0">
            <button 
              onClick={() => setActiveTab('labeling')}
              className={cn("px-4 py-1.5 rounded-md text-sm font-medium transition-all", activeTab === 'labeling' ? "bg-background shadow-sm text-foreground" : "text-muted-foreground hover:text-foreground")}
            >
              Labeling
            </button>
            <button 
              onClick={() => setActiveTab('stats')}
              className={cn("px-4 py-1.5 rounded-md text-sm font-medium transition-all flex items-center justify-center gap-2", activeTab === 'stats' ? "bg-background shadow-sm text-foreground" : "text-muted-foreground hover:text-foreground")}
            >
              <BarChart2 size={14} /> Stats
            </button>
            <button 
              onClick={() => setActiveTab('settings')}
              className={cn("px-4 py-1.5 rounded-md text-sm font-medium transition-all flex items-center justify-center gap-2", activeTab === 'settings' ? "bg-background shadow-sm text-foreground" : "text-muted-foreground hover:text-foreground")}
            >
              <Settings2 size={14} /> Settings
            </button>
          </div>
          
          <button 
            title="Share Task"
            onClick={handleShare}
            className="p-2 bg-card border border-border hover:bg-secondary rounded-lg transition-colors ml-2"
          >
            {copied ? <Check size={18} className="text-green-500" /> : <Share2 size={18} className="text-purple-400" />}
          </button>
          <button 
            title="Delete Task"
            onClick={handleDelete}
            className="p-2 bg-destructive/10 text-destructive hover:bg-destructive/20 rounded-lg transition-colors ml-2"
          >
            <Trash2 size={18} />
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      {activeTab === 'labeling' && (
        <div className="flex-1 flex flex-col gap-6 min-h-0 relative">
          {loadingSample && (
            <div className="absolute inset-0 bg-background/50 z-10 flex items-center justify-center backdrop-blur-sm rounded-xl">
              <Loader2 className="animate-spin text-primary" size={32} />
            </div>
          )}
          
          {/* Top: Sample Data Display */}
          <div className="flex-1 bg-card border border-border rounded-xl p-6 flex flex-col min-h-[250px] overflow-hidden relative">
            <div className="flex items-center justify-between mb-4 shrink-0">
              <h3 className="text-xs font-medium uppercase tracking-wider text-muted-foreground flex items-center gap-2">
                <Info size={14} /> Sample Display
                {currentSample && <span className="ml-2 px-2 py-0.5 bg-secondary text-[10px] rounded-md font-mono">{currentSample.sample_id?.substring(0, 8)}</span>}
              </h3>
              {isImageTask && (
                <div className="flex gap-2">
                  <button onClick={() => setActiveTool('crosshair')} className={cn("p-1.5 rounded transition-colors", activeTool === 'crosshair' ? "bg-primary/20 text-primary" : "bg-secondary text-foreground hover:bg-secondary/80")} title="Draw new box"><Crosshair size={14}/></button>
                  <button onClick={() => setActiveTool('pointer')} className={cn("p-1.5 rounded transition-colors", activeTool === 'pointer' ? "bg-primary/20 text-primary" : "bg-secondary text-foreground hover:bg-secondary/80")} title="Move/Resize existing box"><MousePointer2 size={14}/></button>
                  <button onClick={() => setActiveTool('zoom')} className={cn("p-1.5 rounded transition-colors", activeTool === 'zoom' ? "bg-primary/20 text-primary" : "bg-secondary text-foreground hover:bg-secondary/80")} title="Zoom"><ZoomIn size={14}/></button>
                  <button 
                    onClick={() => setBbox({ x: 0, y: 0, w: 0, h: 0 })} 
                    disabled={bbox.w === 0 || bbox.h === 0}
                    className={cn("p-1.5 rounded transition-colors ml-2", bbox.w > 0 && bbox.h > 0 ? "bg-secondary text-destructive hover:bg-destructive/20" : "bg-secondary/50 text-muted-foreground opacity-50 cursor-not-allowed")} 
                    title="Clear Bounding Box"
                  >
                    <Trash2 size={14}/>
                  </button>
                </div>
              )}
            </div>
            
            <div className="flex-1 flex flex-col gap-4 overflow-hidden pr-2">
              {!currentSample && !loadingSample ? (
                <div className="flex-1 flex items-center justify-center text-muted-foreground">
                  No sample available to label.
                </div>
              ) : taskDetails.display_fields?.map((field: any, idx: number) => {
                const val = getSampleData(field.source, field.key);
                
                if (field.type === 'image') {
                  return (
                    <div key={field.key || idx} className="flex-1 bg-black/5 rounded-lg overflow-hidden relative border border-border flex items-center justify-center min-h-[200px] max-h-[500px]">
                      <div 
                        ref={containerRef}
                        className={cn("relative w-full h-full transition-transform duration-200 origin-center flex items-center justify-center",
                          activeTool === 'crosshair' && "cursor-crosshair",
                          activeTool === 'pointer' && "cursor-default",
                          activeTool === 'zoom' && (zoom >= 3 ? "cursor-zoom-out" : "cursor-zoom-in")
                        )}
                        style={{ transform: `scale(${zoom})` }}
                        onMouseDown={(e) => {
                          if (activeTool === 'zoom') {
                            if (e.shiftKey || zoom >= 3) setZoom(1);
                            else setZoom(z => Math.min(3, z + 1));
                            return;
                          }
                          if (activeTool === 'crosshair') {
                            const rect = containerRef.current!.getBoundingClientRect();
                            const x = ((e.clientX - rect.left) / rect.width) * 100;
                            const y = ((e.clientY - rect.top) / rect.height) * 100;
                            setBbox({ x, y, w: 0, h: 0 });
                            setDragInfo({ type: 'draw', startX: e.clientX, startY: e.clientY, initBox: { x, y, w: 0, h: 0 } });
                          }
                        }}
                      >
                        {val ? <img src={val} alt="Task" className="w-full h-full object-contain pointer-events-none select-none" /> : <div className="text-muted-foreground min-h-[200px] flex items-center">No image source</div>}
                        {bbox.w > 0 && bbox.h > 0 && (
                          <div 
                            className="absolute border-2 border-blue-500 bg-blue-500/10 group shadow-sm"
                            style={{ left: `${bbox.x}%`, top: `${bbox.y}%`, width: `${bbox.w}%`, height: `${bbox.h}%`, cursor: activeTool === 'pointer' ? 'move' : 'inherit' }}
                            onMouseDown={(e) => {
                              if (activeTool !== 'pointer') return;
                              e.stopPropagation();
                              setDragInfo({ type: 'move', startX: e.clientX, startY: e.clientY, initBox: { ...bbox } });
                            }}
                          >
                            {activeTool === 'pointer' && (
                              <>
                                <div className="absolute -top-1.5 -left-1.5 w-3 h-3 bg-white border-2 border-blue-500 rounded-full cursor-nwse-resize" onMouseDown={(e) => { e.stopPropagation(); setDragInfo({ type: 'resize', handle: 'nw', startX: e.clientX, startY: e.clientY, initBox: { ...bbox } }); }} />
                                <div className="absolute -top-1.5 -right-1.5 w-3 h-3 bg-white border-2 border-blue-500 rounded-full cursor-nesw-resize" onMouseDown={(e) => { e.stopPropagation(); setDragInfo({ type: 'resize', handle: 'ne', startX: e.clientX, startY: e.clientY, initBox: { ...bbox } }); }} />
                                <div className="absolute -bottom-1.5 -left-1.5 w-3 h-3 bg-white border-2 border-blue-500 rounded-full cursor-nesw-resize" onMouseDown={(e) => { e.stopPropagation(); setDragInfo({ type: 'resize', handle: 'sw', startX: e.clientX, startY: e.clientY, initBox: { ...bbox } }); }} />
                                <div className="absolute -bottom-1.5 -right-1.5 w-3 h-3 bg-white border-2 border-blue-500 rounded-full cursor-nwse-resize" onMouseDown={(e) => { e.stopPropagation(); setDragInfo({ type: 'resize', handle: 'se', startX: e.clientX, startY: e.clientY, initBox: { ...bbox } }); }} />
                              </>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                }
                
                return (
                  <div key={field.key || idx} className="bg-secondary/30 rounded-lg p-6 overflow-y-auto">
                    <span className="text-[10px] uppercase font-bold text-muted-foreground block mb-2">{field.label || field.key}</span>
                    <div className="text-lg leading-relaxed text-foreground">
                      {val !== undefined ? String(val) : '-'}
                    </div>
                  </div>
                );
              })}

              {!taskDetails.display_fields?.length && currentSample && (
                <div className="flex-1 bg-secondary/30 rounded-lg p-8 overflow-auto">
                  <pre className="text-xs">{JSON.stringify(currentSample.data, null, 2)}</pre>
                </div>
              )}
            </div>
          </div>

          {/* Bottom: Actions Panel */}
          <div className="bg-card border border-border rounded-xl p-6 flex flex-col shrink-0 max-h-[50%] overflow-y-auto relative">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                Annotations
              </h3>
            </div>
            
            <div className="space-y-6 flex-1 pr-2">
              {taskDetails.label_fields?.map((field: any) => (
                <div key={field.key}>
                  <label className="text-sm font-semibold mb-2 flex items-center justify-between">
                    {field.label}
                    {field.required && <span className="text-[10px] text-destructive uppercase">Required</span>}
                  </label>
                  
                  {field.type === 'select' && (
                    <div className="flex flex-wrap gap-2">
                      {field.options?.map((opt: string) => {
                        const shortcut = getShortcut(field.key, opt);
                        return (
                          <button 
                            key={opt}
                            onClick={() => handleFieldChange(field.key, opt, 'select')}
                            disabled={!currentSample}
                            className={cn("px-4 py-2 text-sm border rounded-lg transition-colors w-auto min-w-[120px] flex items-center justify-between gap-3", 
                              formValues[field.key] === opt 
                                ? "bg-primary/10 border-primary text-primary font-medium" 
                                : "bg-background border-border hover:border-primary/50 text-foreground disabled:opacity-50"
                            )}
                          >
                            <span>{opt}</span>
                            {shortcut && <kbd className="font-mono text-[10px] bg-secondary text-muted-foreground px-1.5 py-0.5 rounded border border-border shadow-sm ml-2">{shortcut}</kbd>}
                          </button>
                        );
                      })}
                    </div>
                  )}

                  {field.type === 'bool' && (
                    <div className="flex flex-wrap gap-2">
                      <button 
                        onClick={() => handleFieldChange(field.key, true, 'bool')}
                        disabled={!currentSample}
                        className={cn("px-4 py-2 text-sm border rounded-lg transition-colors w-auto min-w-[120px] flex items-center justify-between gap-3", 
                          formValues[field.key] === true 
                            ? "bg-green-500/10 border-green-500 text-green-600 font-medium" 
                            : "bg-background border-border hover:border-green-500/50 text-foreground disabled:opacity-50"
                        )}
                      >
                        <span>True</span>
                        {getShortcut(field.key, true) && <kbd className="font-mono text-[10px] bg-secondary text-muted-foreground px-1.5 py-0.5 rounded border border-border shadow-sm ml-2">{getShortcut(field.key, true)}</kbd>}
                      </button>
                      <button 
                        onClick={() => handleFieldChange(field.key, false, 'bool')}
                        disabled={!currentSample}
                        className={cn("px-4 py-2 text-sm border rounded-lg transition-colors w-auto min-w-[120px] flex items-center justify-between gap-3", 
                          formValues[field.key] === false 
                            ? "bg-red-500/10 border-red-500 text-red-600 font-medium" 
                            : "bg-background border-border hover:border-red-500/50 text-foreground disabled:opacity-50"
                        )}
                      >
                        <span>False</span>
                        {getShortcut(field.key, false) && <kbd className="font-mono text-[10px] bg-secondary text-muted-foreground px-1.5 py-0.5 rounded border border-border shadow-sm ml-2">{getShortcut(field.key, false)}</kbd>}
                      </button>
                    </div>
                  )}

                  {field.type === 'multiselect' && (
                    <div className="flex flex-wrap gap-2">
                      {field.options?.map((opt: string) => {
                        const isSelected = Array.isArray(formValues[field.key]) && formValues[field.key].includes(opt);
                        const shortcut = getShortcut(field.key, opt);
                        return (
                          <button 
                            key={opt}
                            onClick={() => handleFieldChange(field.key, opt, 'multiselect')}
                            disabled={!currentSample}
                            className={cn("px-4 py-2 text-sm border rounded-lg transition-colors w-auto min-w-[120px] flex items-center justify-between gap-3", 
                              isSelected 
                                ? "bg-primary/10 border-primary text-primary font-medium" 
                                : "bg-background border-border hover:border-primary/50 text-foreground disabled:opacity-50"
                            )}
                          >
                            <span>{opt}</span>
                            {shortcut && <kbd className="font-mono text-[10px] bg-secondary text-muted-foreground px-1.5 py-0.5 rounded border border-border shadow-sm ml-2">{shortcut}</kbd>}
                          </button>
                        );
                      })}
                    </div>
                  )}

                  {field.type === 'text' && (
                    <textarea 
                      rows={2}
                      value={formValues[field.key] || ''}
                      onChange={(e) => handleFieldChange(field.key, e.target.value, 'text')}
                      disabled={!currentSample}
                      placeholder="Enter text..."
                      className="w-full bg-secondary/30 border border-border rounded-lg p-3 text-sm focus:outline-none focus:ring-1 focus:ring-primary resize-y disabled:opacity-50"
                    />
                  )}
                </div>
              ))}

              {!taskDetails.label_fields?.length && (
                 <div className="text-sm text-muted-foreground italic text-center py-4">
                   No label fields configured.
                 </div>
              )}
            </div>

            <div className="pt-4 mt-4 border-t border-border flex items-center justify-between gap-3">
              <label className="flex items-center gap-2 text-xs cursor-pointer select-none ml-2 shrink-0">
                <input 
                  type="checkbox" 
                  checked={autoSubmit} 
                  onChange={(e) => setAutoSubmit(e.target.checked)}
                  className="rounded border-border text-primary focus:ring-primary bg-secondary/50"
                />
                <span className="text-muted-foreground font-medium">Auto-submit</span>
              </label>

              <div className="flex items-center gap-3 w-full justify-end">
                <button 
                  onClick={fetchNextSample}
                  disabled={submitting}
                  className="flex items-center justify-center gap-2 px-6 py-2 bg-secondary hover:bg-secondary/80 text-foreground rounded-lg text-sm font-medium transition-colors disabled:opacity-50 shrink-0"
                >
                  Skip Example
                </button>
                <button 
                  onClick={() => handleSubmit()}
                  disabled={!currentSample || submitting}
                  className="flex items-center justify-center gap-2 px-6 py-2 bg-primary hover:bg-primary/90 text-primary-foreground rounded-lg text-sm font-medium transition-colors shadow-sm disabled:opacity-50"
                >
                  {submitting ? <Loader2 className="animate-spin" size={16} /> : <>Submit Label (↵)</>}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {activeTab === 'stats' && (
        <div className="flex-1 overflow-y-auto space-y-6 pb-6 relative">
          {loadingExport && (
            <div className="absolute inset-0 bg-background/50 z-10 flex items-center justify-center backdrop-blur-sm">
              <Loader2 className="animate-spin text-primary" size={32} />
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Completion Card */}
            <div className="bg-card border border-border rounded-xl p-5 shadow-sm">
              <div className="flex justify-between items-start mb-4">
                <div className="w-10 h-10 rounded-lg bg-purple-500/10 flex items-center justify-center text-purple-500">
                  <CheckCircle2 size={20} />
                </div>
                <span className="text-xs font-medium px-2 py-1 bg-secondary rounded-md text-muted-foreground">Status</span>
              </div>
              <h3 className="text-2xl font-bold mb-1">{taskDetails.completed_samples} / {total}</h3>
              <p className="text-sm text-muted-foreground">Cases Fully Labeled (All Labelers)</p>
            </div>

            {/* Labelers Card */}
            <div className="bg-card border border-border rounded-xl p-5 shadow-sm overflow-hidden flex flex-col max-h-64">
              <div className="flex justify-between items-start mb-4 shrink-0">
                <div className="w-10 h-10 rounded-lg bg-green-500/10 flex items-center justify-center text-green-500">
                  <Users size={20} />
                </div>
                <span className="text-xs font-medium px-2 py-1 bg-secondary rounded-md text-muted-foreground">Active</span>
              </div>
              <h3 className="text-2xl font-bold mb-1 shrink-0">{taskDetails.labelers?.length || 0} <span className="text-sm font-normal text-muted-foreground">Users</span></h3>
              <p className="text-sm text-muted-foreground shrink-0 mb-4">Working on this task</p>
              <div className="space-y-2 overflow-y-auto flex-1 min-h-0 pr-2">
                {taskDetails.labelers?.map((user: any, idx: number) => (
                  <div key={idx} className="flex justify-between text-xs items-center p-1.5 hover:bg-secondary/50 rounded-md transition-colors">
                    <span className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-green-500 shrink-0"></div>
                      <span className="truncate max-w-[120px]">{user.name}</span>
                    </span>
                    <span className="font-medium text-foreground">{user.completed} <span className="text-muted-foreground font-normal">/ {user.assigned}</span></span>
                  </div>
                ))}
                {!taskDetails.labelers?.length && (
                  <div className="text-xs text-muted-foreground">No active users</div>
                )}
              </div>
            </div>
          </div>

          {/* Labeled Data Preview */}
          <div className="bg-card border border-border rounded-xl overflow-hidden shadow-sm flex flex-col">
            <div className="p-4 border-b border-border flex items-center gap-2">
              <Table size={18} className="text-muted-foreground" />
              <h3 className="font-semibold">Recent Labeled Data Preview</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead className="text-xs text-muted-foreground uppercase bg-secondary/30">
                  <tr>
                    <th className="px-6 py-3 border-b border-border">Data ID</th>
                    <th className="px-6 py-3 border-b border-border">Content Preview</th>
                    <th className="px-6 py-3 border-b border-border">Label Details</th>
                    <th className="px-6 py-3 border-b border-border">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {exportedData.length > 0 ? exportedData.map((row, idx) => (
                    <tr key={idx} className="border-b border-border hover:bg-secondary/20">
                      <td className="px-6 py-3 font-mono text-xs text-muted-foreground whitespace-nowrap">{row.sample_id.substring(0, 12)}...</td>
                      <td className="px-6 py-3 min-w-[200px]">
                        {renderPreviewContent(row.data)}
                      </td>
                      <td className="px-6 py-3">
                        <div className="flex flex-col gap-1.5">
                          {row.labels && Object.entries(row.labels).map(([user, data]: [string, any]) => (
                            <div key={user} className="bg-background border border-border rounded px-2 py-1 text-[11px]">
                              <div className="font-medium text-muted-foreground mb-0.5 flex justify-between">
                                <span>{user}</span>
                                {data.labeled_at && <span className="font-normal opacity-70">{new Date(data.labeled_at).toLocaleTimeString()}</span>}
                              </div>
                              <div className="font-mono text-foreground break-all">{JSON.stringify(data.label)}</div>
                            </div>
                          ))}
                        </div>
                      </td>
                      <td className="px-6 py-3 whitespace-nowrap">
                        <span className={cn(
                          "px-2 py-0.5 rounded text-[10px] uppercase font-bold tracking-wider",
                          "bg-green-500/10 text-green-600"
                        )}>
                          COMPLETED
                        </span>
                      </td>
                    </tr>
                  )) : (
                    <tr>
                      <td colSpan={4} className="px-6 py-12 text-center text-muted-foreground">
                        No labeled data available to preview.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'settings' && (
        <div className="flex-1 overflow-y-auto pb-6">
          <div className="max-w-4xl mx-auto space-y-6">
            <div className="bg-card border border-border rounded-xl p-6 shadow-sm">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold flex items-center gap-2">
                  <Settings2 size={18} /> Task Configuration
                </h3>
              </div>

              {settingsError && (
                <div className="mb-6 p-3 bg-destructive/10 text-destructive border border-destructive/20 rounded-lg text-sm">
                  {settingsError}
                </div>
              )}

              <div className="space-y-6">
                
                <div>
                  <label className="text-sm font-medium mb-2 block">Task Name</label>
                  <input
                    type="text"
                    value={settingsForm.name}
                    onChange={(e) => setSettingsForm(prev => ({ ...prev, name: e.target.value }))}
                    className="w-full bg-secondary/30 border border-border rounded-lg p-2.5 text-sm focus:outline-none focus:ring-1 focus:ring-primary"
                    placeholder="Task Name"
                  />
                </div>
                <div>
                  <label className="text-sm font-medium mb-2 block">Description</label>
                  <input
                    type="text"
                    value={settingsForm.description}
                    onChange={(e) => setSettingsForm(prev => ({ ...prev, description: e.target.value }))}
                    className="w-full bg-secondary/30 border border-border rounded-lg p-2.5 text-sm focus:outline-none focus:ring-1 focus:ring-primary"
                    placeholder="Task Description"
                  />
                </div>

                <hr className="border-border" />

                <div>
                  <label className="text-sm font-medium mb-2 block">Display Fields (JSON)</label>
                  <p className="text-xs text-muted-foreground mb-2">Configure how data is rendered.</p>
                  <textarea
                    value={settingsForm.display_fields_json}
                    onChange={(e) => setSettingsForm(prev => ({ ...prev, display_fields_json: e.target.value }))}
                    className="w-full h-48 bg-secondary/30 border border-border rounded-lg p-3 font-mono text-xs focus:outline-none focus:ring-1 focus:ring-primary resize-y"
                    placeholder='[{"key": "image_url", "label": "Image", "source": "data", "type": "image"}]'
                  />
                </div>

                <div>
                  <label className="text-sm font-medium mb-2 block">Label Fields (JSON)</label>
                  <p className="text-xs text-muted-foreground mb-2">Define the input fields for the annotation task.</p>
                  <textarea
                    value={settingsForm.label_fields_json}
                    onChange={(e) => setSettingsForm(prev => ({ ...prev, label_fields_json: e.target.value }))}
                    className="w-full h-48 bg-secondary/30 border border-border rounded-lg p-3 font-mono text-xs focus:outline-none focus:ring-1 focus:ring-primary resize-y"
                    placeholder='[{"key": "quality", "type": "select", "options": ["good"]}]'
                  />
                </div>

                <div>
                  <label className="text-sm font-medium mb-2 block">Custom UI HTML</label>
                  <p className="text-xs text-muted-foreground mb-2">Override the default annotation UI with custom HTML.</p>
                  <textarea
                    value={settingsForm.ui_html}
                    onChange={(e) => setSettingsForm(prev => ({ ...prev, ui_html: e.target.value }))}
                    className="w-full h-32 bg-secondary/30 border border-border rounded-lg p-3 font-mono text-xs focus:outline-none focus:ring-1 focus:ring-primary resize-y"
                    placeholder="<div>...</div>"
                  />
                </div>

                <div>
                  <label className="text-sm font-medium mb-2 block">Extra Configuration (JSON)</label>
                  <p className="text-xs text-muted-foreground mb-2">Additional task metadata or configuration.</p>
                  <textarea
                    value={settingsForm.extra_json}
                    onChange={(e) => setSettingsForm(prev => ({ ...prev, extra_json: e.target.value }))}
                    className="w-full h-32 bg-secondary/30 border border-border rounded-lg p-3 font-mono text-xs focus:outline-none focus:ring-1 focus:ring-primary resize-y"
                    placeholder='{"reviewer": "alice"}'
                  />
                </div>

                <div className="pt-6 border-t border-border flex justify-end">
                  <button 
                    onClick={handleSaveSettings}
                    disabled={savingSettings}
                    className="flex items-center gap-2 px-6 py-2 bg-primary hover:bg-primary/90 text-primary-foreground rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
                  >
                    {savingSettings ? <Loader2 size={16} className="animate-spin" /> : <Save size={16} />}
                    Save Settings
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
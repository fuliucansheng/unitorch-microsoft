import { useState, useEffect, useCallback, useRef } from 'react';
import { useStore } from '../../store/useStore';
import { Tag, Search, CornerDownLeft, Crosshair, MousePointer2, ZoomIn, BarChart2, Users, CheckCircle2, Share2, Check, Table, Image as ImageIcon } from 'lucide-react';
import { cn } from '../../lib/utils';

const MOCK_TEXT_DATA = [
  "The new update completely broke my workflow. It crashes every time I try to export the model weights. Very frustrating!",
  "Absolutely love the new features! The interface is so much cleaner and faster now.",
  "It's okay, nothing special but it gets the job done I suppose.",
  "Customer support was incredibly helpful and resolved my issue within minutes. 5 stars!",
  "I'm having trouble figuring out how to connect my database. The documentation is lacking."
];

const MOCK_IMAGE_DATA = [
  "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?auto=format&fit=crop&w=1200&q=80",
  "https://images.unsplash.com/photo-1568605117036-5fe5e7bab0b7?auto=format&fit=crop&w=1200&q=80",
  "https://images.unsplash.com/photo-1502877338535-766e1452684a?auto=format&fit=crop&w=1200&q=80",
  "https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?auto=format&fit=crop&w=1200&q=80",
  "https://images.unsplash.com/photo-1511919884226-fd3cad34687c?auto=format&fit=crop&w=1200&q=80"
];

export function LabelView() {
  const { labelTasks, selectedLabelId, setView } = useStore();
  const [search, setSearch] = useState('');
  
  // View states
  const [activeTab, setActiveTab] = useState<'labeling' | 'stats'>('labeling');
  const [copied, setCopied] = useState(false);

  // Labeling states
  const [dataIndex, setDataIndex] = useState(0);
  const [autoSubmit, setAutoSubmit] = useState(false);
  const [isMultiSelect, setIsMultiSelect] = useState(false);
  const [selectedLabels, setSelectedLabels] = useState<string[]>([]);
  
  const task = labelTasks.find(t => t.id === selectedLabelId);
  const isImageTask = task?.type === 'image_bbox';

  // BBox Interactive States
  const [bbox, setBbox] = useState({ x: 25, y: 35, w: 45, h: 30 });
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

  // Reset states when switching images or tasks
  useEffect(() => {
    setBbox({ x: 25, y: 35, w: 45, h: 30 });
    setZoom(1);
  }, [dataIndex, selectedLabelId]);

  // Reset tab to labeling when task changes
  useEffect(() => {
    setActiveTab('labeling');
  }, [selectedLabelId]);

  // Handle global drag, resize, and draw events
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

  const handleSubmit = useCallback(() => {
    if (selectedLabels.length === 0 && !isImageTask) return;
    setDataIndex(prev => prev + 1);
    setSelectedLabels([]);
  }, [selectedLabels, isImageTask]);

  const handleSelectLabel = useCallback((labelId: string) => {
    if (isMultiSelect) {
      setSelectedLabels(prev => 
        prev.includes(labelId) ? prev.filter(l => l !== labelId) : [...prev, labelId]
      );
    } else {
      setSelectedLabels([labelId]);
      if (autoSubmit) {
        setTimeout(() => {
          setDataIndex(prev => prev + 1);
          setSelectedLabels([]);
        }, 150);
      }
    }
  }, [isMultiSelect, autoSubmit]);

  useEffect(() => {
    if (!task || activeTab !== 'labeling') return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (document.activeElement?.tagName === 'INPUT' || document.activeElement?.tagName === 'TEXTAREA') return;
      
      if (e.key === '1') handleSelectLabel(isImageTask ? 'car' : 'positive');
      if (e.key === '2') handleSelectLabel(isImageTask ? 'pedestrian' : 'neutral');
      if (e.key === '3') handleSelectLabel(isImageTask ? 'traffic_light' : 'negative');
      if (e.key === 'Enter') {
        e.preventDefault();
        handleSubmit();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [task, handleSelectLabel, handleSubmit, isImageTask, activeTab]);

  const handleShare = () => {
    navigator.clipboard.writeText(`${window.location.origin}/tasks/${task?.id}`);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
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
                    {t.type === 'image_bbox' ? 'Image BBox' : 'Text'}
                  </span>
                </div>
                <Tag size={16} className="text-muted-foreground" />
              </div>
              <div className="flex justify-between text-xs mb-1 text-muted-foreground">
                <span>Progress: {t.progress} / {t.total}</span>
                <span>{Math.round((t.progress / t.total) * 100)}%</span>
              </div>
              <div className="h-1.5 w-full bg-secondary rounded-full overflow-hidden">
                <div 
                  className="h-full bg-purple-500 transition-all" 
                  style={{ width: `${(t.progress / t.total) * 100}%` }} 
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

  if (!task) return <div className="p-8 text-muted-foreground">Task not found</div>;

  const currentText = MOCK_TEXT_DATA[dataIndex % MOCK_TEXT_DATA.length];
  const currentImage = MOCK_IMAGE_DATA[dataIndex % MOCK_IMAGE_DATA.length];
  const progressDisplay = task.progress + dataIndex;

  // Mock data for Stats View
  const mockPreviewRows = Array.from({ length: 5 }).map((_, i) => ({
    id: `ITEM-${task.id}-00${i+1}`,
    isImage: isImageTask,
    contentUrl: isImageTask ? MOCK_IMAGE_DATA[i % MOCK_IMAGE_DATA.length] : undefined,
    contentText: isImageTask ? '' : MOCK_TEXT_DATA[i % MOCK_TEXT_DATA.length],
    bboxCount: isImageTask ? (i % 3) + 1 : 0,
    label: i % 2 === 0 ? (isImageTask ? 'car' : 'positive') : (isImageTask ? 'pedestrian' : 'neutral'),
    agreement: i % 3 === 0 ? '100%' : '66%',
    status: i % 4 === 0 ? 'Review Needed' : 'Consensus Reached'
  }));

  return (
    <div className="h-full flex flex-col bg-background p-8 overflow-hidden">
      {/* Header Area */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-6 shrink-0">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-3 mb-2">
            <Tag className="text-purple-400" /> {task.name}
          </h2>
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <span>Overall Progress: {progressDisplay} / {task.total}</span>
            <div className="w-32 md:w-48 h-2 bg-secondary rounded-full overflow-hidden">
              <div className="h-full bg-purple-500 transition-all duration-300" style={{ width: `${(progressDisplay / task.total) * 100}%` }} />
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3 w-full md:w-auto">
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
              className={cn("px-4 py-1.5 rounded-md text-sm font-medium transition-all flex items-center gap-2", activeTab === 'stats' ? "bg-background shadow-sm text-foreground" : "text-muted-foreground hover:text-foreground")}
            >
              <BarChart2 size={14} /> Stats
            </button>
          </div>
          
          <button 
            onClick={handleShare}
            className="flex items-center gap-2 px-3 py-1.5 bg-card border border-border hover:bg-secondary rounded-lg text-sm font-medium transition-colors ml-auto md:ml-0"
          >
            {copied ? <Check size={16} className="text-green-500" /> : <Share2 size={16} className="text-purple-400" />}
            <span className="hidden sm:inline">{copied ? 'Copied Link' : 'Share Task'}</span>
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      {activeTab === 'labeling' ? (
        <div className="flex-1 flex flex-col gap-6 min-h-0">
          {/* Top: Sample Data */}
          <div className="flex-1 bg-card border border-border rounded-xl p-6 flex flex-col min-h-[300px]">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                {isImageTask ? 'Image Canvas' : 'Sample Data'}
              </h3>
              {isImageTask && (
                <div className="flex gap-2">
                  <button 
                    onClick={() => setActiveTool('crosshair')} 
                    className={cn("p-1.5 rounded transition-colors", activeTool === 'crosshair' ? "bg-primary/20 text-primary" : "bg-secondary text-foreground hover:bg-secondary/80")}
                    title="Draw new box"
                  >
                    <Crosshair size={14}/>
                  </button>
                  <button 
                    onClick={() => setActiveTool('pointer')} 
                    className={cn("p-1.5 rounded transition-colors", activeTool === 'pointer' ? "bg-primary/20 text-primary" : "bg-secondary text-foreground hover:bg-secondary/80")}
                    title="Move/Resize existing box"
                  >
                    <MousePointer2 size={14}/>
                  </button>
                  <button 
                    onClick={() => setActiveTool('zoom')} 
                    className={cn("p-1.5 rounded transition-colors", activeTool === 'zoom' ? "bg-primary/20 text-primary" : "bg-secondary text-foreground hover:bg-secondary/80")}
                    title="Click to zoom in, Shift+Click to zoom out"
                  >
                    <ZoomIn size={14}/>
                  </button>
                </div>
              )}
            </div>
            
            {isImageTask ? (
              <div className="flex-1 bg-black/5 rounded-lg overflow-hidden relative border border-border flex items-center justify-center">
                <div 
                  ref={containerRef}
                  className={cn(
                    "relative w-full h-full transition-transform duration-200 origin-center flex items-center justify-center",
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
                      setDragInfo({ 
                        type: 'draw', 
                        startX: e.clientX, 
                        startY: e.clientY, 
                        initBox: { x, y, w: 0, h: 0 } 
                      });
                    }
                  }}
                >
                  <img src={currentImage} alt="Task" className="w-full h-full object-contain pointer-events-none select-none" />
                  
                  {bbox.w > 0 && bbox.h > 0 && (
                    <div 
                      className="absolute border-2 border-blue-500 bg-blue-500/10 group shadow-sm"
                      style={{ 
                        left: `${bbox.x}%`, 
                        top: `${bbox.y}%`, 
                        width: `${bbox.w}%`, 
                        height: `${bbox.h}%`,
                        cursor: activeTool === 'pointer' ? 'move' : 'inherit'
                      }}
                      onMouseDown={(e) => {
                        if (activeTool !== 'pointer') return;
                        e.stopPropagation();
                        setDragInfo({ type: 'move', startX: e.clientX, startY: e.clientY, initBox: { ...bbox } });
                      }}
                    >
                      {activeTool === 'pointer' && (
                        <>
                          <div 
                            className="absolute -top-1.5 -left-1.5 w-3 h-3 bg-white border-2 border-blue-500 rounded-full cursor-nwse-resize"
                            onMouseDown={(e) => { e.stopPropagation(); setDragInfo({ type: 'resize', handle: 'nw', startX: e.clientX, startY: e.clientY, initBox: { ...bbox } }); }}
                          />
                          <div 
                            className="absolute -top-1.5 -right-1.5 w-3 h-3 bg-white border-2 border-blue-500 rounded-full cursor-nesw-resize"
                            onMouseDown={(e) => { e.stopPropagation(); setDragInfo({ type: 'resize', handle: 'ne', startX: e.clientX, startY: e.clientY, initBox: { ...bbox } }); }}
                          />
                          <div 
                            className="absolute -bottom-1.5 -left-1.5 w-3 h-3 bg-white border-2 border-blue-500 rounded-full cursor-nesw-resize"
                            onMouseDown={(e) => { e.stopPropagation(); setDragInfo({ type: 'resize', handle: 'sw', startX: e.clientX, startY: e.clientY, initBox: { ...bbox } }); }}
                          />
                          <div 
                            className="absolute -bottom-1.5 -right-1.5 w-3 h-3 bg-white border-2 border-blue-500 rounded-full cursor-nwse-resize"
                            onMouseDown={(e) => { e.stopPropagation(); setDragInfo({ type: 'resize', handle: 'se', startX: e.clientX, startY: e.clientY, initBox: { ...bbox } }); }}
                          />
                        </>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="flex-1 bg-secondary/30 rounded-lg p-8 text-xl leading-relaxed flex items-center justify-center text-center transition-all overflow-y-auto">
                "{currentText}"
              </div>
            )}
          </div>

          {/* Bottom: Actions */}
          <div className="bg-card border border-border rounded-xl p-5 flex flex-col shrink-0">
            <h3 className="text-xs font-medium uppercase tracking-wider text-muted-foreground mb-3">
              {isImageTask ? 'BBox Category' : 'Select Label'}
            </h3>
            
            {isImageTask ? (
               <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
                <button onClick={() => handleSelectLabel('car')} className={cn("px-4 py-2.5 border rounded-lg text-left flex justify-between items-center transition-colors group", selectedLabels.includes('car') ? "border-blue-500 bg-blue-500/10 text-blue-500" : "border-border hover:border-blue-500 hover:bg-blue-500/10")}>
                  <span className="font-medium text-sm">Car</span><span className="text-[10px] border rounded px-1.5 py-0.5 border-border group-hover:border-blue-500/50">1</span>
                </button>
                <button onClick={() => handleSelectLabel('pedestrian')} className={cn("px-4 py-2.5 border rounded-lg text-left flex justify-between items-center transition-colors group", selectedLabels.includes('pedestrian') ? "border-orange-500 bg-orange-500/10 text-orange-500" : "border-border hover:border-orange-500 hover:bg-orange-500/10")}>
                  <span className="font-medium text-sm">Pedestrian</span><span className="text-[10px] border rounded px-1.5 py-0.5 border-border group-hover:border-orange-500/50">2</span>
                </button>
                <button onClick={() => handleSelectLabel('traffic_light')} className={cn("px-4 py-2.5 border rounded-lg text-left flex justify-between items-center transition-colors group", selectedLabels.includes('traffic_light') ? "border-green-500 bg-green-500/10 text-green-500" : "border-border hover:border-green-500 hover:bg-green-500/10")}>
                  <span className="font-medium text-sm">Traffic Light</span><span className="text-[10px] border rounded px-1.5 py-0.5 border-border group-hover:border-green-500/50">3</span>
                </button>
               </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
                <button onClick={() => handleSelectLabel('positive')} className={cn("px-4 py-2.5 border rounded-lg text-left flex justify-between items-center transition-colors group", selectedLabels.includes('positive') ? "border-green-500 bg-green-500/10 text-green-600 dark:text-green-400" : "border-border hover:border-green-500 hover:bg-green-500/10")}>
                  <span className="font-medium text-sm">Positive</span><span className="text-[10px] border rounded px-1.5 py-0.5 border-border group-hover:border-green-500/50">1</span>
                </button>
                <button onClick={() => handleSelectLabel('neutral')} className={cn("px-4 py-2.5 border rounded-lg text-left flex justify-between items-center transition-colors group", selectedLabels.includes('neutral') ? "border-yellow-500 bg-yellow-500/10 text-yellow-600 dark:text-yellow-400" : "border-border hover:border-yellow-500 hover:bg-yellow-500/10")}>
                  <span className="font-medium text-sm">Neutral</span><span className="text-[10px] border rounded px-1.5 py-0.5 border-border group-hover:border-yellow-500/50">2</span>
                </button>
                <button onClick={() => handleSelectLabel('negative')} className={cn("px-4 py-2.5 border rounded-lg text-left flex justify-between items-center transition-colors group", selectedLabels.includes('negative') ? "border-red-500 bg-red-500/10 text-red-600 dark:text-red-400" : "border-border hover:border-red-500 hover:bg-red-500/10")}>
                  <span className="font-medium text-sm">Negative</span><span className="text-[10px] border rounded px-1.5 py-0.5 border-border group-hover:border-red-500/50">3</span>
                </button>
              </div>
            )}

            <div className="pt-4 border-t border-border flex flex-col sm:flex-row items-center justify-between gap-4">
              <div className="flex items-center gap-6">
                {!isImageTask && (
                  <>
                    <label className="flex items-center gap-2 text-sm cursor-pointer hover:text-foreground text-muted-foreground transition-colors">
                      <input type="checkbox" checked={isMultiSelect} onChange={(e) => { setIsMultiSelect(e.target.checked); if (e.target.checked) setAutoSubmit(false); }} className="rounded border-border bg-secondary/50 text-primary focus:ring-primary w-4 h-4" />
                      Multi-select
                    </label>
                    <label className={cn("flex items-center gap-2 text-sm transition-colors", isMultiSelect ? "opacity-50 cursor-not-allowed text-muted-foreground" : "cursor-pointer hover:text-foreground text-muted-foreground")}>
                      <input type="checkbox" checked={autoSubmit} disabled={isMultiSelect} onChange={(e) => setAutoSubmit(e.target.checked)} className="rounded border-border bg-secondary/50 text-primary focus:ring-primary w-4 h-4 disabled:opacity-50" />
                      Auto-submit
                    </label>
                  </>
                )}
              </div>
              
              <div className="flex items-center gap-4 w-full sm:w-auto">
                <span className="hidden lg:inline text-xs text-muted-foreground">Use shortcuts (1, 2, 3), Enter to submit.</span>
                <button 
                  onClick={() => handleSubmit()}
                  disabled={selectedLabels.length === 0 && !isImageTask}
                  className="w-full sm:w-auto flex items-center justify-center gap-2 px-6 py-2 bg-primary hover:bg-primary/90 disabled:opacity-50 disabled:hover:bg-primary text-primary-foreground rounded-lg text-sm font-medium transition-colors shadow-sm"
                >
                  Submit <CornerDownLeft size={16} />
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : (
        /* Stats & Overview View */
        <div className="flex-1 overflow-y-auto space-y-6 pb-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Completion Card */}
            <div className="bg-card border border-border rounded-xl p-5 shadow-sm">
              <div className="flex justify-between items-start mb-4">
                <div className="w-10 h-10 rounded-lg bg-purple-500/10 flex items-center justify-center text-purple-500">
                  <CheckCircle2 size={20} />
                </div>
                <span className="text-xs font-medium px-2 py-1 bg-secondary rounded-md text-muted-foreground">Status</span>
              </div>
              <h3 className="text-2xl font-bold mb-1">{progressDisplay} / {task.total}</h3>
              <p className="text-sm text-muted-foreground">Cases Fully Labeled (All Labelers)</p>
              <div className="mt-4 pt-4 border-t border-border/50 text-xs text-muted-foreground flex justify-between">
                <span>Pending Review: {Math.floor(task.total * 0.15)}</span>
                <span>Partially Done: {Math.floor(task.total * 0.05)}</span>
              </div>
            </div>

            {/* Agreement Card */}
            <div className="bg-card border border-border rounded-xl p-5 shadow-sm">
              <div className="flex justify-between items-start mb-4">
                <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center text-blue-500">
                  <BarChart2 size={20} />
                </div>
                <span className="text-xs font-medium px-2 py-1 bg-secondary rounded-md text-muted-foreground">Consensus</span>
              </div>
              <h3 className="text-2xl font-bold mb-1 flex items-baseline gap-2">
                87.4% <span className="text-sm font-normal text-green-500">+1.2%</span>
              </h3>
              <p className="text-sm text-muted-foreground">Inter-annotator Agreement</p>
              <div className="mt-4 w-full h-1.5 bg-secondary rounded-full overflow-hidden flex">
                <div className="h-full bg-blue-500" style={{ width: '87.4%' }}></div>
                <div className="h-full bg-orange-400" style={{ width: '12.6%' }}></div>
              </div>
            </div>

            {/* Labelers Card */}
            <div className="bg-card border border-border rounded-xl p-5 shadow-sm">
              <div className="flex justify-between items-start mb-4">
                <div className="w-10 h-10 rounded-lg bg-green-500/10 flex items-center justify-center text-green-500">
                  <Users size={20} />
                </div>
                <span className="text-xs font-medium px-2 py-1 bg-secondary rounded-md text-muted-foreground">Active</span>
              </div>
              <h3 className="text-2xl font-bold mb-1">3 <span className="text-sm font-normal text-muted-foreground">Users</span></h3>
              <p className="text-sm text-muted-foreground">Working on this task</p>
              <div className="mt-4 space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="flex items-center gap-1.5"><div className="w-1.5 h-1.5 rounded-full bg-green-500"></div>Alice (You)</span>
                  <span className="font-medium text-foreground">{progressDisplay}</span>
                </div>
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span className="flex items-center gap-1.5"><div className="w-1.5 h-1.5 rounded-full bg-secondary-foreground/20"></div>Bob.Smith</span>
                  <span>{progressDisplay - 15}</span>
                </div>
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span className="flex items-center gap-1.5"><div className="w-1.5 h-1.5 rounded-full bg-secondary-foreground/20"></div>Charlie_D</span>
                  <span>{progressDisplay - 42}</span>
                </div>
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
                    <th className="px-6 py-3 border-b border-border">Consensus Label</th>
                    <th className="px-6 py-3 border-b border-border">Agreement</th>
                    <th className="px-6 py-3 border-b border-border">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {mockPreviewRows.map((row, idx) => (
                    <tr key={idx} className="border-b border-border hover:bg-secondary/20">
                      <td className="px-6 py-3 font-mono text-xs text-muted-foreground whitespace-nowrap">{row.id}</td>
                      <td className="px-6 py-3 min-w-[200px]">
                        {row.isImage ? (
                          <div className="flex items-center gap-3">
                            <div className="w-12 h-10 rounded border border-border bg-secondary overflow-hidden shrink-0 flex items-center justify-center">
                              {row.contentUrl ? (
                                <img src={row.contentUrl} alt="preview" className="w-full h-full object-cover" />
                              ) : (
                                <ImageIcon size={16} className="text-muted-foreground" />
                              )}
                            </div>
                            <span className="text-xs text-muted-foreground whitespace-nowrap">{row.bboxCount} bbox</span>
                          </div>
                        ) : (
                          <div className="line-clamp-2 max-w-sm text-sm" title={row.contentText}>
                            {row.contentText}
                          </div>
                        )}
                      </td>
                      <td className="px-6 py-3 whitespace-nowrap">
                        <span className="px-2 py-1 bg-secondary rounded-md text-xs font-medium">{row.label}</span>
                      </td>
                      <td className="px-6 py-3 whitespace-nowrap">
                        <span className={cn("text-xs font-medium", row.agreement === '100%' ? "text-green-500" : "text-orange-500")}>
                          {row.agreement}
                        </span>
                      </td>
                      <td className="px-6 py-3 whitespace-nowrap">
                        <span className={cn(
                          "px-2 py-0.5 rounded text-[10px] uppercase font-bold tracking-wider",
                          row.status === 'Consensus Reached' ? "bg-green-500/10 text-green-600" : "bg-orange-500/10 text-orange-600"
                        )}>
                          {row.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
import { useState, useEffect } from 'react';
import { Outlet, NavLink, useLocation } from 'react-router-dom';
import { Moon, Sun, Menu, X } from 'lucide-react';

const Layout = () => {
  const [isDarkMode, setIsDarkMode] = useState(() => {
    if (typeof window !== 'undefined') {
      return document.documentElement.classList.contains('dark') || 
             (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches);
    }
    return true;
  });

  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [isDarkMode]);

  useEffect(() => {
    setIsMobileMenuOpen(false);
  }, [location.pathname]);

  const toggleTheme = () => setIsDarkMode(!isDarkMode);

  const navItems = [
    { path: '/picasso', label: '💡 Picasso' },
    { path: '/demos', label: '🚀 Demos' },
    { path: '/docs', label: '📚 Docs' },
  ];

  return (
    <div className="flex flex-col h-screen w-full bg-[#F9FAFB] dark:bg-[#0a0a0c] text-zinc-600 dark:text-zinc-300 font-sans overflow-hidden transition-colors duration-300">
      {/* 顶部导航栏：使用更强的毛玻璃和极细的边框 */}
      <header className="h-16 flex-shrink-0 border-b border-black/[0.04] dark:border-white/[0.05] bg-white/60 dark:bg-zinc-950/60 backdrop-blur-2xl flex items-center justify-between px-4 sm:px-6 relative z-50">
        <NavLink to="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity">
          <img src="/icon.ico" alt="Ads Spaces Logo" className="w-7 h-7 sm:w-8 sm:h-8 object-contain" />
          <h1 className="text-lg sm:text-xl font-semibold text-zinc-900 dark:text-zinc-100 tracking-tight">Ads Spaces</h1>
        </NavLink>
        
        {/* 桌面端导航 */}
        <nav className="hidden md:flex items-center gap-1.5 p-1 rounded-full bg-black/[0.02] dark:bg-white/[0.02] border border-black/[0.03] dark:border-white/[0.02]">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `px-4 py-1.5 rounded-full text-sm font-medium transition-all duration-300 ${
                  isActive
                    ? 'bg-white dark:bg-white/10 text-zinc-900 dark:text-zinc-100 shadow-[0_2px_8px_rgba(0,0,0,0.06)] border border-black/[0.04] dark:border-white/[0.05]'
                    : 'text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200 border border-transparent'
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
          <div className="w-px h-4 bg-black/10 dark:bg-white/10 mx-2" />
          <button 
            onClick={toggleTheme}
            className="p-1.5 mr-1 text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100 hover:bg-black/5 dark:hover:bg-white/10 rounded-full transition-all"
          >
            {isDarkMode ? <Sun size={16} /> : <Moon size={16} />}
          </button>
        </nav>

        {/* 移动端菜单按钮 */}
        <div className="flex items-center gap-2 md:hidden">
          <button 
            onClick={toggleTheme}
            className="p-2 text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100 hover:bg-black/5 dark:hover:bg-white/10 rounded-full transition-all"
          >
            {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
          </button>
          <button 
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="p-2 text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100 hover:bg-black/5 dark:hover:bg-white/10 rounded-full transition-all"
          >
            {isMobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
      </header>

      {/* 移动端下拉菜单 */}
      {isMobileMenuOpen && (
        <div className="md:hidden absolute top-16 left-0 right-0 bg-white/90 dark:bg-zinc-950/90 backdrop-blur-2xl border-b border-black/[0.04] dark:border-white/[0.05] z-40 p-4 flex flex-col gap-2 shadow-2xl shadow-black/5">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `px-4 py-3 rounded-xl text-base font-medium transition-all ${
                  isActive
                    ? 'bg-white dark:bg-white/10 text-zinc-900 dark:text-zinc-100 shadow-[0_2px_10px_rgba(0,0,0,0.06)] border border-black/[0.04] dark:border-white/[0.05]'
                    : 'text-zinc-500 dark:text-zinc-400 hover:bg-black/[0.03] dark:hover:bg-white/[0.05] hover:text-zinc-900 dark:hover:text-zinc-200 border border-transparent'
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </div>
      )}

      {/* 页面内容区域 */}
      <div className="flex-1 overflow-hidden flex flex-col relative">
        <Outlet />
      </div>

      {/* 全局底部 Footer */}
      <footer className="h-auto min-h-[3rem] py-3 flex-shrink-0 border-t border-black/[0.04] dark:border-white/[0.05] bg-white/40 dark:bg-zinc-950/40 backdrop-blur-2xl flex flex-col sm:flex-row items-center justify-between px-4 sm:px-6 relative z-50 text-sm gap-3 sm:gap-0">
        <div className="flex items-center text-zinc-500 dark:text-zinc-400">
          <span className="font-medium">© AdsPlus Team</span>
        </div>
        <div className="flex items-center gap-6 text-zinc-500 dark:text-zinc-400">
          <a 
            href="https://github.com/fuliucansheng/unitorch" 
            target="_blank" 
            rel="noreferrer" 
            className="flex items-center gap-1.5 hover:text-zinc-900 dark:hover:text-zinc-200 transition-colors"
          >
            <span>🍀</span> Github
          </a>
        </div>
      </footer>
    </div>
  );
};

export default Layout;
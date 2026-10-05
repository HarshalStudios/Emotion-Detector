import React, { useState } from 'react';
import { Scan, Menu, X, ArrowRight } from 'lucide-react';

export type NavTab = 'home' | 'analyze' | 'how-it-works' | 'research' | 'history';

interface HeaderProps {
  activeTab: NavTab;
  onTabChange: (tab: NavTab) => void;
  backendConnected: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  onTabChange,
  backendConnected,
}) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems: { id: NavTab; label: string }[] = [
    { id: 'home', label: 'Home' },
    { id: 'analyze', label: 'Analyze' },
    { id: 'how-it-works', label: 'How It Works' },
    { id: 'research', label: 'Research' },
    { id: 'history', label: 'History' },
  ];

  return (
    <header className="sticky top-0 z-50 bg-slate-950/85 backdrop-blur-xl border-b border-slate-800/80">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-14 sm:h-16 flex items-center justify-between">
        
        {/* Left: Product Branding */}
        <div
          onClick={() => onTabChange('home')}
          className="flex items-center gap-3 cursor-pointer group select-none"
        >
          <div className="w-8 h-8 rounded-lg bg-cyan-950/80 border border-cyan-500/40 flex items-center justify-center text-cyan-400 group-hover:border-cyan-400 transition-colors shadow-sm">
            <Scan className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-extrabold text-sm sm:text-base tracking-tight text-white flex items-center gap-1.5">
                <span className="text-cyan-400">◉</span> EMOTION DETECTOR
              </span>
              <span className="hidden sm:inline-flex items-center px-1.5 py-0.2 rounded text-[9px] font-mono font-medium bg-slate-900 text-slate-400 border border-slate-800">
                A4 ONNX
              </span>
            </div>
            <p className="text-[10px] font-mono text-slate-400 tracking-wider hidden sm:block uppercase">
              REAL-TIME FACIAL EXPRESSION ANALYSIS
            </p>
          </div>
        </div>

        {/* Center Navigation: Desktop */}
        <nav className="hidden md:flex items-center gap-1 p-1 rounded-lg bg-slate-900/60 border border-slate-800/80">
          {navItems.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onTabChange(item.id)}
                className={`px-3.5 py-1.5 rounded-md text-xs font-medium transition-all ${
                  isActive
                    ? 'bg-cyan-400 text-slate-950 font-semibold shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                }`}
              >
                {item.label}
              </button>
            );
          })}
        </nav>

        {/* Right Actions */}
        <div className="hidden sm:flex items-center gap-3">
          {/* Backend Status indicator representing REAL health-check */}
          <div className="flex items-center gap-2 px-2.5 py-1 rounded-md bg-slate-900/80 border border-slate-800 text-[11px] font-mono text-slate-400">
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                backendConnected
                  ? 'bg-emerald-400 shadow-[0_0_6px_#34d399]'
                  : 'bg-rose-500 shadow-[0_0_6px_rgba(244,63,94,0.6)]'
              }`}
            />
            <span className={backendConnected ? 'text-emerald-300 font-semibold' : 'text-slate-400'}>
              {backendConnected ? 'FASTAPI ●' : 'FASTAPI DISCONNECTED'}
            </span>
          </div>

          {/* Launch Analyzer CTA */}
          <button
            onClick={() => onTabChange('analyze')}
            className="px-3.5 py-1.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-xs font-bold tracking-wide transition-all shadow-[0_0_12px_rgba(6,182,212,0.3)] flex items-center gap-1.5 active:scale-95"
          >
            <span>LAUNCH ANALYZER</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Mobile Hamburger Toggle */}
        <div className="flex md:hidden items-center gap-2">
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-900 border border-slate-800"
            aria-label="Toggle navigation menu"
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>

      </div>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="md:hidden border-b border-slate-800 bg-slate-950/95 px-4 pt-2 pb-4 space-y-2">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => {
                onTabChange(item.id);
                setMobileMenuOpen(false);
              }}
              className={`w-full text-left px-3.5 py-2.5 rounded-lg text-xs font-medium transition-colors ${
                activeTab === item.id
                  ? 'bg-cyan-400 text-slate-950 font-bold'
                  : 'text-slate-300 hover:bg-slate-900'
              }`}
            >
              {item.label}
            </button>
          ))}
          <div className="pt-2">
            <button
              onClick={() => {
                onTabChange('analyze');
                setMobileMenuOpen(false);
              }}
              className="w-full py-2.5 rounded-lg bg-cyan-500 text-slate-950 text-xs font-bold text-center flex items-center justify-center gap-2"
            >
              <span>LAUNCH ANALYZER</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      )}
    </header>
  );
};
